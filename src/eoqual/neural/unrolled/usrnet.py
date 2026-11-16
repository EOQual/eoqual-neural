#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
USRNet — réseau déplié (Zhang, Van Gool & Timofte, CVPR 2020) : la PSF et le
niveau de bruit sont des **entrées** du réseau.

K. Zhang, L. Van Gool, R. Timofte, « Deep unfolding network for image
super-resolution », CVPR 2020.

Modèle : ``y = (k ∗ x)↓s + n``. Le problème MAP
``min ½‖y − (k ∗ x)↓s‖² + λΦ(x)`` est résolu par éclatement
semi-quadratique (HQS) déroulé sur ``n_iter`` itérations :

- **module de données** (sans paramètre appris) : solution exacte, en
  Fourier, de ``min ‖y − (k ∗ z)↓s‖² + α_k‖z − x_{k−1}‖²``. Pour ``s = 1``
  (déconvolution) : ``Z = (H* Y + α_k X) / (|H|² + α_k)`` — un filtre de
  Wiener–Tikhonov rappelé vers l'estimée courante ;
- **module d'a priori** : un ResUNet débruiteur conditionné par β_k (carte
  constante concaténée à l'image) ;
- **module d'hyperparamètres** : un petit MLP qui donne ``(α_k, β_k)``
  pour chaque itération à partir de ``(σ, s)``.

Ce module ne contient que le réseau : :func:`usrnet_restore` travaille sur
un domaine périodique déjà préparé, dans l'échelle [0, 1]. Le traitement
des bords, des pixels masqués ou saturés et la mise à l'échelle relèvent du
module de la tâche (``eoqual-deconvolution``, méthode ``"usrnet"``).

Licence : architecture et poids publiés de KAIR (© 2019 Kai Zhang, MIT,
https://github.com/cszn/KAIR) ; le code ci-dessous en reprend la structure
et les noms de paramètres (pour charger ces poids), réécrit pour l'API FFT
complexe actuelle de PyTorch. Voir THIRD_PARTY_LICENSES.md.
"""
from __future__ import annotations

import numpy as np

from ..torch_utils import pick_device, require_torch

torch = require_torch()
nn = torch.nn

__all__ = ["USRNet", "ResUNet", "HyPaNet", "load_usrnet", "usrnet_restore", "p2o", "data_step",
           "select_device", "learned_schedule"]

# ---------------------------------------------------------------------------
# Architecture (noms de paramètres identiques à KAIR)
# ---------------------------------------------------------------------------

def _seq(*mods):
    """``nn.Sequential`` aplati ; un seul module est renvoyé tel quel (comme KAIR)."""
    flat = []
    for m in mods:
        flat.extend(m.children() if isinstance(m, nn.Sequential) else [m])
    return flat[0] if len(flat) == 1 else nn.Sequential(*flat)


class _ResBlock(nn.Module):
    def __init__(self, nc: int):
        super().__init__()
        self.res = nn.Sequential(nn.Conv2d(nc, nc, 3, 1, 1, bias=False), nn.ReLU(inplace=True),
                                 nn.Conv2d(nc, nc, 3, 1, 1, bias=False))

    def forward(self, x):
        return x + self.res(x)


class ResUNet(nn.Module):
    """Débruiteur U-Net résiduel (module d'a priori)."""

    def __init__(self, in_nc=4, out_nc=3, nc=(64, 128, 256, 512), nb=2):
        super().__init__()
        rb = lambda c: [_ResBlock(c) for _ in range(nb)]
        down = lambda a, b: nn.Conv2d(a, b, 2, 2, 0, bias=False)
        up = lambda a, b: nn.ConvTranspose2d(a, b, 2, 2, 0, bias=False)
        self.m_head = nn.Conv2d(in_nc, nc[0], 3, 1, 1, bias=False)
        self.m_down1 = _seq(*rb(nc[0]), down(nc[0], nc[1]))
        self.m_down2 = _seq(*rb(nc[1]), down(nc[1], nc[2]))
        self.m_down3 = _seq(*rb(nc[2]), down(nc[2], nc[3]))
        self.m_body = _seq(*rb(nc[3]))
        self.m_up3 = _seq(up(nc[3], nc[2]), *rb(nc[2]))
        self.m_up2 = _seq(up(nc[2], nc[1]), *rb(nc[1]))
        self.m_up1 = _seq(up(nc[1], nc[0]), *rb(nc[0]))
        self.m_tail = nn.Conv2d(nc[0], out_nc, 3, 1, 1, bias=False)

    def forward(self, x):
        h, w = x.shape[-2:]
        x = nn.functional.pad(x, (0, int(np.ceil(w / 8) * 8 - w), 0, int(np.ceil(h / 8) * 8 - h)), mode="replicate")
        x1 = self.m_head(x)
        x2 = self.m_down1(x1)
        x3 = self.m_down2(x2)
        x4 = self.m_down3(x3)
        x = self.m_body(x4)
        x = self.m_up3(x + x4)
        x = self.m_up2(x + x3)
        x = self.m_up1(x + x2)
        x = self.m_tail(x + x1)
        return x[..., :h, :w]


class HyPaNet(nn.Module):
    """``(σ, s) ↦ (α_1..α_K, β_1..β_K)``, positifs (Softplus)."""

    def __init__(self, in_nc=2, out_nc=16, channel=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Conv2d(in_nc, channel, 1), nn.ReLU(inplace=True),
                                 nn.Conv2d(channel, channel, 1), nn.ReLU(inplace=True),
                                 nn.Conv2d(channel, out_nc, 1), nn.Softplus())

    def forward(self, x):
        return self.mlp(x) + 1e-6


def p2o(psf, shape):
    """OTF complexe ``N×C×H×W`` d'une PSF ``N×C×h×w`` centrée en ``(h//2, w//2)`` (convention du paquet)."""
    otf = torch.zeros(psf.shape[:-2] + tuple(shape), dtype=psf.dtype, device=psf.device)
    otf[..., : psf.shape[-2], : psf.shape[-1]] = psf
    otf = torch.roll(otf, shifts=(-(psf.shape[-2] // 2), -(psf.shape[-1] // 2)), dims=(-2, -1))
    return torch.fft.fft2(otf)


def _splits(a, sf):
    """Blocs ``sf × sf`` distincts du plan de Fourier, empilés sur un dernier axe."""
    b = torch.stack(torch.chunk(a, sf, dim=2), dim=-1)
    return torch.cat(torch.chunk(b, sf, dim=3), dim=-1)


def data_step(x, FB, FBC, F2B, FBFy, alpha, sf):
    """
    Module de données : ``argmin_z ‖y − (k ∗ z)↓s‖² + α‖z − x‖²`` en Fourier
    (Zhao et al. 2016, formule fermée). Pour ``s = 1`` :
    ``Z = (H* Y + α X) / (|H|² + α)``.
    """
    FR = FBFy + torch.fft.fft2(alpha * x)
    x1 = FB * FR
    FBR = torch.mean(_splits(x1, sf), dim=-1)
    invW = torch.mean(_splits(F2B, sf), dim=-1)
    invWBR = FBR / (invW + alpha)
    FCBinvWBR = FBC * invWBR.repeat(1, 1, sf, sf)
    FX = (FR - FCBinvWBR) / alpha
    return torch.fft.ifft2(FX).real


class USRNet(nn.Module):
    """
    Réseau déplié : ``forward(y, k, sf, sigma)`` avec ``y`` ``N×3×H×W`` dans
    [0, 1], ``k`` ``N×1×h×w`` (somme 1), ``sigma`` ``N×1×1×1`` (écart-type du
    bruit dans les unités de ``y``), ``sf`` facteur de super-résolution
    (1 : déconvolution).
    """

    def __init__(self, n_iter=8, h_nc=64, in_nc=4, out_nc=3, nc=(64, 128, 256, 512), nb=2):
        super().__init__()
        self.p = ResUNet(in_nc=in_nc, out_nc=out_nc, nc=nc, nb=nb)
        self.h = HyPaNet(in_nc=2, out_nc=n_iter * 2, channel=h_nc)
        self.n = n_iter

    def forward(self, y, k, sf, sigma):
        H, W = y.shape[-2:]
        FB = p2o(k, (H * sf, W * sf))
        FBC = torch.conj(FB)
        F2B = (FB.real**2 + FB.imag**2).to(FB.dtype)
        STy = torch.zeros(y.shape[0], y.shape[1], H * sf, W * sf, dtype=y.dtype, device=y.device)
        STy[..., ::sf, ::sf] = y
        FBFy = FBC * torch.fft.fft2(STy)
        x = nn.functional.interpolate(y, scale_factor=sf, mode="nearest") if sf > 1 else y
        ab = self.h(torch.cat((sigma, torch.full_like(sigma, float(sf))), dim=1))
        for i in range(self.n):
            x = data_step(x, FB, FBC, F2B, FBFy, ab[:, i : i + 1], sf)
            beta = ab[:, i + self.n : i + self.n + 1].expand(-1, -1, x.shape[2], x.shape[3])
            x = self.p(torch.cat((x, beta), dim=1))
        return x



# ---------------------------------------------------------------------------
# Chargement et inférence
# ---------------------------------------------------------------------------

_MODELS: dict = {}


def select_device(device=None):
    """``device`` explicite, sinon CUDA, puis MPS (GPU Apple), puis CPU."""
    return torch.device(pick_device(device, allow_mps=True))


def load_usrnet(weights="usrnet", device=None, arch: dict | None = None) -> USRNet:
    """
    USRNet en mode évaluation, poids chargés strictement (gardé en cache).

    Parameters
    ----------
    weights : str or path
        ``"usrnet"`` (défaut, 17 M de paramètres), ``"usrnet_tiny"``
        (0,6 M), ou chemin d'un fichier de poids (réentraînement,
        :mod:`eoqual.neural.unrolled.training`) ; un point de contrôle
        produit par l'entraînement contient aussi son architecture.
    arch : dict, optional
        Architecture (``n_iter``, ``h_nc``, ``nc``, ``nb``) d'un fichier de
        poids qui ne la contient pas.
    """
    from ..weights import KNOWN_WEIGHTS, resolve_weights

    dev = select_device(device)
    key = (str(weights), str(dev))
    if key in _MODELS:
        return _MODELS[key]
    path = resolve_weights(weights)
    state = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(state, dict) and "state_dict" in state:
        arch = arch or state.get("arch")
        state = state["state_dict"]
    arch = arch or KNOWN_WEIGHTS.get(str(weights), {}).get("arch") or KNOWN_WEIGHTS["usrnet"]["arch"]
    model = USRNet(n_iter=arch["n_iter"], h_nc=arch["h_nc"], nc=tuple(arch["nc"]), nb=arch["nb"])
    model.load_state_dict(state, strict=True)
    model.eval().to(dev)
    for p in model.parameters():
        p.requires_grad_(False)
    _MODELS[key] = model
    return model


def learned_schedule(model: USRNet, sigma: float, sf: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Calendrier appris par HyPaNet pour un niveau de bruit ``sigma``
    (échelle [0, 1]) : poids ``α_k`` du module de données et forces
    ``β_k`` du débruitage, pour les ``n_iter`` étapes.
    """
    dev = next(model.parameters()).device
    with torch.no_grad():
        ab = model.h(torch.tensor([[[[float(sigma)]], [[float(sf)]]]], dtype=torch.float32, device=dev))
    ab = ab.cpu().numpy().ravel()
    return ab[: model.n], ab[model.n :]


def usrnet_restore(y_unit: np.ndarray, psf: np.ndarray, sigma: float, weights="usrnet", device=None,
                   model: USRNet | None = None, callback=None) -> np.ndarray:
    """
    Déconvolution par USRNet (``s = 1``) d'une image 2D **déjà préparée** :
    domaine périodique cohérent avec la PSF (flou circulaire), dans
    l'échelle [0, 1].

    Le réseau, en couleur, reçoit l'image répliquée sur trois canaux ; la
    sortie est leur moyenne.

    Parameters
    ----------
    y_unit : np.ndarray
        Image floue et bruitée, 2D, dans [0, 1].
    psf : np.ndarray
        PSF centrée en ``(h//2, w//2)``, de somme 1.
    sigma : float
        Écart-type du bruit dans l'échelle [0, 1] ; les poids publiés
        couvrent σ de 0 à 25/255.
    weights : str or path
        Voir :func:`load_usrnet` ; ignoré si ``model`` est fourni.
    model : USRNet, optional
        Réseau déjà chargé (par exemple en cours d'entraînement).
    callback : callable, optional
        ``callback(k, x)`` après chaque étape, ``x`` dans [0, 1].

    Returns
    -------
    np.ndarray
        Image restaurée sur le même domaine, échelle [0, 1], ``float64``.
    """
    model = model if model is not None else load_usrnet(weights, device)
    dev = next(model.parameters()).device
    f32 = lambda a: torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev)
    with torch.no_grad():
        y_t = f32(y_unit)[None, None].repeat(1, 3, 1, 1)
        FB = p2o(f32(psf)[None, None], y_unit.shape)
        FBC = torch.conj(FB)
        F2B = (FB.real**2 + FB.imag**2).to(FB.dtype)
        FBFy = FBC * torch.fft.fft2(y_t)
        sig_t = torch.full((1, 1, 1, 1), float(sigma), dtype=torch.float32, device=dev)
        ab = model.h(torch.cat((sig_t, torch.ones_like(sig_t)), dim=1))
        x = y_t
        for i in range(model.n):
            x = data_step(x, FB, FBC, F2B, FBFy, ab[:, i : i + 1], 1)
            beta = ab[:, i + model.n : i + model.n + 1].expand(-1, -1, x.shape[2], x.shape[3])
            x = model.p(torch.cat((x, beta), dim=1))
            if callback is not None:
                callback(i, x.mean(dim=1)[0].float().cpu().numpy().astype(np.float64))
        return x.mean(dim=1)[0].float().cpu().numpy().astype(np.float64)
