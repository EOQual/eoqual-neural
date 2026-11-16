#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Débruiteurs appris de DeepInverse (deepinv, BSD-3-Clause) : DRUNet,
GS-DRUNet, SwinIR, Restormer, en niveaux de gris.

Fines couches d'appel : ce paquet n'embarque ni code ni poids. Les poids
sont téléchargés par deepinv au premier appel (cache de PyTorch). Origine
et licence de chaque jeu de poids : ``THIRD_PARTY_LICENSES.md``.

**Convention.** Chaque fonction reçoit une image 2D **déjà ramenée dans
[0, 1]** et l'écart-type du bruit ``sigma`` **dans cette même échelle**,
et renvoie l'image débruitée dans [0, 1] (``float64``). La mise à
l'échelle depuis les unités du capteur, l'estimation de σ et la
stabilisation de variance relèvent d'eoqual-denoising
(``denoise(image, "drunet", sigma=…)``, ``vst_denoise``).

Références (complètes dans ``BIBLIOGRAPHY.md``) :

- DRUNet — Zhang et al. (2022), IEEE TPAMI 44(10) ;
- GS-DRUNet — Hurault, Leclaire & Papadakis (2022), ICLR ;
- SwinIR — Liang et al. (2021), ICCV Workshops ;
- Restormer — Zamir et al. (2022), CVPR ;
- bibliothèque — Tachella et al. (2025), DeepInverse, JOSS 10(115).
"""
from __future__ import annotations

import numpy as np

from ..torch_utils import cached_model, import_deepinv, pick_device, run_on_unit_image

__all__ = ["drunet", "gsdrunet", "swinir", "restormer", "SWINIR_NOISE_LEVELS"]

#: Niveaux de bruit (échelle [0, 255]) des poids SwinIR niveaux de gris publiés.
SWINIR_NOISE_LEVELS = (15, 25, 50)


def drunet(unit_image: np.ndarray, sigma: float, device=None, pretrained: str = "download") -> np.ndarray:
    """
    DRUNet : U-Net à blocs résiduels, **conditionné par σ** (une carte du
    niveau de bruit en canal d'entrée) ; un seul réseau pour toute la plage
    de bruit.

    Parameters
    ----------
    unit_image : np.ndarray
        Image 2D dans [0, 1].
    sigma : float
        Écart-type du bruit, dans l'échelle [0, 1].
    device : str, optional
        Périphérique torch ; par défaut ``"cuda"`` si disponible, sinon ``"cpu"``.
    pretrained : str
        ``"download"`` (poids deepinv) ou chemin d'un fichier ``.pth`` compatible.
    """
    _, dm = import_deepinv()
    device = pick_device(device)
    model = cached_model(
        ("drunet", device, pretrained),
        lambda: dm.DRUNet(in_channels=1, out_channels=1, pretrained=pretrained, device=device),
    )
    return run_on_unit_image(lambda x: model(x, sigma=float(sigma)), unit_image, device)


def swinir(unit_image: np.ndarray, sigma: float, device=None, noise_level: int | None = None) -> np.ndarray:
    """
    SwinIR : transformeur à fenêtres glissantes, **un jeu de poids par
    niveau de bruit** (σ = 15, 25, 50 sur [0, 255]). Le jeu le plus proche
    de ``sigma`` est choisi, sauf si ``noise_level`` est imposé ; entre deux
    niveaux, le résultat se dégrade.
    """
    _, dm = import_deepinv()
    device = pick_device(device)
    if noise_level is None:
        noise_level = min(SWINIR_NOISE_LEVELS, key=lambda lvl: abs(lvl - 255.0 * float(sigma)))
    if noise_level not in SWINIR_NOISE_LEVELS:
        raise ValueError(f"noise_level doit valoir {SWINIR_NOISE_LEVELS}, reçu {noise_level}.")
    model = cached_model(
        ("swinir", device, noise_level),
        lambda: dm.SwinIR(in_chans=1, pretrained="download", pretrained_noise_level=noise_level).to(device),
    )
    return run_on_unit_image(lambda x: model(x), unit_image, device)


def restormer(unit_image: np.ndarray, sigma: float | None = None, device=None) -> np.ndarray:
    """
    Restormer : transformeur à attention entre canaux, poids **aveugles**
    (entraînés sur σ ∈ [0, 50]/255 sans connaître σ). ``sigma`` est ignoré
    (accepté pour garder la signature commune) ; hors de ce domaine,
    notamment à faible flux, le réseau échoue sans le signaler.
    """
    _, dm = import_deepinv()
    device = pick_device(device)
    model = cached_model(
        ("restormer", device),
        lambda: dm.Restormer(in_channels=1, out_channels=1, pretrained="denoising", device=device),
    )
    return run_on_unit_image(lambda x: model(x), unit_image, device)


def gsdrunet(unit_image: np.ndarray, sigma: float, device=None) -> np.ndarray:
    """
    GS-DRUNet : le débruiteur est le **pas de gradient d'un potentiel
    appris**, ``D(x) = x − ∇g(x)``, ce qui garantit la convergence d'un
    schéma Plug-and-Play qui l'emploie. Environ 0,2 dB sous DRUNet utilisé
    seul. Le calcul passe par l'autograd.
    """
    _, dm = import_deepinv()
    device = pick_device(device)
    model = cached_model(("gsdrunet", device), lambda: _load_gsdrunet(dm, device))
    return run_on_unit_image(lambda x: model(x, sigma=float(sigma)), unit_image, device, needs_grad=True)


_GSDRUNET_WEIGHTS = "GSDRUNet_grayscale_torch.ckpt"


def _load_gsdrunet(dm, device):
    """
    Construit GS-DRUNet niveaux de gris et charge ses poids deepinv.

    Contournement : le fichier de poids publié contient, en plus des poids
    du réseau, des compteurs de métriques d'entraînement
    (``*_PSNR.data_range``) que ``GSDRUNet(pretrained="download")`` refuse
    en chargement strict (constaté avec deepinv 0.4.2). On ne garde que les
    clés du réseau, et on vérifie qu'aucun poids ne manque.
    """
    import torch
    from deepinv.models.utils import get_weights_url

    model = dm.GSDRUNet(in_channels=1, out_channels=1, pretrained=None, device=device)
    url = get_weights_url(model_name="gradientstep", file_name=_GSDRUNET_WEIGHTS)
    ckpt = torch.hub.load_state_dict_from_url(url, map_location="cpu", file_name=_GSDRUNET_WEIGHTS)
    ckpt = ckpt.get("state_dict", ckpt)
    expected = model.state_dict().keys()
    missing = set(expected) - set(ckpt)
    if missing:
        raise RuntimeError(f"Poids GS-DRUNet incomplets, clés manquantes : {sorted(missing)[:5]}...")
    model.load_state_dict({k: v for k, v in ckpt.items() if k in expected}, strict=True)
    return model.to(device)
