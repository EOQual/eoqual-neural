#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Boucle d'apprentissage et d'ajustement fin d'USRNet, indépendante de la
tâche.

Les paires d'entraînement sont fournies par un **tireur** (objet muni d'une
méthode ``batch(n, device)`` qui renvoie ``(y, x, k, s)`` : images
dégradées ``N×3×H×W`` dans [0, 1], images nettes, PSF ``N×1×h×w``, σ
``N×1×1×1``). Le tireur porte la physique de la tâche — lois des PSF
d'instrument et du bruit, traitement des bords — et relève donc du module
de la tâche : ``eoqual.deconvolution.training`` pour la déconvolution, qui
fournit aussi ``train_usrnet(images_entraînement, images_validation, cfg)``.

Produits dans ``cfg.out_dir`` : ``best.pth`` et ``last.pth`` (poids,
architecture, réglages, historique), ``history.csv`` (itération, perte,
PSNR de validation), ``config.json``, ``best.sha256``.
"""
from __future__ import annotations

import csv
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from ..torch_utils import require_torch

torch = require_torch()

__all__ = ["TrainingConfig", "build_usrnet", "evaluate_usrnet", "fit_usrnet"]


@dataclass
class TrainingConfig:
    """
    Réglages d'entraînement (valeurs par défaut : ajustement fin depuis les
    poids publiés ; pour un entraînement complet, ``init_weights=None``,
    ``iterations`` ~ 4·10⁵ et ``lr`` 1e-4 comme KAIR).
    """

    init_weights: str | None = "usrnet"
    arch: dict | None = None                  # si init_weights=None : {"n_iter", "h_nc", "nc", "nb"}
    iterations: int = 20000
    batch_size: int = 8
    lr: float = 5e-5
    milestones: tuple = (0.5, 0.75)           # fractions des itérations où lr est divisé par 2
    loss: str = "l1"
    seed: int = 0
    device: str | None = None
    val_every: int = 1000
    val_samples: int = 32
    out_dir: str = "usrnet_training"


def build_usrnet(cfg: TrainingConfig, device):
    """
    USRNet à entraîner : architecture et poids de départ de ``cfg``.

    Returns
    -------
    (model, arch)
        Réseau sur ``device`` et son architecture (dictionnaire sérialisable).
    """
    from ..weights import KNOWN_WEIGHTS, resolve_weights
    from .usrnet import USRNet, select_device

    if cfg.init_weights is not None:
        state = torch.load(resolve_weights(cfg.init_weights), map_location="cpu", weights_only=True)
        arch = cfg.arch
        if isinstance(state, dict) and "state_dict" in state:
            arch, state = arch or state.get("arch"), state["state_dict"]
        arch = arch or KNOWN_WEIGHTS.get(str(cfg.init_weights), {}).get("arch") or KNOWN_WEIGHTS["usrnet"]["arch"]
    else:
        arch, state = cfg.arch or KNOWN_WEIGHTS["usrnet"]["arch"], None
    model = USRNet(n_iter=arch["n_iter"], h_nc=arch["h_nc"], nc=tuple(arch["nc"]), nb=arch["nb"])
    if state is not None:
        model.load_state_dict(state, strict=True)
    return model.to(select_device(device)), {k: (list(v) if isinstance(v, tuple) else v) for k, v in arch.items()}


def evaluate_usrnet(model, sampler, n: int, device) -> float:
    """PSNR moyen (dB, dynamique 1) sur ``n`` paires du tireur de validation."""
    model.eval()
    vals = []
    with torch.no_grad():
        for _ in range(n):
            y, x, k, s = sampler.batch(1, device)
            out = model(y, k, 1, s)[..., : x.shape[-2], : x.shape[-1]].mean(1)
            vals.append(float(-10 * torch.log10(torch.mean((out - x[:, 0]) ** 2))))
    model.train()
    return float(np.mean(vals))


def fit_usrnet(train_sampler, make_val_sampler: Callable[[], object], cfg: TrainingConfig | None = None,
               log=print) -> Path:
    """
    Entraîne (ou ajuste finement) USRNet et renvoie le chemin du meilleur
    point de contrôle (PSNR de validation maximal).

    Parameters
    ----------
    train_sampler :
        Tireur des paires d'entraînement (``batch(n, device)``).
    make_val_sampler : callable
        Fabrique un tireur de validation **à tirage fixe** (rappelée à
        chaque validation, pour comparer les itérations sur les mêmes
        paires). Les scènes de validation doivent être disjointes de
        celles de l'entraînement.
    cfg : TrainingConfig, optional
        Réglages ; une sous-classe peut ajouter les réglages du tireur, qui
        sont alors enregistrés dans ``config.json``.
    """
    from ..weights import sha256
    from .usrnet import select_device

    cfg = cfg or TrainingConfig()
    torch.manual_seed(cfg.seed)
    device = select_device(cfg.device)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    model, arch = build_usrnet(cfg, device)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt, [int(m * cfg.iterations) for m in cfg.milestones], 0.5)
    lossf = torch.nn.L1Loss() if cfg.loss == "l1" else torch.nn.MSELoss()
    conf = asdict(cfg)
    (out / "config.json").write_text(json.dumps(conf, indent=1, default=str))

    def val_psnr():
        return evaluate_usrnet(model, make_val_sampler(), cfg.val_samples, device)

    best = val_psnr()
    history = [(0, float("nan"), best)]
    log(f"itération 0 : PSNR de validation {best:.2f} dB (poids de départ)")
    save = lambda path: torch.save({"state_dict": model.state_dict(), "arch": arch, "config": conf,
                                    "history": history}, path)
    save(out / "best.pth")
    running, t0 = [], time.time()
    model.train()
    for it in range(1, cfg.iterations + 1):
        y, x, k, s = train_sampler.batch(cfg.batch_size, device)
        P = x.shape[-1]
        loss = lossf(model(y, k, 1, s)[..., :P, :P], x)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        running.append(loss.item())
        if it % cfg.val_every == 0 or it == cfg.iterations:
            v = val_psnr()
            history.append((it, float(np.mean(running)), v))
            log(f"itération {it} : perte {np.mean(running):.4f}, PSNR de validation {v:.2f} dB "
                f"({time.time() - t0:.0f} s)")
            running = []
            if v > best:
                best = v
                save(out / "best.pth")
    save(out / "last.pth")
    with open(out / "history.csv", "w", newline="") as f:
        csv.writer(f).writerows([("iteration", "loss", "val_psnr_db"), *history])
    (out / "best.sha256").write_text(sha256(out / "best.pth") + "  best.pth\n")
    return out / "best.pth"
