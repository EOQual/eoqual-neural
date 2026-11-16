#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Outils PyTorch communs à tous les réseaux du paquet : import paresseux avec
message d'installation clair, choix du périphérique, cache des modèles
chargés, conversion numpy ↔ tenseur.

Ces fonctions remplacent les copies qui existaient dans eoqual-denoising
(``backends/_torch.py``) et eoqual-deconvolution (``_common_torch.py``).
"""
from __future__ import annotations

from typing import Callable

import numpy as np

__all__ = [
    "require_torch",
    "import_deepinv",
    "pick_device",
    "cached_model",
    "clear_model_cache",
    "run_on_unit_image",
]

_INSTALL = 'pip install "eoqual-neural @ git+https://github.com/EOQual/eoqual-neural.git"'

_MODELS: dict[tuple, object] = {}


def require_torch():
    """Importe et renvoie ``torch``, ou lève une ``ImportError`` explicite."""
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - dépend de l'environnement
        raise ImportError(f"Cette méthode nécessite PyTorch : {_INSTALL}") from exc
    return torch


def import_deepinv():
    """Importe ``torch`` et ``deepinv.models``, ou lève une ``ImportError`` explicite."""
    torch = require_torch()
    try:
        import deepinv.models as dm
    except ImportError as exc:  # pragma: no cover - dépend de l'environnement
        raise ImportError(f"Cette méthode nécessite deepinv : {_INSTALL}") from exc
    return torch, dm


def pick_device(device=None, allow_mps: bool = False):
    """
    Périphérique de calcul.

    Parameters
    ----------
    device : str or torch.device, optional
        Choix explicite (``"cpu"``, ``"cuda"``, ``"mps"``...), renvoyé tel
        quel.
    allow_mps : bool
        Autoriser le GPU Apple (MPS) dans le choix automatique. Désactivé
        par défaut pour les réseaux de deepinv, dont certaines opérations
        ne sont pas implémentées sur MPS ; activé pour USRNet.

    Returns
    -------
    str
        ``device`` s'il est fourni, sinon ``"cuda"`` si disponible, puis
        ``"mps"`` (si autorisé et disponible), sinon ``"cpu"``.
    """
    torch = require_torch()
    if device is not None:
        return str(device)
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if allow_mps and mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def cached_model(key: tuple, factory: Callable[[], object]):
    """
    Modèle associé à ``key``, construit une seule fois par ``factory`` puis
    gardé en mémoire (évite de recharger les poids à chaque appel). Le
    modèle est mis en mode évaluation.
    """
    if key not in _MODELS:
        model = factory()
        if hasattr(model, "eval"):
            model.eval()
        _MODELS[key] = model
    return _MODELS[key]


def clear_model_cache() -> None:
    """Vide le cache des modèles chargés (libère la mémoire, GPU compris)."""
    _MODELS.clear()


def run_on_unit_image(
    model_call: Callable, unit_image: np.ndarray, device, needs_grad: bool = False
) -> np.ndarray:
    """
    Applique ``model_call(x)`` à une image 2D dans [0, 1], convertie en
    tenseur ``(1, 1, H, W)`` ``float32``, et renvoie le résultat 2D
    ``float64``.

    ``needs_grad=True`` pour les débruiteurs qui dérivent un potentiel
    (GS-DRUNet) : le calcul se fait alors avec l'autograd activé.
    """
    torch = require_torch()
    x = torch.from_numpy(np.ascontiguousarray(unit_image, dtype=np.float32))[None, None].to(device)
    if needs_grad:
        with torch.enable_grad():
            y = model_call(x)
    else:
        with torch.no_grad():
            y = model_call(x)
    return y.detach()[0, 0].cpu().numpy().astype(np.float64)
