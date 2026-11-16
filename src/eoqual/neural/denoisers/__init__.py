"""
Rôle 1 — le débruiteur appris.

Un réseau appris à retirer un bruit blanc gaussien. Il s'emploie seul
(eoqual-denoising) ou comme a priori dans un algorithme qui porte une autre
tâche : Plug-and-Play de déconvolution (eoqual-deconvolution), de
dématriçage, de super-résolution.

:func:`get_denoiser` renvoie une fonction ``f(unit_image, sigma, **kw)``
qui travaille dans l'échelle [0, 1] (voir :mod:`.deepinv_nets`).
"""
from __future__ import annotations

from typing import Callable

from .deepinv_nets import SWINIR_NOISE_LEVELS, drunet, gsdrunet, restormer, swinir

__all__ = ["DENOISERS", "get_denoiser", "list_denoisers", "drunet", "gsdrunet", "swinir", "restormer",
           "SWINIR_NOISE_LEVELS"]

#: Débruiteurs appris : fonction et façon dont le réseau connaît σ.
DENOISERS = {
    "drunet": {"function": drunet, "sigma": "conditionné (carte de σ en entrée)"},
    "gsdrunet": {"function": gsdrunet, "sigma": "conditionné ; pas de gradient d'un potentiel (PnP convergent)"},
    "swinir": {"function": swinir, "sigma": "un jeu de poids par niveau (15, 25, 50 sur [0, 255])"},
    "restormer": {"function": restormer, "sigma": "aveugle (σ ignoré)"},
}


def list_denoisers() -> list[str]:
    """Noms des débruiteurs appris disponibles."""
    return sorted(DENOISERS)


def get_denoiser(name: str) -> Callable:
    """
    Fonction ``f(unit_image, sigma, **kw)`` du débruiteur ``name``
    (image et σ dans l'échelle [0, 1]).

    Raises
    ------
    ValueError
        Nom inconnu.
    """
    if name not in DENOISERS:
        raise ValueError(f"Débruiteur appris inconnu {name!r} : un de {list_denoisers()}.")
    return DENOISERS[name]["function"]
