#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Outils de validation d'une restauration non linéaire (réseau ou méthode non
locale), indépendants du réseau.

Une restauration non linéaire n'a pas de MTF : elle transmet différemment
un bord franc et une texture fine. La MTF mesurée sur un bord incliné après
restauration ne vaut que pour des bords. :func:`transfer_by_band` mesure ce
qui est transmis d'une scène quelconque, bande de fréquence par bande, et
:func:`flat_bias` le biais radiométrique sur des aplats.

numpy seul : ces fonctions s'emploient aussi sans PyTorch.
"""
from __future__ import annotations

from typing import Callable

import numpy as np

__all__ = ["transfer_by_band", "flat_bias"]


def transfer_by_band(truth: np.ndarray, output: np.ndarray, frequencies=None,
                     width: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """
    Transfert d'une restauration par anneau de fréquence :
    ``Re⟨O, T⟩ / ⟨T, T⟩``, où ``T`` et ``O`` sont les spectres de la vérité
    et de la sortie (moyennes retirées).

    1 : contenu de la vérité transmis intact ; < 1 : atténué ; > 1 :
    amplifié au-delà de la vérité (sur-netteté, détail ajouté corrélé à la
    scène). L'énergie de la sortie **non corrélée** à la vérité (bruit
    résiduel, détail inventé) n'entre pas dans ce rapport.

    Parameters
    ----------
    truth, output : np.ndarray
        Images 2D de même forme.
    frequencies : array_like, optional
        Centres des anneaux, en cycles par pixel ; par défaut 0,05 à 0,45
        par pas de 0,05.
    width : float
        Largeur des anneaux (cycles par pixel).

    Returns
    -------
    (frequencies, transfer) : tuple of np.ndarray
    """
    truth = np.asarray(truth, dtype=np.float64)
    output = np.asarray(output, dtype=np.float64)
    if truth.shape != output.shape or truth.ndim != 2:
        raise ValueError("truth et output doivent être deux images 2D de même forme.")
    f = np.arange(0.05, 0.451, 0.05) if frequencies is None else np.asarray(frequencies, dtype=np.float64)
    T = np.fft.fft2(truth - truth.mean())
    O = np.fft.fft2(output - output.mean())
    r = np.hypot(*np.meshgrid(np.fft.fftfreq(truth.shape[1]), np.fft.fftfreq(truth.shape[0])))
    cross = np.real(O * np.conj(T))
    power = np.abs(T) ** 2
    out = np.full(f.shape, np.nan)
    for i, fc in enumerate(f):
        m = (r > fc - width / 2) & (r <= fc + width / 2)
        if power[m].sum() > 0:
            out[i] = cross[m].sum() / power[m].sum()
    return f, out


def flat_bias(restore: Callable[[np.ndarray], np.ndarray], levels, noise: Callable[[np.ndarray, np.random.Generator], np.ndarray],
              shape=(128, 128), seed: int = 0) -> np.ndarray:
    """
    Biais relatif d'une restauration sur des aplats : ``mean(restore(y)) / L − 1``
    pour chaque niveau ``L``.

    Parameters
    ----------
    restore : callable
        ``restore(y) -> x`` (image 2D bruitée vers image restaurée).
    levels : array_like
        Niveaux des aplats (unités de l'image), non nuls.
    noise : callable
        ``noise(clean, rng) -> noisy`` : modèle de bruit du capteur.
    shape : tuple
        Taille des aplats.
    seed : int
        Graine du tirage.

    Returns
    -------
    np.ndarray
        Biais relatif par niveau (0 : aucun biais).
    """
    rng = np.random.default_rng(seed)
    bias = []
    for L in np.asarray(levels, dtype=np.float64):
        if L == 0:
            raise ValueError("Les niveaux doivent être non nuls (biais relatif).")
        y = noise(np.full(shape, L), rng)
        bias.append(float(np.mean(restore(y)) / L - 1.0))
    return np.array(bias)
