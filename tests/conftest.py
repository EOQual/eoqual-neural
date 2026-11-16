"""Fixtures communes."""
import os

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _weights_cache(tmp_path_factory, monkeypatch):
    # cache des poids partagé entre tests (variable d'environnement respectée si définie)
    if "EOQUAL_WEIGHTS_DIR" not in os.environ:
        monkeypatch.setenv("EOQUAL_WEIGHTS_DIR", str(tmp_path_factory.getbasetemp() / "weights"))


@pytest.fixture
def scene():
    """Scène lisse par morceaux 128 × 128 dans [0, 1] (sans téléchargement)."""
    y, x = np.mgrid[0:128, 0:128] / 128.0
    img = 0.3 + 0.4 * ((x > 0.4) & (y < 0.7)) + 0.2 * np.sin(2 * np.pi * 6 * x) * (y > 0.5)
    return np.clip(img, 0, 1)


def gaussian_psf(sigma=1.2, size=9):
    r = np.arange(size) - size // 2
    k = np.exp(-(r[:, None] ** 2 + r[None, :] ** 2) / (2 * sigma**2))
    return k / k.sum()


def circular_blur(img, k):
    """Flou circulaire, PSF centrée en (h//2, w//2)."""
    pad = np.zeros_like(img)
    pad[: k.shape[0], : k.shape[1]] = k
    pad = np.roll(pad, (-(k.shape[0] // 2), -(k.shape[1] // 2)), axis=(0, 1))
    return np.fft.ifft2(np.fft.fft2(img) * np.fft.fft2(pad)).real
