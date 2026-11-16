#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Démarrage rapide d'eoqual-neural : un débruiteur appris (rôle 1), le réseau
déplié USRNet (rôle 2), et la mesure de ce que chacun transmet de la scène.

Télécharge au premier appel les poids DRUNet (deepinv) et USRNet réduit
(KAIR). Scène synthétique, sans autre téléchargement.

Usage ::

    python examples/quickstart.py
"""
import numpy as np

from eoqual.neural import get_denoiser, transfer_by_band
from eoqual.neural.unrolled.usrnet import usrnet_restore


def scene(n=192):
    y, x = np.mgrid[0:n, 0:n] / n
    img = 0.3 + 0.4 * ((x > 0.4) & (y < 0.7)) + 0.15 * np.sin(2 * np.pi * 9 * x) * (y > 0.5)
    return np.clip(img, 0, 1)


def gaussian_psf(sigma=1.6, size=15):
    r = np.arange(size) - size // 2
    k = np.exp(-(r[:, None] ** 2 + r[None, :] ** 2) / (2 * sigma**2))
    return k / k.sum()


def circular_blur(img, k):
    pad = np.zeros_like(img)
    pad[: k.shape[0], : k.shape[1]] = k
    pad = np.roll(pad, (-(k.shape[0] // 2), -(k.shape[1] // 2)), axis=(0, 1))
    return np.fft.ifft2(np.fft.fft2(img) * np.fft.fft2(pad)).real


def psnr(a, ref):
    return 10 * np.log10(1 / np.mean((a - ref) ** 2))


def main():
    rng = np.random.default_rng(0)
    x = scene()

    # Rôle 1 : débruiteur appris (image et σ dans [0, 1])
    sigma = 0.05
    noisy = x + sigma * rng.standard_normal(x.shape)
    den = get_denoiser("drunet")(noisy, sigma, device="cpu")
    print(f"DRUNet : {psnr(noisy, x):.1f} dB → {psnr(den, x):.1f} dB")

    # Rôle 2 : réseau déplié, domaine périodique (flou circulaire)
    k = gaussian_psf()
    blurred = circular_blur(x, k) + 0.01 * rng.standard_normal(x.shape)
    rest = usrnet_restore(blurred, k, 0.01, weights="usrnet_tiny", device="cpu")
    print(f"USRNet (réduit) : {psnr(blurred, x):.1f} dB → {psnr(rest, x):.1f} dB")

    # Validation : transfert par bande de fréquence
    f, t_den = transfer_by_band(x, den)
    _, t_rest = transfer_by_band(x, rest)
    print("fréquence (c/px) :", " ".join(f"{v:5.2f}" for v in f))
    print("transfert DRUNet :", " ".join(f"{v:5.2f}" for v in t_den))
    print("transfert USRNet :", " ".join(f"{v:5.2f}" for v in t_rest))


if __name__ == "__main__":
    main()
