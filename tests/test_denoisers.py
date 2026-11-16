"""Débruiteurs appris : registre, et appel réel (poids téléchargés par deepinv)."""
import numpy as np
import pytest

from eoqual.neural import get_denoiser, list_denoisers


def test_registry():
    assert list_denoisers() == ["drunet", "gsdrunet", "restormer", "swinir"]
    with pytest.raises(ValueError):
        get_denoiser("nl_bayes")                      # méthode classique : eoqual-denoising


@pytest.mark.weights
def test_drunet_denoises(scene):
    pytest.importorskip("deepinv")
    rng = np.random.default_rng(0)
    noisy = scene + 0.05 * rng.standard_normal(scene.shape)
    out = get_denoiser("drunet")(noisy, 0.05, device="cpu")
    mse = lambda a: np.mean((a - scene) ** 2)
    assert out.shape == scene.shape and out.dtype == np.float64
    assert mse(out) < 0.2 * mse(noisy)


@pytest.mark.weights
def test_swinir_weight_choice(scene):
    pytest.importorskip("deepinv")
    from eoqual.neural.denoisers import swinir

    with pytest.raises(ValueError):
        swinir(scene, 0.05, device="cpu", noise_level=20)
