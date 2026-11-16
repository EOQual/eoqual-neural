"""Transfert par bande de fréquence et biais sur aplats (numpy seul)."""
import numpy as np
import pytest

from eoqual.neural.validation import flat_bias, transfer_by_band


def test_identity_and_scaling(scene):
    f, t = transfer_by_band(scene, scene)
    assert np.allclose(t, 1.0)
    _, t_half = transfer_by_band(scene, 0.5 * scene + 3.0)        # gain 0,5, décalage sans effet
    assert np.allclose(t_half, 0.5)


def test_lowpass_attenuates_high_frequencies(scene):
    from scipy import ndimage as ndi

    f, t = transfer_by_band(scene, ndi.gaussian_filter(scene, 1.5, mode="wrap"))
    assert t[0] > 0.8 and t[-1] < 0.05 and np.all(np.diff(t) < 0.02)


def test_uncorrelated_noise_does_not_count(scene):
    rng = np.random.default_rng(0)
    _, t = transfer_by_band(scene, scene + 0.05 * rng.standard_normal(scene.shape))
    assert np.allclose(t, 1.0, atol=0.15)


def test_flat_bias():
    noise = lambda c, rng: c + 0.01 * rng.standard_normal(c.shape)
    assert np.allclose(flat_bias(lambda y: y, [0.1, 0.5], noise), 0.0, atol=1e-3)
    assert np.allclose(flat_bias(lambda y: 1.1 * y, [0.1, 0.5], noise), 0.1, atol=2e-3)
    with pytest.raises(ValueError):
        flat_bias(lambda y: y, [0.0], noise)


def test_shape_mismatch_raises(scene):
    with pytest.raises(ValueError):
        transfer_by_band(scene, scene[:-1])
