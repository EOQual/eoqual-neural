"""USRNet : module de données, réseau, inférence sur domaine préparé, apprentissage."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from conftest import circular_blur, gaussian_psf  # noqa: E402
from eoqual.neural.unrolled.training import TrainingConfig, fit_usrnet  # noqa: E402
from eoqual.neural.unrolled.usrnet import (  # noqa: E402
    USRNet, data_step, learned_schedule, load_usrnet, p2o, usrnet_restore,
)

TINY = {"n_iter": 2, "h_nc": 8, "nc": [8, 8, 8, 8], "nb": 1}


def _t(a):
    return torch.as_tensor(a, dtype=torch.float64)[None, None]


def _otf(k, shape):
    pad = np.zeros(shape)
    pad[: k.shape[0], : k.shape[1]] = k
    return np.fft.fft2(np.roll(pad, (-(k.shape[0] // 2), -(k.shape[1] // 2)), axis=(0, 1)))


def test_p2o_matches_numpy_otf():
    rng = np.random.default_rng(0)
    k = rng.random((7, 9))
    np.testing.assert_allclose(p2o(_t(k), (40, 36))[0, 0].numpy(), _otf(k, (40, 36)), atol=1e-12)


def test_data_step_is_wiener_tikhonov():
    # s = 1 : Z = (H* Y + α X) / (|H|² + α)
    rng = np.random.default_rng(0)
    y, x, k, a = rng.random((32, 40)), rng.random((32, 40)), gaussian_psf(1.5, 9), 0.03
    H = _otf(k, y.shape)
    ref = np.fft.ifft2((np.conj(H) * np.fft.fft2(y) + a * np.fft.fft2(x)) / (np.abs(H) ** 2 + a)).real
    FB = p2o(_t(k), y.shape)
    FBC = torch.conj(FB)
    z = data_step(_t(x), FB, FBC, (FB.abs() ** 2).to(FB.dtype), FBC * torch.fft.fft2(_t(y)),
                  torch.tensor(a, dtype=torch.float64), 1)
    np.testing.assert_allclose(z[0, 0].numpy(), ref, atol=1e-12)


def _tiny_model():
    torch.manual_seed(0)
    return USRNet(n_iter=TINY["n_iter"], h_nc=TINY["h_nc"], nc=tuple(TINY["nc"]), nb=TINY["nb"]).eval()


def test_restore_matches_forward(scene):
    model = _tiny_model()
    k = gaussian_psf(1.2, 9)
    y = circular_blur(scene, k)
    steps = []
    out = usrnet_restore(y, k, 0.01, model=model, callback=lambda i, x: steps.append(i))
    with torch.no_grad():
        ref = model(torch.as_tensor(y, dtype=torch.float32)[None, None].repeat(1, 3, 1, 1),
                    torch.as_tensor(k, dtype=torch.float32)[None, None], 1,
                    torch.full((1, 1, 1, 1), 0.01)).mean(1)[0].numpy()
    np.testing.assert_allclose(out, ref, atol=1e-5)
    assert steps == [0, 1] and out.dtype == np.float64


def test_learned_schedule_shape():
    a, b = learned_schedule(_tiny_model(), 0.01)
    assert a.shape == b.shape == (TINY["n_iter"],) and (a > 0).all() and (b > 0).all()


class _Sampler:
    """Paires à flou circulaire, comme l'entraînement d'origine (sans dépendance à une tâche)."""

    def __init__(self, img, seed):
        self.img, self.rng = img.astype(np.float32), np.random.default_rng(seed)

    def batch(self, n, device):
        ys, xs, ks = [], [], []
        for _ in range(n):
            r, c = self.rng.integers(0, self.img.shape[0] - 32, 2)
            x = self.img[r : r + 32, c : c + 32]
            k = gaussian_psf(float(self.rng.uniform(0.8, 2.0)), 9)
            ys.append(circular_blur(x, k) + 0.01 * self.rng.standard_normal(x.shape))
            xs.append(x)
            ks.append(k)
        t = lambda a: torch.as_tensor(np.stack(a), dtype=torch.float32, device=device)
        return (t(ys)[:, None].repeat(1, 3, 1, 1), t(xs)[:, None].repeat(1, 3, 1, 1), t(ks)[:, None],
                torch.full((n, 1, 1, 1), 0.01, device=device))


def test_fit_from_scratch_and_reload(tmp_path, scene):
    cfg = TrainingConfig(init_weights=None, arch=TINY, iterations=4, batch_size=2, val_every=2,
                         val_samples=2, out_dir=str(tmp_path), device="cpu", seed=0)
    ckpt = fit_usrnet(_Sampler(scene, 0), lambda: _Sampler(scene, 123), cfg, log=lambda m: None)
    assert ckpt.exists() and (tmp_path / "history.csv").exists() and (tmp_path / "best.sha256").exists()
    model = load_usrnet(str(ckpt), "cpu")
    assert isinstance(model, USRNet) and model.n == TINY["n_iter"]


def test_unknown_weights_raise():
    with pytest.raises(ValueError):
        load_usrnet("inconnu")


@pytest.mark.weights
def test_published_tiny_weights(scene):
    model = load_usrnet("usrnet_tiny", "cpu")
    assert sum(p.numel() for p in model.parameters()) == 590332
    k = gaussian_psf(1.6, 15)
    rng = np.random.default_rng(0)
    y = circular_blur(scene, k) + 0.01 * rng.standard_normal(scene.shape)
    out = usrnet_restore(y, k, 0.01, model=model)
    assert np.mean((out - scene) ** 2) < 0.5 * np.mean((y - scene) ** 2)
