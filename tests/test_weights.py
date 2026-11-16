"""Registre des poids : chemins, noms inconnus, empreintes."""
import pytest

from eoqual.neural.weights import DEEPINV_WEIGHTS, KNOWN_WEIGHTS, resolve_weights, sha256, weights_dir


def test_registry_entries_are_complete():
    for name, info in KNOWN_WEIGHTS.items():
        assert {"url", "sha256", "license", "arch"} <= set(info), name
        assert len(info["sha256"]) == 64
    assert {"drunet", "swinir", "restormer", "gsdrunet"} <= set(DEEPINV_WEIGHTS)


def test_existing_file_is_returned(tmp_path):
    p = tmp_path / "poids.pth"
    p.write_bytes(b"abc")
    assert resolve_weights(p) == p
    assert sha256(p) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"


def test_unknown_name_raises():
    with pytest.raises(ValueError):
        resolve_weights("inconnu")


def test_weights_dir_follows_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("EOQUAL_WEIGHTS_DIR", str(tmp_path))
    assert weights_dir() == tmp_path
