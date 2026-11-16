#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Registre des poids pré-entraînés gérés par le paquet : téléchargement à la
demande, cache local, vérification par empreinte SHA-256, licence. Les
poids ne sont jamais inclus dans le paquet.

Les poids des réseaux de deepinv (DRUNet, SwinIR, Restormer, GS-DRUNet)
sont gérés par deepinv lui-même (cache de PyTorch, ``TORCH_HOME``) ; ils
sont décrits dans :data:`DEEPINV_WEIGHTS` pour l'audit des licences.

Cache : ``$EOQUAL_WEIGHTS_DIR`` s'il est défini, sinon
``~/.cache/eoqual/neural``. Un fichier déjà présent dans l'ancien cache
d'eoqual-deconvolution (``~/.cache/eoqual/deconvolution``) est réutilisé
sans nouveau téléchargement.
"""
from __future__ import annotations

import hashlib
import os
import urllib.request
from pathlib import Path

__all__ = ["KNOWN_WEIGHTS", "DEEPINV_WEIGHTS", "weights_dir", "resolve_weights", "sha256"]

#: Poids gérés par ce paquet : URL, empreinte SHA-256, licence, architecture.
KNOWN_WEIGHTS = {
    "usrnet": {
        "url": "https://github.com/cszn/KAIR/releases/download/v1.0/usrnet.pth",
        "sha256": "2b8c32777fc56bde7fe81f47e60b83c109606c8559aaaafecf306a2108a2082b",
        "license": "MIT (KAIR, Kai Zhang)",
        "arch": {"n_iter": 8, "h_nc": 64, "nc": (64, 128, 256, 512), "nb": 2},
    },
    "usrnet_tiny": {
        "url": "https://github.com/cszn/KAIR/releases/download/v1.0/usrnet_tiny.pth",
        "sha256": "d82929fc053d6bbe181adba9e2be3d23548eeca8b7b4232eb122f020352295cf",
        "license": "MIT (KAIR, Kai Zhang)",
        "arch": {"n_iter": 6, "h_nc": 32, "nc": (16, 32, 64, 64), "nb": 2},
    },
}

#: Poids téléchargés par deepinv (origine et licence vérifiées, THIRD_PARTY_LICENSES.md).
DEEPINV_WEIGHTS = {
    "drunet": {"file": "drunet_deepinv_gray_finetune_26k.pth", "license": "BSD-3-Clause (deepinv)"},
    "swinir": {"file": "SwinIR (σ = 15, 25, 50, niveaux de gris)", "license": "Apache-2.0 (SwinIR)"},
    "restormer": {"file": "Restormer, débruitage niveaux de gris aveugle", "license": "MIT (Restormer)"},
    "gsdrunet": {"file": "GSDRUNet_grayscale_torch.ckpt", "license": "non déclarée par l'amont (GS-PnP, MIT pour le code)"},
}

_LEGACY_DIR = Path.home() / ".cache" / "eoqual" / "deconvolution"


def weights_dir() -> Path:
    """Répertoire du cache des poids."""
    return Path(os.environ.get("EOQUAL_WEIGHTS_DIR", Path.home() / ".cache" / "eoqual" / "neural"))


def sha256(path: str | Path) -> str:
    """Empreinte SHA-256 d'un fichier."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def resolve_weights(name_or_path: str | Path) -> Path:
    """
    Chemin local des poids : fichier existant, ou nom de
    :data:`KNOWN_WEIGHTS` (téléchargé au premier appel puis vérifié).

    Raises
    ------
    ValueError
        Nom inconnu, ou empreinte différente de celle attendue.
    """
    p = Path(name_or_path)
    if p.suffix and p.exists():
        return p
    name = str(name_or_path)
    if name not in KNOWN_WEIGHTS:
        raise ValueError(f"Poids inconnus {name!r} : chemin de fichier ou un de {sorted(KNOWN_WEIGHTS)}.")
    info = KNOWN_WEIGHTS[name]
    dest = weights_dir() / f"{name}.pth"
    legacy = _LEGACY_DIR / f"{name}.pth"
    if not dest.exists() and "EOQUAL_WEIGHTS_DIR" not in os.environ and legacy.exists():
        dest = legacy
    if not dest.exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(".part")
        urllib.request.urlretrieve(info["url"], tmp)
        tmp.replace(dest)
    digest = sha256(dest)
    if digest != info["sha256"]:
        raise ValueError(f"Empreinte inattendue pour {dest} : {digest} (attendu {info['sha256']}). "
                         "Supprimer le fichier pour le retélécharger.")
    return dest
