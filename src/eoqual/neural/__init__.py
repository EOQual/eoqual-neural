"""
eoqual-neural — les réseaux de neurones de restauration d'EOQual, rangés par
**rôle** plutôt que par tâche.

Un réseau n'est presque jamais spécialisé par sa conception : c'est son
apprentissage qui le spécialise, et le bruit fait toujours partie de cet
apprentissage. Les modules par tâche (eoqual-denoising,
eoqual-deconvolution…) restent consacrés au problème et aux méthodes
classiques ; ils appellent ce paquet pour leurs méthodes neuronales.

- :mod:`.denoisers` — rôle 1, le débruiteur appris (DRUNet, GS-DRUNet,
  SwinIR, Restormer), seul ou comme a priori d'un Plug-and-Play ;
- :mod:`.unrolled` — rôle 2, le réseau déplié (USRNet) et son
  ajustement fin ;
- rôle 3 (réseaux de bout en bout) et rôle 4 (a priori génératifs) : à venir ;
- :mod:`.weights` — registre des poids (URL, empreinte, licence, cache) ;
- :mod:`.torch_utils` — import de PyTorch, périphérique, cache des modèles ;
- :mod:`.validation` — transfert par bande de fréquence, biais sur aplats.

PyTorch n'est importé qu'au premier appel d'un réseau.
"""
from __future__ import annotations

import importlib

from .denoisers import DENOISERS, get_denoiser, list_denoisers
from .validation import flat_bias, transfer_by_band
from .weights import DEEPINV_WEIGHTS, KNOWN_WEIGHTS, resolve_weights, weights_dir

__version__ = "0.1.1"

__all__ = [
    "__version__",
    "DENOISERS",
    "get_denoiser",
    "list_denoisers",
    "KNOWN_WEIGHTS",
    "DEEPINV_WEIGHTS",
    "resolve_weights",
    "weights_dir",
    "transfer_by_band",
    "flat_bias",
    "load_usrnet",
    "usrnet_restore",
]

_LAZY = {"load_usrnet": "eoqual.neural.unrolled.usrnet", "usrnet_restore": "eoqual.neural.unrolled.usrnet"}


def __getattr__(name):
    """Accès paresseux aux fonctions qui chargent PyTorch."""
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module 'eoqual.neural' has no attribute {name!r}")
