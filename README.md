<p align="center">
  <img src="resources/eoqual_rounded.png" alt="EOQual" width="280">
</p>

# eoqual-neural

Les réseaux de neurones de restauration d'images d'EOQual, rangés par
**rôle** plutôt que par tâche. Un réseau n'est presque jamais spécialisé
par sa conception : c'est son apprentissage qui le spécialise, et le bruit
fait toujours partie de cet apprentissage. Le même débruiteur appris sert
ainsi au débruitage, au Plug-and-Play de déconvolution ou de
super-résolution ; un réseau déplié comme USRNet traite d'un même bloc le
flou, la réduction d'échelle et le bruit.

Les modules par tâche ([eoqual-denoising](https://github.com/EOQual/eoqual-denoising),
[eoqual-deconvolution](https://github.com/EOQual/eoqual-deconvolution)…)
restent consacrés au problème et aux méthodes classiques ; ils appellent ce
paquet pour leurs méthodes neuronales, derrière leur appel habituel
(`denoise(image, "drunet", …)`, `deconvolve(image, psf, "usrnet", …)`).
PyTorch, deepinv et la gestion des poids sont ici, et seulement ici.

Voir **[REFERENCE_TECHNIQUE.md](REFERENCE_TECHNIQUE.md)** pour l'organisation
et les choix, **[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md)** pour
l'audit des licences (code et poids) et **[BIBLIOGRAPHY.md](BIBLIOGRAPHY.md)**
pour les publications de référence.

## Sommaire

1. [Installation](#installation)
2. [Démarrage rapide](#démarrage-rapide)
3. [Les rôles](#les-rôles)
4. [Conventions](#conventions)
5. [Licence](#licence)

---

## Installation

```bash
pip install "eoqual-neural @ git+https://github.com/EOQual/eoqual-neural.git@v0.1.1"
```

Dépendances : numpy, PyTorch (≥ 2.2), deepinv, einops. En pratique, on
installe plutôt l'extra du module de la tâche, qui tire ce paquet :
`eoqual-denoising[neural]`, `eoqual-deconvolution[deep]`.

Les poids ne sont jamais inclus : ceux de deepinv sont téléchargés par
deepinv (cache de PyTorch, `TORCH_HOME`) ; ceux d'USRNet par ce paquet,
vérifiés par SHA-256, dans `$EOQUAL_WEIGHTS_DIR` ou `~/.cache/eoqual/neural`.

## Démarrage rapide

```python
import numpy as np
from eoqual.neural import get_denoiser, transfer_by_band
from eoqual.neural.unrolled.usrnet import usrnet_restore

# Rôle 1 : débruiteur appris, image et σ dans l'échelle [0, 1]
propre = get_denoiser("drunet")(bruitee_unit, 0.05)

# Rôle 2 : réseau déplié, sur un domaine périodique déjà préparé
restauree = usrnet_restore(floue_unit, psf, 0.01, weights="usrnet")

# Validation : ce que la restauration transmet, bande de fréquence par bande
f, t = transfer_by_band(verite, restauree)
```

Exemple complet : `examples/quickstart.py`.

## Les rôles

| Rôle | Sous-paquet | Contenu | Utilisé par |
|---|---|---|---|
| 1. débruiteur appris | `eoqual.neural.denoisers` | DRUNet, GS-DRUNet (conditionnés par σ), SwinIR (un jeu de poids par niveau), Restormer (aveugle) | eoqual-denoising ; Plug-and-Play d'eoqual-deconvolution |
| 2. réseau déplié | `eoqual.neural.unrolled` | USRNet, calendrier appris, boucle d'ajustement fin | eoqual-deconvolution (`usrnet`, `training`) |
| 3. réseau de bout en bout | — | à venir (dématriçage-débruitage conjoint, super-résolution) | — |
| 4. a priori génératif | — | à venir (a priori de diffusion) | — |

Outils communs : `weights` (registre URL / empreinte / licence / cache),
`torch_utils` (import, périphérique, cache des modèles), `validation`
(transfert par bande de fréquence, biais sur aplats — numpy seul).

## Conventions

- Les fonctions de ce paquet travaillent dans l'**échelle [0, 1]** : image
  et σ. La conversion depuis les unités du capteur, l'estimation de σ, la
  stabilisation de variance et le traitement des bords relèvent du module
  de la tâche.
- Ce paquet n'importe jamais un module par tâche ; les modules par tâche
  l'importent (extra optionnel).
- Les méthodes classiques calculées avec PyTorch (EPLL, TGV, NL-Ridge…)
  ne sont pas des réseaux et restent dans leur module.

## Licence

MIT (`license.txt`). Le code d'USRNet reprend la structure de KAIR (MIT,
© 2019 Kai Zhang) pour charger ses poids publiés ; voir
[THIRD_PARTY_LICENSES.md](THIRD_PARTY_LICENSES.md).
