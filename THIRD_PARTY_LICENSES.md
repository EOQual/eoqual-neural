# Licences tierces

Le code d'eoqual-neural est sous licence MIT (`license.txt`). Ce document
recense les dépendances, l'origine et la licence des poids, et l'unique
code repris d'un tiers.

## 1. Dépendances

| Paquet | Usage | Licence |
|---|---|---|
| numpy | calcul | BSD-3-Clause |
| torch | tous les réseaux | BSD-3-Clause (avec composants sous licences permissives) |
| deepinv | débruiteurs appris (DRUNet, GS-DRUNet, SwinIR, Restormer) | BSD-3-Clause |
| einops | requis par certains réseaux de deepinv | MIT |

## 2. Poids pré-entraînés

Aucun poids n'est inclus dans le paquet. Ils sont téléchargés à la demande.

| Réseau | Fichier | Origine | Licence |
|---|---|---|---|
| DRUNet | `drunet_deepinv_gray_finetune_26k.pth` | deepinv (Hugging Face `deepinv/drunet`) | BSD-3-Clause |
| SwinIR | `004_grayDN_DFWB_s128w8_SwinIR-M_noise{15,25,50}.pth` | publication SwinIR | Apache-2.0 |
| Restormer | `gaussian_gray_denoising_blind.pth` | publication Restormer | MIT |
| GS-DRUNet | `GSDRUNet_grayscale_torch.ckpt` | deepinv (Hugging Face `deepinv/gradientstep`) | ⚠️ **non déclarée** par l'amont (code GS-PnP sous MIT) |
| USRNet | `usrnet.pth`, `usrnet_tiny.pth` | publication v1.0 de KAIR, vérifiés par SHA-256 | MIT (© 2019 Kai Zhang) |

Les poids de GS-DRUNet n'ont pas de licence déclarée : à vérifier avant un
usage dans une chaîne distribuée.

## 3. Code repris : USRNet (KAIR, MIT)

`src/eoqual/neural/unrolled/usrnet.py` reprend la structure et les noms de
paramètres du réseau USRNet de KAIR (`models/network_usrnet.py`,
`models/basicblock.py`, © 2019 Kai Zhang, **licence MIT**,
https://github.com/cszn/KAIR), afin de charger les poids publiés par
l'auteur. Le code a été adapté (API FFT complexe actuelle de PyTorch au
lieu de `torch.rfft`, supprimée ; boucle d'inférence réécrite) ; la notice
de licence est citée en tête du module. Ce code provient d'eoqual-deconvolution
0.5.0, où cette exception était déjà documentée. La boucle d'apprentissage
(`unrolled/training.py`) est écrite pour EOQual, depuis l'article.
