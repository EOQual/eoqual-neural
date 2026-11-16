# Change Log

Versions publiées et contenu de chacune.

## Version History

0.1.0 (2026-11-16)

:   -   **Création** : les réseaux de neurones d'EOQual, rangés par rôle.
    -   Rôle 1, `denoisers` : DRUNet, GS-DRUNet, SwinIR, Restormer
        (deepinv), repris d'eoqual-denoising 0.2.2 ; `get_denoiser`,
        `list_denoisers`, convention d'échelle [0, 1].
    -   Rôle 2, `unrolled` : USRNet (architecture KAIR, MIT),
        `usrnet_restore`, `learned_schedule`, `load_usrnet`, et boucle
        d'apprentissage générique `fit_usrnet`, repris
        d'eoqual-deconvolution 0.5.0.
    -   Outils communs : `weights` (registre, empreintes, licences, cache
        `~/.cache/eoqual/neural` réutilisant l'ancien cache),
        `torch_utils` (remplace les deux copies des modules par tâche),
        `validation` (`transfer_by_band`, `flat_bias`).
    -   Migration : eoqual-denoising 0.3.0 et eoqual-deconvolution 0.6.0
        délèguent leurs réseaux à ce paquet ; sorties identiques au bit
        près.
