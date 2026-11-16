"""
Rôle 2 — le réseau déplié.

Un algorithme itératif (ici l'éclatement semi-quadratique du Plug-and-Play)
écrit comme un réseau de ``K`` étapes, appris de bout en bout : le module
de données reste la formule exacte où entre la PSF, le débruiteur et le
calendrier sont appris sur le résultat final.

- :mod:`.usrnet` : USRNet (Zhang, Van Gool & Timofte, CVPR 2020) ;
- :mod:`.training` : boucle d'apprentissage et d'ajustement fin, les
  paires étant fournies par le module de la tâche.

Import paresseux : PyTorch n'est chargé qu'à l'accès à ces sous-modules.
"""
