# Contribuer à eoqual-neural

Les contributions sont bienvenues : signalement de bug, correction,
nouveau réseau, documentation.

## Où ranger un réseau

Par **rôle**, pas par tâche :

- un débruiteur appris → `denoisers/` (fonction `f(image_unit, sigma, **kw)`,
  déclarée dans `DENOISERS`) ;
- un algorithme déplié et appris de bout en bout → `unrolled/` ;
- un réseau de bout en bout, un a priori génératif → nouveaux
  sous-paquets, à proposer d'abord en issue.

Règles :

- travailler dans l'échelle [0, 1] ; la conversion depuis les unités du
  capteur, les bords et le modèle de bruit relèvent du module de la tâche ;
- ne jamais importer un module par tâche ;
- poids non inclus, téléchargés et vérifiés par SHA-256 (`weights.py`),
  licence des poids vérifiée et inscrite dans `THIRD_PARTY_LICENSES.md` ;
- une méthode qui utilise PyTorch sans être un réseau (EPLL, TGV…) reste
  dans le module de sa tâche.

## Tests

`pytest` ; les tests qui téléchargent des poids portent le marqueur
`weights`. Vérifier qu'un réseau déplacé depuis un module par tâche donne
des sorties identiques avant et après.
