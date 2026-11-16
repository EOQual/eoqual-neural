# eoqual-neural — référence technique

Organisation du paquet, choix de conception et limites. Les méthodes
elles-mêmes sont décrites dans la référence technique du module de leur
tâche (eoqual-denoising §3, eoqual-deconvolution §7), qui les expose à
l'utilisateur.

## 1. Pourquoi un module des réseaux

Jusqu'à la version 0.2 d'eoqual-denoising et 0.5 d'eoqual-deconvolution,
chaque module par tâche portait ses réseaux, avec sa propre copie des
outils PyTorch (`backends/_torch.py`, `_common_torch.py`) et son propre
cache de poids. Trois raisons de les regrouper :

1. **Les réseaux ne sont pas spécialisés par tâche.** Un débruiteur appris
   (DRUNet) sert seul, mais aussi d'a priori dans le Plug-and-Play de
   déconvolution, de dématriçage ou de super-résolution. USRNet, rangé en
   déconvolution, est un réseau de super-résolution qui traite ensemble le
   flou, la réduction d'échelle et le bruit. Ce qui spécialise un réseau,
   ce sont ses paires d'entraînement.
2. **Une seule dépendance lourde.** PyTorch, deepinv et les poids sont
   ici ; les modules par tâche redeviennent légers et classiques par
   défaut.
3. **Un endroit pour les rôles à venir** (réseaux de bout en bout, a priori
   génératifs), qui n'appartiennent à aucune tâche unique.

## 2. Organisation par rôle

| Rôle | Ce que fait le réseau | Ce qui reste explicite | Sous-paquet |
|---|---|---|---|
| 1. débruiteur appris | débruite, σ en entrée | modèle de bruit ; dans un Plug-and-Play, le modèle direct | `denoisers` |
| 2. réseau déplié | remplace l'a priori et règle le calendrier | modèle direct (PSF à chaque étape) | `unrolled` |
| 3. bout en bout | tout | rien (la dégradation n'est que dans les données) | à venir |
| 4. a priori génératif | propose des scènes vraisemblables | modèle direct ; résultat aléatoire | à venir |

## 3. Frontière avec les modules par tâche

**Règle de dépendance.** eoqual-neural ne dépend d'aucun module EOQual.
Les modules par tâche en dépendent par un extra optionnel :

| Module | Extra | Ce qu'il garde | Ce qu'il délègue |
|---|---|---|---|
| eoqual-denoising ≥ 0.3.0 | `neural` (et `deep`, qui l'inclut) | unités du capteur, σ estimé, `data_range`, stabilisation de variance (`backends/neural.py`) | `drunet`, `swinir`, `restormer`, `gsdrunet` |
| eoqual-deconvolution ≥ 0.6.0 | `deep` | bords, pixels masqués ou saturés, échelle (`backends/usrnet.py`) ; paires de réentraînement : lois des PSF d'instrument, bruit, bords réels (`training.py`) ; algorithme du Plug-and-Play | réseau USRNet, poids, boucle d'apprentissage |

**Convention d'échelle.** Toutes les fonctions de ce paquet reçoivent et
rendent des images dans [0, 1], σ dans la même échelle. C'est l'échelle
d'entraînement des poids publiés ; la conversion appartient à la tâche.

**Ce qui n'est pas un réseau reste dans sa tâche.** EPLL (mélange gaussien
appris), TGV et le dictionnaire d'ondelettes sont calculés avec deepinv
mais ne sont pas des réseaux de neurones ; NL-Ridge utilise PyTorch comme
moteur de calcul. Ils restent dans eoqual-denoising (extra `deep`).

## 4. Rôle 1 — débruiteurs appris (`denoisers`)

Fines couches autour de deepinv : DRUNet, GS-DRUNet, SwinIR, Restormer, en
niveaux de gris. `get_denoiser(nom)` renvoie `f(image_unit, sigma, **kw)`.

| Réseau | Comment il connaît σ | Conséquence |
|---|---|---|
| DRUNet | carte de σ en entrée | un seul réseau pour toute la plage ; se combine à la stabilisation |
| GS-DRUNet | idem ; pas de gradient d'un potentiel | Plug-and-Play convergent ; ≈ 0,2 dB sous DRUNet seul |
| SwinIR | un jeu de poids par niveau (15, 25, 50 sur [0, 255]) | se dégrade entre deux niveaux |
| Restormer | aveugle (σ ignoré) | échoue hors de son domaine (σ > 50/255, faible flux) |

GS-DRUNet : le fichier de poids publié contient des compteurs de métriques
que deepinv 0.4.2 refuse en chargement strict ; seules les clés du réseau
sont chargées, après vérification qu'aucune ne manque.

Périphérique : CUDA si disponible, sinon CPU ; MPS (GPU Apple) n'est pas
choisi automatiquement pour ces réseaux (opérations deepinv non
implémentées), mais peut être imposé.

## 5. Rôle 2 — réseau déplié (`unrolled`)

**USRNet** (Zhang, Van Gool & Timofte, CVPR 2020) déplie l'éclatement
semi-quadratique du Plug-and-Play en `K` étapes (8, ou 6 pour le modèle
réduit) : module de données exact en Fourier (où entre la PSF), module
d'a priori ResUNet conditionné par β_k, et HyPaNet qui donne le calendrier
(α_k, β_k) à partir de (σ, s).

- `usrnet_restore(y, psf, sigma, …)` : inférence (``s = 1``) sur un
  domaine **périodique déjà préparé**, cohérent avec la PSF. Le
  remplissage de la marge (Wiener–Hunt à marge inconnue) est fait par
  eoqual-deconvolution ; compléter à chaque itération diverge, les
  estimées intermédiaires n'étant pas des images.
- `learned_schedule(model, sigma)` : calendrier appris (α_k croît, β_k
  décroît d'étape en étape, comme la décroissance géométrique des σ_k du
  Plug-and-Play).
- `load_usrnet(poids)` : `"usrnet"` (17,0 M de paramètres),
  `"usrnet_tiny"` (0,59 M), ou point de contrôle réentraîné (qui contient
  son architecture) ; chargement strict, mis en cache.
- `training.fit_usrnet(tireur, fabrique_validation, cfg)` : boucle
  d'apprentissage générique. Le **tireur** (`batch(n, device)` →
  `(y, x, k, s)`) porte la physique de la tâche et vient du module de la
  tâche ; `TrainingConfig` peut être étendue par sous-classe (c'est ce que
  fait eoqual-deconvolution, dont `train_usrnet(images, images_val, cfg)`
  reste l'entrée habituelle).

Code repris de KAIR (MIT) pour charger les poids publiés : voir
THIRD_PARTY_LICENSES.md.

## 6. Poids (`weights`)

`KNOWN_WEIGHTS` : URL, empreinte SHA-256, licence, architecture des poids
gérés par ce paquet (USRNet). `DEEPINV_WEIGHTS` : poids téléchargés par
deepinv, décrits pour l'audit des licences. Cache
`$EOQUAL_WEIGHTS_DIR` ou `~/.cache/eoqual/neural` ; un fichier présent dans
l'ancien cache `~/.cache/eoqual/deconvolution` est réutilisé. Une empreinte
différente lève une erreur (fichier à supprimer pour le retélécharger).

## 7. Validation (`validation`)

Une restauration non linéaire n'a pas de MTF : elle transmet différemment
un bord franc (motif qu'un réseau sait redessiner, parfois plus net que la
réalité) et une texture fine (qu'il efface en partie). Deux outils,
numpy seul :

- `transfer_by_band(vérité, sortie)` : ``Re⟨O, T⟩ / ⟨T, T⟩`` par anneau de
  fréquence. 1 : transmis intact ; > 1 : amplifié au-delà de la vérité.
  L'énergie non corrélée à la vérité (bruit résiduel, détail inventé)
  n'y entre pas ;
- `flat_bias(restaurer, niveaux, bruit)` : biais relatif sur des aplats.

Protocole recommandé avant tout usage opérationnel d'un réseau : scènes
simulées représentatives de la mission, transfert par contenu (bord et
textures), radiométrie (aplats sombres et clairs), cas limites (σ hors
plage, PSF un peu fausse, bruit corrélé), comparaison à la meilleure
méthode classique, traçabilité des poids.

## 8. Tests

`pytest` (20 tests) : registre et empreintes des poids, transfert et biais
(identité, gain, passe-bas, bruit non corrélé), module de données égal à la
formule fermée à 10⁻¹² près, OTF égale à une FFT numpy, inférence égale au
`forward` du réseau, calendrier appris, apprentissage de 4 itérations et
rechargement, exemple. Les tests marqués `weights` téléchargent les poids (DRUNet,
USRNet réduit) et vérifient un gain réel.

Migration vérifiée : sorties **identiques au bit près** avant et après le
déplacement, pour DRUNet, SwinIR, Restormer, GS-DRUNet (via
eoqual-denoising) et pour USRNet et le Plug-and-Play DRUNet (via
eoqual-deconvolution, flous gaussien et bougé).

## 9. Limites

- Niveaux de gris seulement pour les débruiteurs (comme avant la
  migration).
- Les poids publiés sont appris sur des photographies : valider sur des
  scènes de la mission (§7) avant tout usage opérationnel.
- Rôles 3 et 4 non encore implémentés.
