# FaceVision v2.0 — Plateforme de Vérification Faciale (SVM RBF |e₁−e₂| 512D)

Plateforme reconstruite à partir de `notebook_corrige_avec_analyse_scientifique.ipynb` :
pipeline FaceNet (InceptionResnetV1, VGGFace2, 512D normalisé L2) → |e₁−e₂| ∈ ℝ⁵¹²
→ StandardScaler → SVM RBF + Calibration Platt-Sigmoid → décision ternaire
(ACCEPTÉ / INCERTAIN / REJETÉ) via marge fixe ±0.15 autour du seuil EER.

**Recadrage scientifique** : l'adaptativité du système ne repose plus sur des variables
comportementales externes (tentatives, échecs) mais sur un **score de confiance appris
automatiquement** à partir des embeddings. Le SVM apprend quelles dimensions de |e₁−e₂|
sont discriminantes pour séparer genuine/impostor.

## Lancer la plateforme — 3 étapes

```bash
pip install -r requirements.txt
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install facenet-pytorch

# 1) Placez vos données brutes :
#      DATA/ORL_DATA/<sujet>/<image>.pgm        (40 sujets x 10 images)
#      DATA/Yale_grouped/<sujet>/<image...>     (15 sujets x 11 conditions)

# 2) Génère data/ (split 50/50 + CLAHE + MTCNN + FaceNet 512D) puis models/
#    (Seuil Fixe + SVM RBF + GridSearchCV 5-fold + Calibration Platt)
python run_all.py

# 3) Démarre le backend + sert la plateforme
python app.py
```

Puis ouvrir http://localhost:5000

Les deux étapes de `run_all.py` peuvent aussi être lancées séparément :
```bash
python prepare_data.py          # DATA brutes -> data/*_MTCNN, data/*_EMBEDDINGS
python train_from_notebook.py   # data/ -> models/*.pkl + *_meta.json
```

⚠️ Tant que `models/` et `data/` n'existent pas, la page **Test** affiche un bandeau
rouge « Backend non connecté — simulation de démonstration » au lieu de prétendre
utiliser le vrai SVM.

## Structure

```
facevision/
├── app.py                     # Backend Flask v2.0 : API + pages HTML
├── prepare_data.py            # Cellules 2-3 du notebook -> data/ (split, CLAHE, MTCNN, FaceNet)
├── train_from_notebook.py     # Cellules 5-6 du notebook -> models/ (Seuil Fixe + SVM + Platt)
├── run_all.py                 # Enchaîne prepare_data.py puis train_from_notebook.py
├── requirements.txt
├── DATA/                        # Données brutes à fournir
│   ├── ORL_DATA/<sujet>/*.pgm
│   └── Yale_grouped/<sujet>/*
├── data/                         # Généré par prepare_data.py
│   ├── ORL_MTCNN/, YALE_MTCNN/             # Visages alignés 160x160 (train/test)
│   └── ORL_EMBEDDINGS/, YALE_EMBEDDINGS/   # Embeddings FaceNet 512D .npy normalisés L2
├── models/                      # Généré par train_from_notebook.py
│   ├── orl_svm_calibrated.pkl / orl_scaler.pkl / orl_meta.json / orl_seuil_meta.json
│   └── yale_svm_calibrated.pkl / yale_scaler.pkl / yale_meta.json / yale_seuil_meta.json
└── platform/
    ├── index.html       # Accueil
    ├── modeles.html      # Résultats SVM RBF 512D (onglets ORL / Yale / Comparaison)
    ├── resultats.html    # Évaluation complète, calibration, erreurs, discussion
    ├── apropos.html      # Positionnement scientifique, limites, perspectives
    ├── methodo.html      # Méthodologie pas-à-pas du pipeline
    ├── test.html         # Test : import d'une image -> décision ACCÈS/INCERTAIN/REFUS
    ├── shared.css         # Thème
    └── plots/              # Figures du notebook (fig1 à fig13 + figA)
```

## Résultats clés (test set, split 50/50)

| Métrique          | Seuil Fixe ORL | SVM RBF 512D ORL | Seuil Fixe Yale | SVM RBF 512D Yale |
|--------------------|----------------|-------------------|------------------|----------------------|
| AUC ROC            | 0.9994         | 0.9995            | 1.0000           | 1.0000               |
| EER                | ~1.6%          | ~1.5%             | ~0.67%           | ~0.67%               |
| Accuracy           | ~98.4%         | ~98.5%            | ~99.3%           | ~99.3%               |
| Zones              | binaire        | ternaire (~2.1% INCERTAIN) | binaire | ternaire (~0.7% INCERTAIN) |

Décision ternaire : `Tlow = clip(seuil_EER - 0.15, 0.05)`, `Thigh = clip(seuil_EER + 0.15, 0.95)`.
P(genuine) ≥ Thigh → ACCEPTÉ · Tlow ≤ P(genuine) < Thigh → INCERTAIN · P(genuine) < Tlow → REJETÉ.

## Feature engineering — le choix clé

```
X = |e1 - e2| ∈ R^512   (différence absolue, invariante à l'ordre de la paire)
```

512 dimensions au lieu d'une seule distance cosine : le SVM apprend quelles composantes
de l'embedding FaceNet sont discriminantes pour séparer genuine/impostor.

## Pourquoi le SVM n'améliore pas drastiquement l'AUC ?

FaceNet est déjà extrêmement discriminant sur ORL/Yale (bases frontales contrôlées,
AUC ≈ 1.00 dès le seuil fixe). Le gain n'est donc pas dans l'exactitude brute, mais dans
la **calibration du niveau de confiance** (P(genuine) interprétable comme probabilité,
validée par Brier Score & ECE) et la **gestion explicite de l'incertitude** (zone INCERTAIN
ancrée sur le chevauchement statistique réel des distributions).

## Page Test — protocole (v2.0)

⚠️ **Important** : ce pipeline n'utilise plus de variables comportementales
(plus de champs tentatives/échecs). La décision repose uniquement sur le vecteur visuel
|e₁−e₂| ∈ ℝ⁵¹².

1. Import d'une image de visage (JPEG/PNG/BMP/WebP).
2. Le backend extrait l'embedding FaceNet 512D (MTCNN + CLAHE), puis cherche par
   identification 1:N l'identité enrôlée la plus proche (ORL ou Yale, cosine).
3. Calcule |e₁−e₂| réel avec cette référence, applique le StandardScaler + SVM calibré
   du dataset correspondant, puis la décision ternaire (ACCÈS / INCERTAIN / REFUS).
4. Bouton « Afficher la justification scientifique » : dimensions les plus contributives,
   comparaison avec le seuil fixe, position dans la zone de décision.

Si `models/` ou `data/` sont absents (backend non préparé), un bandeau rouge explicite
apparaît et le résultat affiché est une simulation, jamais présentée comme une vraie
prédiction du SVM.
