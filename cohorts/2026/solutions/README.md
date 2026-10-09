# Solutions des devoirs ML Zoomcamp 2026

Scripts qui rejouent les devoirs fermés de la cohorte 2026. Les données
tabulaires sont celles déjà versionnées dans `cohorts/2026/data/`. Les
versions utilisées sont celles du release : NumPy 2.3.3, pandas 2.3.2,
scikit-learn 1.7.2, XGBoost 3.2.0.

```bash
python cohorts/2026/solutions/hw01_intro.py
python cohorts/2026/solutions/hw02_regression.py
python cohorts/2026/solutions/hw03_classification.py
python cohorts/2026/solutions/hw04_evaluation.py
python cohorts/2026/solutions/hw05_deployment.py
python cohorts/2026/solutions/hw06_trees.py
python cohorts/2026/solutions/hw08_deep_learning.py
python cohorts/2026/solutions/hw09_serverless.py
python cohorts/2026/solutions/hw10_kubernetes.py
python cohorts/2026/data/validate_homework.py
```

Le validateur officiel (`validate_homework.py`) confirme que les réponses de
HW1 à HW4 et HW6 font partie des options publiées.

## Homework 1 — introduction

| Question | Réponse |
| --- | --- |
| Q1. Version de pandas | 2.3.2 |
| Q2. Nombre de lignes | 10000 |
| Q3. Types de carburant | 3 |
| Q4. Colonnes avec valeurs manquantes | 2 |
| Q5. MPG max, origine Asie | 41.2 |
| Q6. Médiane de `horsepower` après le mode | Yes, it decreased (254.0 puis 252.0) |
| Q7. Somme des poids | 0.369 |

La publication « learning in public » est un post sur les réseaux sociaux, pas
un calcul. Elle n'est pas faite ici.

## Homework 2 — régression

| Question | Réponse |
| --- | --- |
| Q1. Colonne manquante | `horsepower` |
| Q2. Médiane de `horsepower` | 254 |
| Q3. Meilleure imputation | With mean (RMSE 2.202 contre 2.205 avec 0) |
| Q4. Meilleur `r` | 0 |
| Q5. Écart-type des RMSE | 0.029 |
| Q6. RMSE test, seed 9, `r=0.001` | 2.236 |

La fonction du cours ajoute `r` sur toute la diagonale, y compris le biais.
Le contrôle de release met le coefficient du biais à zéro. Les deux méthodes
choisissent les mêmes options.

## Homework 3 — classification

| Question | Réponse |
| --- | --- |
| Q1. Mode de `industry` | `technology` |
| Q2. Paire la plus corrélée | `interaction_count` et `lead_score` (0.916) |
| Q3. Plus forte information mutuelle | `lead_source` (0.03) |
| Q4. Accuracy validation | 0.65 |
| Q5. Feature la moins utile | `number_of_courses_viewed` |
| Q6. Meilleur `C` | 0.001 |

## Homework 4 — évaluation

| Question | Réponse |
| --- | --- |
| Q1. Meilleur AUC univarié | `lead_score` |
| Q2. AUC du modèle | 0.732 |
| Q3. Croisement précision / rappel | 0.63 |
| Q4. Seuil du F1 maximal | 0.41 |
| Q5. Écart-type des AUC en 5-fold | 0.007 |
| Q6. Meilleur `C` | 0.001 |

## Homework 5 — déploiement

| Question | Réponse |
| --- | --- |
| Q1. `uv --version` | uv 0.12.23 (contrôle local, non noté) |
| Q2. scikit-learn verrouillé | 1.7.2 |
| Q3. Probabilité du premier lead | 0.533 |
| Q4. Probabilité de l'API | 0.770 |
| Q5. Image de base | `python:3.11.15-slim-bookworm` |
| Q6. Même requête dans le conteneur | 0.770 |

`sha256sum pipeline.bin` correspond au checksum du devoir. `uv sync --locked`
puis `smoke_test.py` renvoient `conversion_probability` 0.769799.
L'API locale (`uvicorn` sur le port 9696) et le conteneur
`zoomcamp-model:2026-hw5` répondent la même valeur, avec
`{"status":"ok", ...}` et le checksum de `model_metadata.json`.

## Homework 6 — arbres et XGBoost

| Question | Réponse |
| --- | --- |
| Q1. Feature du premier split | `model_year` |
| Q2. RMSE, 10 arbres | 1.837 |
| Q3. Meilleur `n_estimators` | 100 |
| Q4. Meilleur `max_depth` | 10 |
| Q5. Feature la plus importante | `vehicle_weight` |
| Q6. Meilleur `eta` | 0.1 (RMSE 1.725 contre 1.826 pour 0.3) |

## Homework 8 — réseau de neurones

Entraînement CPU avec `reference_train.py`, PyTorch 2.9.0+cpu,
torchvision 0.24.0+cpu, seed 42. L'archive `data.zip` a le checksum attendu.
Les historiques sont dans `hw08_history/`.

| Question | Réponse |
| --- | --- |
| Q1. Fonction de perte | `nn.BCEWithLogitsLoss()` |
| Q2. Paramètres entraînables | 20073473 |
| Q3. Médiane de `train_accuracy` | 0.81 |
| Q4. Écart-type de `train_loss` | 0.139 |
| Q5. Moyenne de `evaluation_loss` augmenté | 0.575 |
| Q6. Moyenne des 5 dernières accuracies | 0.72 |

## Homework 9 — serverless

Les trois fichiers (ONNX, données externes, image) ont les checksums de
`asset_manifest.json`.

| Question | Réponse |
| --- | --- |
| Q1. Nom de la sortie ONNX | `output` |
| Q2. Taille cible | 200x200 |
| Q3. Première valeur du canal R | -1.056 |
| Q4. Probabilité `straight` | 0.728 |
| Q5. Image de base Lambda | `public.ecr.aws/lambda/python:3.13` |
| Q6. Invocation du conteneur | 0.728 |

`smoke_test.py` et le conteneur `mlzoomcamp-2026-serverless` renvoient
`straight_probability` 0.727697. Le déploiement AWS est optionnel et non noté.

## Homework 10 — Kubernetes

| Question | Réponse |
| --- | --- |
| Q1. Probabilité du conteneur | 0.770 |
| Q2. Versions locales | kind 0.27.0, kubectl client v1.37.1 |
| Q3. Plus petite unité | Pod |
| Q4. Type du service `kubernetes` | ClusterIP |
| Q5. Commande de chargement | `kind load docker-image` |
| Q6. Port du conteneur | 9696 |
| Q7. Sélecteur | `app: subscription` |
| Q8. `maxReplicas` | 3 |

Q1 est la même prédiction que HW5, vérifiée dans l'image
`zoomcamp-model:2026-hw10`. Q6, Q7 et Q8 sont lus dans les manifests.
Le cluster `kind` n'a pas pu démarrer dans cet environnement : kubeadm
n'arrive pas à rendre l'API saine (`context deadline exceeded`, cgroups
imbriqués). Q4 n'a donc pas été observée avec `kubectl get services` ;
ClusterIP est le type du service `kubernetes` créé par kubeadm dans un
cluster kind.
