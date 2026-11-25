# Prédiction de la Bactériémie (Event Prediction)

Développement et validation de modèles de Machine Learning (dont un **Super Learner**) et de Deep Learning (**PyTorch**) pour prédire la survenue d'une bactériémie à partir de données biologiques cliniques fortement déséquilibrées.

## Contexte & Données

* **Source :** Hôpital général de Vienne (Autriche, 2006–2010), version anonymisée approuvée pour un usage public (DC 2019-0054).
* **Volume :** 11 753 patients présentant une suspicion clinique de bactériémie, décrits par **51 variables explicatives** (âge, sexe, bilan sanguin).
* **Variable cible (`BloodCulture`) :** Résultat de l'hémoculture — **8,0 % de cas positifs** (944 positifs vs 10 809 négatifs).
* **Données manquantes :** 73 % des patients présentent au moins une valeur manquante (jusqu'à 48 % sur `PAMY`).

## Méthodologie

1. **Nettoyage clinique :** Remplacement par `NaN` des valeurs physiologiquement impossibles (`HCT == 0 %` et `POTASS > 10 mmol/L`) tout en conservant les valeurs extrêmes cliniquement plausibles (ex. `CK` élevée).
2. **Validation sans fuite de données :** Découpage stratifié 80 % apprentissage (9 402 patients) / 20 % test interne (2 351 patients) et validation croisée stratifiée à 5 plis (`StratifiedKFold`).
3. **Prétraitement (`QuantileTransformer` → `SimpleImputer`) :**
   * Compétition de 4 scalers (`StandardScaler`, `MinMaxScaler`, `RobustScaler`, `QuantileTransformer`) sur l'AUC-PR : le **`QuantileTransformer`** (distribution normale) obtient le meilleur score.
   * Imputation par la médiane avec ajout d'indicateurs de données manquantes (`add_indicator=True`).
   * *Choix clé :* Application du scaler **avant** l'imputation afin que les colonnes indicatrices restent binaires (0/1) et ne soient pas projetées à ±5,2 par le `QuantileTransformer`.
4. **Modélisation :**
   * **Notebook 1 (Machine Learning & Ensembling) :** Régression Logistique, KNN, Random Forest, XGBoost, LightGBM (optimisés par `GridSearchCV` sur l'AUC-PR), puis **Soft Voting** et **Super Learner (Stacking)** combinant 3 familles distinctes (Régression Logistique + KNN + LightGBM).
   * **Notebook 2 (Deep Learning) :** Réseau de neurones MLP sous **PyTorch** (2 couches cachées 32 → 16 neurones, ReLU, `BCEWithLogitsLoss`, optimiseur Adam avec `weight_decay=0.01`, 20 époques).
5. **Gestion du déséquilibre :** Optimisation du seuil de décision sur les probabilités issues de la validation croisée (`cross_val_predict`) pour maximiser le **F1-score**.

## Résultats

Le modèle final a été sélectionné sur le **F1-score en validation croisée** (critère le plus robuste face à la variance du test interne).

| Modèle | Seuil | F1 (CV) | F1 (Test) | Précision (Test) | Rappel (Test) | AUC-ROC (Test) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Super Learner (Modèle final)** | **0.15** | **0.360** | **0.359** | **0.343** | **0.376** | **0.776** |
| Soft Voting | 0.16 | 0.354 | 0.353 | 0.329 | 0.381 | 0.776 |
| Random Forest | 0.18 | 0.346 | 0.340 | 0.304 | 0.386 | 0.753 |
| PyTorch MLP (32 → 16) | 0.19 | 0.344 | 0.329 | 0.307 | 0.354 | 0.768 |
| Régression Logistique | 0.16 | 0.343 | 0.339 | 0.283 | 0.423 | 0.770 |
| LightGBM | 0.18 | 0.343 | 0.363 | 0.335 | 0.397 | 0.771 |
| XGBoost | 0.15 | 0.342 | 0.346 | 0.288 | 0.434 | 0.778 |
| KNN | 0.12 | 0.311 | 0.295 | 0.232 | 0.402 | 0.750 |

### Indicateurs cliniques du modèle final (Super Learner sur le test interne)
* **Sensibilité (Rappel) :** 37,6 % (VP = 71, FN = 118)
* **Spécificité :** 93,7 % (VN = 2 026, FP = 136)
* **VPP (Précision) :** 34,3 % | **VPN :** 94,5 %
