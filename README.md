# Prédiction de la Demande Nette en Énergie Électrique

## Auteurs
- Félix OLLIVIER
- Lylian CHALLIER

## Date
10 mars 2025

## Résumé du Projet
Ce projet vise à prédire la demande nette en énergie électrique (`Net_demand`) en France en utilisant diverses approches de modélisation prédictive. L'objectif est de déterminer avec précision la consommation nette d'énergie à partir de données historiques couvrant la période jusqu'à fin 2021.

## Méthodologie

### Prétraitement des Données
- Conversion des variables catégorielles en facteurs
- Séparation des données en ensembles d'entraînement et d'évaluation

### Analyse Exploratoire des Données
- **Analyse unidimensionnelle** : Identification d'une forte saisonnalité annuelle dans la demande énergétique et observation de la relation entre la consommation totale (`Load`) et la somme de la demande nette et des énergies renouvelables
- **Analyse bidimensionnelle** : Étude des corrélations entre la demande nette et diverses variables catégorielles (jours fériés, vacances d'été) et continues (température, vent, nébulosité)

### Sélection des Variables
- **Corrélation linéaire** : Identification des variables ayant une corrélation significative (>0.3) avec la variable cible
- **Régression linéaire** : Comparaison de différentes stratégies de sélection (modèle complet, modèle réduit, sélection backward)

### Modélisation
- **Modèle GAM (Generalized Additive Model)** : Modélisation des relations non linéaires et cycliques
- **Modèles Random Forest** : Différentes approches de sélection de variables (complet, backward, mixte)

### Évaluation des Modèles
- **Métriques d'évaluation** : RMSE (Root Mean Square Error), MAPE (Mean Absolute Percentage Error), Pinball Loss
- **Comparaison des performances** : Tableau comparatif des différents modèles

## Résultats Principaux
- Identification de variables clés influençant la demande nette d'énergie (température, jours fériés, périodes de vacances)
- Le modèle GAM a montré de bonnes performances dans la capture des relations non linéaires
- Les modèles Random Forest ont également montré des résultats prometteurs

## Structure des Fichiers
- `rapport.Rmd` : Document principal contenant l'analyse complète et le code R
- `R/score.R` : Script contenant les fonctions d'évaluation des modèles
- `Data/train.csv` : Données d'entraînement
- `Data/test.csv` : Données de test pour la prédiction

## Dépendances
- Bibliothèques R requises : mgcv, corrplot, gt, tidyverse, ranger, randomForest, xgboost, yarrr

## Utilisation
Pour reproduire l'analyse, exécutez le code dans le fichier rapport.Rmd. Assurez-vous d'avoir installé toutes les dépendances nécessaires.

```r
# Installation des packages requis
install.packages(c("mgcv", "corrplot", "gt", "tidyverse", "ranger", "randomForest", "xgboost", "yarrr"))

# Exécution du rapport
rmarkdown::render("rapport.Rmd")
```
