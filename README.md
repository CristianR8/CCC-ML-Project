
## Metodos

El codigo conserva los metodos que aparecen en notebooks:

- Modelos: `DecisionTree`, `KNN`, `SVM`, `RandomForest`, `Bagging`, `AdaBoost`, `GradientBoosting`, `XGBoost`, `LightGBM`, `CatBoost`, `GaussianNB`.
- Balanceo: `RandomOverSampler`, `RandomUnderSampler`, `SMOTE`.
- Sinteticos: `GaussianCopulaSynthesizer`, `GaussianMixture`, `DP-GNB`, wrapper para `DataSynthesizer`.
- Seleccion/importancia: `GridSearchCV`, `permutation_importance`, `ANOVA (f_classif)`, `SequentialFeatureSelector`, `drop-column impact`.
- Metodos avanzados: `PSO` (pyswarms), `Fuzzy C-Means` (scikit-fuzzy).
- Analisis/proyecciones: `PCA`, `Isomap`, `LDA`.

Instalar con:

```bash
pip install -r requirements.txt
```
