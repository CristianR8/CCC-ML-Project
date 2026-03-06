"""Model and hyperparameter registries."""

from __future__ import annotations

from typing import Any

from sklearn.ensemble import (
    AdaBoostClassifier,
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

try:
    from xgboost import XGBClassifier
except ImportError:  
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
except ImportError:  
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
except ImportError: 
    CatBoostClassifier = None


def _catboost_weights(class_weights: dict[int, float] | None) -> list[float] | None:
    if not class_weights:
        return None
    return [class_weights.get(0, 1.0), class_weights.get(1, 1.0)]


def _optional_models(
    random_state: int,
    class_weights: dict[int, float] | None,
    scale_pos_weight: float | None,
) -> dict[str, Any]:
    models: dict[str, Any] = {}

    if XGBClassifier is not None:
        xgb_args: dict[str, Any] = {
            "n_estimators": 200,
            "learning_rate": 0.01,
            "max_depth": 3,
            "reg_lambda": 5,
            "eval_metric": "logloss",
            "random_state": random_state,
        }
        if scale_pos_weight is not None:
            xgb_args["scale_pos_weight"] = scale_pos_weight
        models["XGBoost"] = XGBClassifier(**xgb_args)

    if LGBMClassifier is not None:
        lgb_args: dict[str, Any] = {
            "n_estimators": 200,
            "learning_rate": 0.01,
            "num_leaves": 31,
            "max_depth": -1,
            "verbose": -1,
            "random_state": random_state,
        }
        if class_weights:
            lgb_args["class_weight"] = class_weights
            lgb_args["is_unbalance"] = True
        models["LightGBM"] = LGBMClassifier(**lgb_args)

    if CatBoostClassifier is not None:
        cat_args: dict[str, Any] = {
            "iterations": 200,
            "learning_rate": 0.01,
            "depth": 4,
            "l2_leaf_reg": 7,
            "verbose": 0,
            "random_state": random_state,
        }
        weights = _catboost_weights(class_weights)
        if weights is not None:
            cat_args["class_weights"] = weights
        models["CatBoost"] = CatBoostClassifier(**cat_args)

    return models


def build_baseline_models(
    random_state: int = 42,
    class_weights: dict[int, float] | None = None,
    scale_pos_weight: float | None = None,
) -> dict[str, Any]:
    """Main model catalog."""
    models: dict[str, Any] = {
        "KNN": KNeighborsClassifier(leaf_size=20, n_neighbors=3, p=1, weights="uniform"),
        "SVM (RBF)": SVC(
            C=1,
            gamma="scale",
            kernel="rbf",
            probability=True,
            random_state=random_state,
            class_weight=class_weights,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            min_samples_leaf=2,
            min_samples_split=5,
            bootstrap=False,
            class_weight=class_weights,
            random_state=random_state,
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight=class_weights,
            criterion="gini",
            max_depth=5,
            min_samples_leaf=4,
            min_samples_split=2,
            random_state=random_state,
        ),
        "Bagging (DT)": BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=random_state),
            n_estimators=50,
            max_samples=1.0,
            max_features=1.0,
            bootstrap=False,
            random_state=random_state,
        ),
        "AdaBoost (DT)": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=5, random_state=random_state),
            n_estimators=50,
            learning_rate=0.1,
            random_state=random_state,
        ),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=50, random_state=random_state),
        "GaussianNB": GaussianNB(),
    }
    models.update(_optional_models(random_state, class_weights, scale_pos_weight))
    return models


def build_tuned_models(
    random_state: int = 42,
    class_weights: dict[int, float] | None = None,
    scale_pos_weight: float | None = None,
) -> dict[str, Any]:
    """Alternative tuned profile for feature-importance workflows."""
    models: dict[str, Any] = {
        "KNN": KNeighborsClassifier(n_neighbors=3, p=1, weights="uniform", leaf_size=20),
        "SVM (RBF)": SVC(
            C=100,
            gamma="scale",
            kernel="rbf",
            probability=True,
            random_state=random_state,
            class_weight=class_weights,
        ),
        "Random Forest": RandomForestClassifier(
            class_weight=class_weights,
            max_depth=5,
            n_estimators=100,
            random_state=random_state,
        ),
        "Decision Tree": DecisionTreeClassifier(
            class_weight=class_weights,
            criterion="entropy",
            max_depth=5,
            random_state=random_state,
        ),
        "Bagging (DT)": BaggingClassifier(
            estimator=DecisionTreeClassifier(random_state=random_state),
            max_samples=1.0,
            n_estimators=50,
            random_state=random_state,
        ),
        "AdaBoost (DT)": AdaBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=5, random_state=random_state),
            learning_rate=0.01,
            n_estimators=200,
            random_state=random_state,
        ),
    }
    models.update(_optional_models(random_state, class_weights, scale_pos_weight))
    return models


def build_param_grids() -> dict[str, dict[str, list[Any]]]:
    """GridSearchCV parameter space for Mortalidad and Complicaciones."""
    return {
        "KNN": {
            "n_neighbors": [3, 5, 7, 9, 11],
            "p": [1, 2],
            "weights": ["uniform", "distance"],
            "leaf_size": [20, 30, 40],
        },
        "SVM (RBF)": {
            "C": [0.1, 1, 10, 100, 1000],
            "gamma": ["scale", "auto", 0.1, 0.01, 0.001],
            "kernel": ["rbf"],
        },
        "Random Forest": {
            "n_estimators": [100, 200, 400],
            "max_depth": [5, 10, 20, None],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "bootstrap": [True, False],
            "class_weight": ["balanced"],
        },
        "Decision Tree": {
            "max_depth": [3, 5, 10, 20, None],
            "criterion": ["gini", "entropy", "log_loss"],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "class_weight": ["balanced"],
        },
        "Bagging (DT)": {
            "n_estimators": [50, 100, 200],
            "max_samples": [0.6, 0.8, 1.0],
            "max_features": [0.6, 0.8, 1.0],
            "bootstrap": [True, False],
        },
        "AdaBoost (DT)": {
            "n_estimators": [50, 100, 200, 400],
            "learning_rate": [0.001, 0.01, 0.1, 1.0],
            "estimator__max_depth": [1, 2, 3, 5],
        },
        "XGBoost": {
            "n_estimators": [100, 200],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.1],
            "reg_lambda": [1, 5],
        },
        "LightGBM": {
            "n_estimators": [200, 300],
            "learning_rate": [0.01, 0.1],
            "num_leaves": [31, 63],
            "max_depth": [-1, 10],
        },
        "CatBoost": {
            "depth": [4, 6, 8],
            "learning_rate": [0.01, 0.1],
            "l2_leaf_reg": [3, 5, 7],
        },
    }
