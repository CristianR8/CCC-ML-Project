"""Feature importance and feature-selection methods"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import SequentialFeatureSelector, f_classif
from sklearn.inspection import permutation_importance
from sklearn.preprocessing import StandardScaler

import config
from modeling.model_registry import build_tuned_models


def to_importance_df(importances: Any, columns: pd.Index) -> pd.DataFrame:
    """Normalize different importance outputs into one DataFrame schema."""
    if isinstance(importances, pd.Series):
        df = importances.reset_index()
        df.columns = ["feature", "importance"]
    elif isinstance(importances, np.ndarray):
        df = pd.DataFrame({"feature": list(columns), "importance": importances})
    elif isinstance(importances, pd.DataFrame):
        df = importances[["feature", "importance"]].copy()
    else:
        raise TypeError(f"Unsupported importances type: {type(importances)!r}")

    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def aggregate_tree_importances(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = config.DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Average importances across tree-based models."""
    models = build_tuned_models(random_state=random_state)
    all_importances: list[pd.DataFrame] = []

    for name, model in models.items():
        if name in config.SCALE_REQUIRED_MODELS:
            continue

        fitted = deepcopy(model)
        fitted.fit(X_train, y_train)

        if hasattr(fitted, "feature_importances_"):
            raw = np.asarray(fitted.feature_importances_, dtype=float)
            if raw.sum() > 0:
                raw = raw / raw.sum()
            df = to_importance_df(raw, X_train.columns)
        elif name == "Bagging (DT)" and hasattr(fitted, "estimators_"):
            bag_raw = np.mean([tree.feature_importances_ for tree in fitted.estimators_], axis=0)
            if bag_raw.sum() > 0:
                bag_raw = bag_raw / bag_raw.sum()
            df = to_importance_df(bag_raw, X_train.columns)
        else:
            continue

        df["model"] = name
        all_importances.append(df)

    if not all_importances:
        return pd.DataFrame(), pd.DataFrame()

    combined = pd.concat(all_importances, ignore_index=True)
    avg = (
        combined.groupby("feature")["importance"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    return combined, avg


def aggregate_permutation_importances(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    models: dict[str, Any] | None = None,
    random_state: int = config.DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Permutation importance averaged across all configured models."""
    if models is None:
        models = build_tuned_models(random_state=random_state)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    all_importances: list[pd.DataFrame] = []
    for name, model in models.items():
        fitted = deepcopy(model)

        if name in config.SCALE_REQUIRED_MODELS:
            fitted.fit(X_train_scaled, y_train)
            result = permutation_importance(
                fitted,
                X_test_scaled,
                y_test,
                n_repeats=10,
                random_state=random_state,
                scoring="accuracy",
            )
        else:
            fitted.fit(X_train, y_train)
            result = permutation_importance(
                fitted,
                X_test,
                y_test,
                n_repeats=10,
                random_state=random_state,
                scoring="accuracy",
            )

        df = to_importance_df(result.importances_mean, X_train.columns)
        df["model"] = name
        all_importances.append(df)

    combined = pd.concat(all_importances, ignore_index=True)
    avg = (
        combined.groupby("feature")["importance"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    return combined, avg


def anova_feature_scores(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """ANOVA F-test ranking."""
    f_values, p_values = f_classif(X, y)
    out = pd.DataFrame({"Feature": X.columns, "F_value": f_values, "p_value": p_values})
    return out.sort_values("F_value", ascending=False).reset_index(drop=True)


def sequential_feature_selection_by_model(
    X: pd.DataFrame,
    y: pd.Series,
    models: dict[str, Any] | None = None,
    cv: int = 5,
) -> dict[str, list[str]]:
    """Forward SFS for each model."""
    if models is None:
        models = build_tuned_models()

    selected: dict[str, list[str]] = {}
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    for name, model in models.items():
        selector = SequentialFeatureSelector(
            model,
            n_features_to_select="auto",
            direction="forward",
            cv=cv,
            n_jobs=-1,
        )

        if name in config.SCALE_REQUIRED_MODELS:
            selector.fit(X_scaled, y)
        else:
            selector.fit(X, y)

        selected[name] = X.columns[selector.get_support()].tolist()

    return selected


def drop_column_impact(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    model: Any,
) -> pd.DataFrame:
    """Performance-drop method (leave-one-feature-out)."""
    fitted = deepcopy(model)
    fitted.fit(X_train, y_train)
    base_score = fitted.score(X_val, y_val)

    impact_rows: list[tuple[str, float]] = []
    for col in X_train.columns:
        X_train_tmp = X_train.drop(columns=[col])
        X_val_tmp = X_val.drop(columns=[col])
        tmp_model = deepcopy(model)
        tmp_model.fit(X_train_tmp, y_train)
        score = tmp_model.score(X_val_tmp, y_val)
        impact_rows.append((col, base_score - score))

    impact_df = pd.DataFrame(impact_rows, columns=["Feature", "Performance Drop"])
    return impact_df.sort_values("Performance Drop", ascending=False).reset_index(drop=True)
