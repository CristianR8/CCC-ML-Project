"""Training and evaluation loops"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import config
from modeling.metrics import calculate_binary_metrics
from modeling.model_registry import build_baseline_models, build_param_grids
from modeling.sampling import apply_sampling


ModelBuilder = Callable[[int, dict[int, float] | None, float | None], dict[str, Any]]


def _safe_score_vector(model: Any, X_eval: Any) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X_eval)[:, 1]
    if hasattr(model, "decision_function"):
        decision = model.decision_function(X_eval)
        return 1.0 / (1.0 + np.exp(-decision))
    return model.predict(X_eval)


def _cv_score_vector(model: Any, X: pd.DataFrame, y: pd.Series, cv: StratifiedKFold) -> np.ndarray:
    try:
        return cross_val_predict(model, X, y, cv=cv, method="predict_proba")[:, 1]
    except Exception:
        try:
            decision = cross_val_predict(model, X, y, cv=cv, method="decision_function")
            return 1.0 / (1.0 + np.exp(-decision))
        except Exception:
            return cross_val_predict(model, X, y, cv=cv, method="predict")


def _compute_weights(y: pd.Series) -> tuple[dict[int, float], float | None]:
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    class_weights = dict(zip(classes.tolist(), weights.tolist()))
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else None
    return class_weights, scale_pos_weight


def run_train_test_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    models: dict[str, Any],
    scale_model_names: set[str] = config.SCALE_REQUIRED_MODELS,
) -> pd.DataFrame:
    """Train and evaluate all models in a single train/test split."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    rows: list[dict[str, Any]] = []

    for name, model in models.items():
        fitted = deepcopy(model)
        if name in scale_model_names:
            fitted.fit(X_train_scaled, y_train)
            y_pred = fitted.predict(X_test_scaled)
            y_score = _safe_score_vector(fitted, X_test_scaled)
        else:
            fitted.fit(X_train, y_train)
            y_pred = fitted.predict(X_test)
            y_score = _safe_score_vector(fitted, X_test)

        rows.append(calculate_binary_metrics(y_test, y_pred, y_score, model_name=name))

    return pd.DataFrame(rows).sort_values("F1-Score", ascending=False).reset_index(drop=True)


def run_kfold_models(
    X: pd.DataFrame,
    y: pd.Series,
    models: dict[str, Any],
    k: int = 5,
    sampling: str | None = None,
    random_state: int = config.DEFAULT_RANDOM_STATE,
    scale_model_names: set[str] = config.SCALE_REQUIRED_MODELS,
    evaluate_full_dataset: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]], pd.DataFrame | None]:
    """Stratified K-fold with optional sampling."""
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=random_state)

    fold_metrics: list[dict[str, Any]] = []
    full_metrics: list[dict[str, Any]] = []
    top_models: list[dict[str, Any]] = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), start=1):
        X_train = X.iloc[train_idx].copy()
        X_test = X.iloc[test_idx].copy()
        y_train = y.iloc[train_idx].copy()
        y_test = y.iloc[test_idx].copy()

        train_means = X_train.mean(numeric_only=True)
        X_train = X_train.fillna(train_means)
        X_test = X_test.fillna(train_means)

        if sampling is not None:
            X_train, y_train = apply_sampling(X_train, y_train, sampling, random_state=random_state)

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        X_full = X.fillna(train_means)
        X_full_scaled = scaler.transform(X_full)

        for name, model in models.items():
            fitted = deepcopy(model)

            if name in scale_model_names:
                fitted.fit(X_train_scaled, y_train)
                y_pred = fitted.predict(X_test_scaled)
                y_score = _safe_score_vector(fitted, X_test_scaled)
                y_pred_full = fitted.predict(X_full_scaled)
                y_score_full = _safe_score_vector(fitted, X_full_scaled)
            else:
                fitted.fit(X_train, y_train)
                y_pred = fitted.predict(X_test)
                y_score = _safe_score_vector(fitted, X_test)
                y_pred_full = fitted.predict(X_full)
                y_score_full = _safe_score_vector(fitted, X_full)

            metrics = calculate_binary_metrics(
                y_test,
                y_pred,
                y_score,
                model_name=name,
                fold=fold,
            )
            fold_metrics.append(metrics)

            top_models.append(
                {
                    "f1": metrics["F1-Score"],
                    "model_name": name,
                    "fold": fold,
                    "fitted_model": fitted,
                    "scaler": scaler if name in scale_model_names else None,
                    "train_means": train_means,
                    "requires_scaling": name in scale_model_names,
                }
            )

            if evaluate_full_dataset:
                full_metrics.append(
                    calculate_binary_metrics(
                        y,
                        y_pred_full,
                        y_score_full,
                        model_name=name,
                        fold=fold,
                    )
                )

    fold_df = pd.DataFrame(fold_metrics)
    avg_df = (
        fold_df.drop(columns=["Fold"]).groupby("Model").mean(numeric_only=True).sort_values("F1-Score", ascending=False)
    )

    top_models = sorted(top_models, key=lambda row: row["f1"], reverse=True)

    full_df: pd.DataFrame | None = None
    if evaluate_full_dataset and full_metrics:
        full_df = (
            pd.DataFrame(full_metrics)
            .drop(columns=["Fold"]) 
            .groupby("Model")
            .mean(numeric_only=True)
            .sort_values("F1-Score", ascending=False)
        )

    return avg_df, fold_df, top_models, full_df


def evaluate_top_models_on_full_dataset(
    X: pd.DataFrame,
    y: pd.Series,
    top_models: list[dict[str, Any]],
    top_n: int = 3,
) -> pd.DataFrame:
    """Evaluate best fold-specific models on full dataset."""
    selected: list[dict[str, Any]] = []
    used_names: set[str] = set()

    for item in top_models:
        if item["model_name"] in used_names:
            continue
        selected.append(item)
        used_names.add(item["model_name"])
        if len(selected) == top_n:
            break

    rows: list[dict[str, Any]] = []
    for item in selected:
        train_means = item["train_means"]
        model = item["fitted_model"]

        X_eval = X.fillna(train_means)
        if item["requires_scaling"]:
            X_eval = item["scaler"].transform(X_eval)

        y_pred = model.predict(X_eval)
        y_score = _safe_score_vector(model, X_eval)

        metrics = calculate_binary_metrics(y, y_pred, y_score, model_name=item["model_name"])
        metrics["Fold_Source"] = item["fold"]
        metrics["F1_Score_CrossVal"] = item["f1"]
        rows.append(metrics)

    return pd.DataFrame(rows).sort_values("F1-Score", ascending=False).reset_index(drop=True)


def run_gridsearch_by_completeness(
    df: pd.DataFrame,
    target: str,
    completeness_levels: list[float] | None = None,
    model_builder: ModelBuilder = build_baseline_models,
    param_grids: dict[str, dict[str, Any]] | None = None,
    random_state: int = config.DEFAULT_RANDOM_STATE,
) -> pd.DataFrame:
    """GridSearchCV + out-of-fold metrics loop."""
    if completeness_levels is None:
        completeness_levels = [1.0]
    if param_grids is None:
        param_grids = build_param_grids()

    completeness = df.notna().mean()
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
    rows: list[dict[str, Any]] = []

    for level in completeness_levels:
        data = df.loc[:, completeness >= level].dropna()
        if target not in data.columns:
            continue

        X = data.drop(columns=[target])
        y = data[target]

        class_weights, scale_pos_weight = _compute_weights(y)
        models = model_builder(random_state, class_weights, scale_pos_weight)

        for name, model in models.items():
            grid = param_grids.get(name)
            if not grid:
                continue

            try:
                grid_search = GridSearchCV(
                    model,
                    grid,
                    cv=skf,
                    scoring="f1",
                    n_jobs=-1,
                    verbose=0,
                )
                grid_search.fit(X, y)
                best_model = grid_search.best_estimator_

                y_pred = cross_val_predict(best_model, X, y, cv=skf, method="predict")
                y_score = _cv_score_vector(best_model, X, y, cv=skf)
                rows.append(
                    calculate_binary_metrics(
                        y,
                        y_pred,
                        y_score,
                        model_name=name,
                        completeness=level,
                    )
                )
            except Exception:
                continue

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(["Completeness", "F1-Score"], ascending=[False, False]).reset_index(drop=True)
