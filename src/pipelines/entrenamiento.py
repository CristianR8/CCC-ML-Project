"""Pipeline for training experiments."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import config
from data.preprocessing import (
    build_full_columns_dataset,
    build_mean_imputed_dataset,
    prepare_complicaciones_dataframe,
)
from modeling.evaluation import run_train_test_models
from modeling.model_registry import build_baseline_models
from modeling.sampling import apply_sampling
from pipelines.common import ensure_dir, write_df


DEFAULT_METHODS = [
    "full_columns",
    "mean_imputation",
    "undersampling",
    "oversampling",
    "smote",
]


def _models_for_target(y: pd.Series, random_state: int) -> dict[str, object]:
    classes = y.unique()
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    class_weights = dict(zip(classes.tolist(), weights.tolist()))

    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else None

    return build_baseline_models(
        random_state=random_state,
        class_weights=class_weights,
        scale_pos_weight=scale_pos_weight,
    )


def run_entrenamiento_pipeline(
    data_path: str | Path,
    output_dir: str | Path | None = None,
    methods: list[str] | None = None,
    random_state: int = config.DEFAULT_RANDOM_STATE,
) -> dict[str, object]:
    if not methods or "all" in methods:
        methods = DEFAULT_METHODS

    out = ensure_dir(output_dir)
    exported: dict[str, object] = {}

    df = prepare_complicaciones_dataframe(data_path)
    X_full, y_full = build_full_columns_dataset(df, config.TARGET_COMPLICACIONES)
    X_mean, y_mean = build_mean_imputed_dataset(df, config.TARGET_COMPLICACIONES)

    if "full_columns" in methods:
        X_train, X_test, y_train, y_test = train_test_split(
            X_full,
            y_full,
            test_size=0.2,
            random_state=random_state,
            stratify=y_full,
        )
        models = _models_for_target(y_train, random_state=random_state)
        eval_df = run_train_test_models(X_train, y_train, X_test, y_test, models=models)
        path = (out / "entrenamiento_full_columns.csv") if out is not None else None
        write_df(eval_df, path)
        exported["full_columns"] = path if path is not None else eval_df

    if "mean_imputation" in methods:
        X_train, X_test, y_train, y_test = train_test_split(
            X_mean,
            y_mean,
            test_size=0.2,
            random_state=random_state,
            stratify=y_mean,
        )
        models = _models_for_target(y_train, random_state=random_state)
        eval_df = run_train_test_models(X_train, y_train, X_test, y_test, models=models)
        path = (out / "entrenamiento_mean_imputation.csv") if out is not None else None
        write_df(eval_df, path)
        exported["mean_imputation"] = path if path is not None else eval_df

    if "undersampling" in methods or "oversampling" in methods or "smote" in methods:
        X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(
            X_mean,
            y_mean,
            test_size=0.2,
            random_state=random_state,
            stratify=y_mean,
        )

        for method, key in [
            ("under", "undersampling"),
            ("over", "oversampling"),
            ("smote", "smote"),
        ]:
            if key not in methods:
                continue
            X_train, y_train = apply_sampling(X_train_base, y_train_base, method=method, random_state=random_state)
            models = _models_for_target(y_train, random_state=random_state)
            eval_df = run_train_test_models(X_train, y_train, X_test_base, y_test_base, models=models)
            path = (out / f"entrenamiento_{key}.csv") if out is not None else None
            write_df(eval_df, path)
            exported[key] = path if path is not None else eval_df

    return exported
