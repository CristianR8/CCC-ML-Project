"""Pipeline for the 'Mortalidad menor a 2 años' target."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

import config
from analysis.exploration import pca_before_after_sampling
from data.preprocessing import (
    build_full_columns_dataset,
    build_mean_imputed_dataset,
    prepare_mortalidad_dataframe,
    train_val_test_split,
)
from modeling.evaluation import (
    evaluate_top_models_on_full_dataset,
    run_gridsearch_by_completeness,
    run_kfold_models,
    run_train_test_models,
)
from modeling.feature_selection import (
    aggregate_permutation_importances,
    aggregate_tree_importances,
    anova_feature_scores,
    drop_column_impact,
    sequential_feature_selection_by_model,
)
from modeling.model_registry import build_baseline_models, build_param_grids
from modeling.sampling import apply_sampling
from modeling.synthetic import (
    generate_minority_synthetic_data,
    gmm_generate_minority_samples,
    gmm_oversampling_train_test,
    private_gaussian_naive_bayes_params,
)
from pipelines.common import ensure_dir, write_df


DEFAULT_METHODS = [
    "gridsearch",
    "feature_tree",
    "feature_perm",
    "kfold",
    "kfold_sampling",
    "anova",
    "sfs",
    "drop_column_impact",
    "gmm",
    "gmm_visual",
    "dp_gnb",
    "pca_smote",
]


def _weights_from_target(y: pd.Series) -> tuple[dict[int, float], float | None]:
    classes = np.unique(y)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    class_weights = dict(zip(classes.tolist(), weights.tolist()))
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    scale_pos_weight = (n_neg / n_pos) if n_pos > 0 else None
    return class_weights, scale_pos_weight


def _models_for_target(y: pd.Series, random_state: int) -> dict[str, object]:
    class_weights, scale_pos_weight = _weights_from_target(y)
    return build_baseline_models(
        random_state=random_state,
        class_weights=class_weights,
        scale_pos_weight=scale_pos_weight,
    )


def _save_npz(path: Path | None, **kwargs: np.ndarray) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **kwargs)


def run_mortalidad_pipeline(
    data_path: str | Path,
    output_dir: str | Path | None = None,
    methods: list[str] | None = None,
    random_state: int = config.DEFAULT_RANDOM_STATE,
) -> dict[str, object]:
    """Run selected methods for Mortalidad target.

    If ``output_dir`` is ``None``, results are returned in-memory and nothing is written to disk.
    """
    if not methods or "all" in methods:
        methods = DEFAULT_METHODS

    out = ensure_dir(output_dir)
    exported: dict[str, object] = {}

    df = prepare_mortalidad_dataframe(data_path)
    X_full, y_full = build_full_columns_dataset(df, config.TARGET_MORTALIDAD)
    X_mean, y_mean = build_mean_imputed_dataset(df, config.TARGET_MORTALIDAD)

    if "gridsearch" in methods:
        grid = run_gridsearch_by_completeness(
            df,
            target=config.TARGET_MORTALIDAD,
            completeness_levels=[1.0],
            param_grids=build_param_grids(),
            random_state=random_state,
        )
        path = (out / "mortalidad_gridsearch_completeness.csv") if out is not None else None
        write_df(grid, path)
        exported["gridsearch"] = path if path is not None else grid

    if "feature_tree" in methods or "feature_perm" in methods:
        X_train, X_test, y_train, y_test = train_test_split(
            X_mean,
            y_mean,
            test_size=0.2,
            random_state=random_state,
            stratify=y_mean,
        )

        if "feature_tree" in methods:
            tree_all, tree_avg = aggregate_tree_importances(X_train, y_train, random_state=random_state)
            path_all = (out / "mortalidad_tree_importances_all.csv") if out is not None else None
            path_avg = (out / "mortalidad_tree_importances_avg.csv") if out is not None else None
            write_df(tree_all, path_all)
            write_df(tree_avg, path_avg)
            exported["feature_tree_all"] = path_all if path_all is not None else tree_all
            exported["feature_tree_avg"] = path_avg if path_avg is not None else tree_avg

        if "feature_perm" in methods:
            models = _models_for_target(y_mean, random_state=random_state)
            perm_all, perm_avg = aggregate_permutation_importances(
                X_train,
                y_train,
                X_test,
                y_test,
                models=models,
                random_state=random_state,
            )
            path_all = (out / "mortalidad_permutation_importances_all.csv") if out is not None else None
            path_avg = (out / "mortalidad_permutation_importances_avg.csv") if out is not None else None
            write_df(perm_all, path_all)
            write_df(perm_avg, path_avg)
            exported["feature_perm_all"] = path_all if path_all is not None else perm_all
            exported["feature_perm_avg"] = path_avg if path_avg is not None else perm_avg

    if "kfold" in methods:
        models = _models_for_target(y_mean, random_state=random_state)
        avg_df, fold_df, top_models, full_df = run_kfold_models(
            X_mean,
            y_mean,
            models=models,
            k=10,
            sampling=None,
            random_state=random_state,
            evaluate_full_dataset=True,
        )
        avg_path = (out / "mortalidad_kfold_avg.csv") if out is not None else None
        fold_path = (out / "mortalidad_kfold_folds.csv") if out is not None else None
        top_path = (out / "mortalidad_kfold_top3_full_eval.csv") if out is not None else None
        full_path = (out / "mortalidad_kfold_full_dataset_avg.csv") if out is not None else None

        write_df(avg_df.reset_index(), avg_path)
        write_df(fold_df, fold_path)
        top_eval = evaluate_top_models_on_full_dataset(X_mean, y_mean, top_models, top_n=3)
        write_df(top_eval, top_path)
        if full_df is not None:
            write_df(full_df.reset_index(), full_path)
            exported["kfold_full"] = full_path if full_path is not None else full_df.reset_index()

        exported["kfold_avg"] = avg_path if avg_path is not None else avg_df.reset_index()
        exported["kfold_folds"] = fold_path if fold_path is not None else fold_df
        exported["kfold_top3"] = top_path if top_path is not None else top_eval

    if "kfold_sampling" in methods:
        for sampling in ["over", "under", "smote"]:
            models = _models_for_target(y_mean, random_state=random_state)
            avg_df, fold_df, top_models, full_df = run_kfold_models(
                X_mean,
                y_mean,
                models=models,
                k=10,
                sampling=sampling,
                random_state=random_state,
                evaluate_full_dataset=True,
            )
            avg_path = (out / f"mortalidad_kfold_{sampling}_avg.csv") if out is not None else None
            fold_path = (out / f"mortalidad_kfold_{sampling}_folds.csv") if out is not None else None
            top_path = (out / f"mortalidad_kfold_{sampling}_top3_full_eval.csv") if out is not None else None

            write_df(avg_df.reset_index(), avg_path)
            write_df(fold_df, fold_path)
            top_eval = evaluate_top_models_on_full_dataset(X_mean, y_mean, top_models, top_n=3)
            write_df(top_eval, top_path)
            if full_df is not None:
                full_path = (out / f"mortalidad_kfold_{sampling}_full_dataset_avg.csv") if out is not None else None
                write_df(full_df.reset_index(), full_path)

            exported[f"kfold_{sampling}"] = avg_path if avg_path is not None else avg_df.reset_index()

    if "anova" in methods:
        anova_df = anova_feature_scores(X_mean, y_mean)
        path = (out / "mortalidad_anova_scores.csv") if out is not None else None
        write_df(anova_df, path)
        exported["anova"] = path if path is not None else anova_df

    if "sfs" in methods:
        selected = sequential_feature_selection_by_model(X_mean, y_mean)
        path = (out / "mortalidad_sequential_feature_selection.json") if out is not None else None
        if path is not None:
            path.write_text(json.dumps(selected, indent=2), encoding="utf-8")
        exported["sfs"] = path if path is not None else selected

    if "drop_column_impact" in methods:
        X_train, X_val, X_test, y_train, y_val, _ = train_val_test_split(
            X_mean,
            y_mean,
            test_size=0.3,
            val_size_from_temp=0.5,
            random_state=random_state,
        )
        rf_model = _models_for_target(y_train, random_state=random_state)["Random Forest"]
        impact_df = drop_column_impact(X_train, X_val, y_train, y_val, model=rf_model)
        path = (out / "mortalidad_drop_column_impact.csv") if out is not None else None
        write_df(impact_df, path)
        exported["drop_column_impact"] = path if path is not None else impact_df

    if "gmm" in methods:
        X_train, X_test, y_train, y_test = gmm_oversampling_train_test(
            X_mean,
            y_mean,
            test_size=0.3,
            synth_multiplier=1.0,
            n_components=3,
            random_state=random_state,
        )
        models = _models_for_target(y_train, random_state=random_state)
        gmm_eval = run_train_test_models(X_train, y_train, X_test, y_test, models=models)
        path = (out / "mortalidad_gmm_train_test.csv") if out is not None else None
        write_df(gmm_eval, path)
        exported["gmm"] = path if path is not None else gmm_eval

    if "gmm_visual" in methods:
        X_scaled, X_syn = gmm_generate_minority_samples(X_mean, y_mean, n_components=3, random_state=random_state)
        path = (out / "mortalidad_gmm_visual_data.npz") if out is not None else None
        _save_npz(path, X_scaled=X_scaled, X_syn=X_syn, y=y_mean.to_numpy())
        exported["gmm_visual"] = (
            path if path is not None else {"X_scaled": X_scaled, "X_syn": X_syn, "y": y_mean.to_numpy()}
        )

    if "dp_gnb" in methods:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_mean)
        y_array = y_mean.to_numpy()

        minority_class = int(min(np.unique(y_array), key=lambda c: (y_array == c).sum()))
        data_range = float(X_scaled.max() - X_scaled.min())

        private_params, _ = private_gaussian_naive_bayes_params(
            X_scaled,
            y_array,
            eps=1.0,
            data_range=data_range,
        )
        X_syn, y_syn = generate_minority_synthetic_data(private_params, y_array, minority_class, multiplier=3)
        path = (out / "mortalidad_dp_gnb_synthetic.npz") if out is not None else None
        _save_npz(path, X_scaled=X_scaled, y=y_array, X_syn=X_syn, y_syn=y_syn)
        exported["dp_gnb"] = (
            path
            if path is not None
            else {"X_scaled": X_scaled, "y": y_array, "X_syn": X_syn, "y_syn": y_syn}
        )

    if "pca_smote" in methods:
        X_res, y_res = apply_sampling(X_mean, y_mean, method="smote", random_state=random_state)
        X_pca, X_res_pca, pca = pca_before_after_sampling(X_mean, y_mean, X_res, y_res)
        path = (out / "mortalidad_pca_smote.npz") if out is not None else None
        _save_npz(
            path,
            X_pca=X_pca,
            X_res_pca=X_res_pca,
            explained_variance_ratio=pca.explained_variance_ratio_,
        )
        exported["pca_smote"] = (
            path
            if path is not None
            else {
                "X_pca": X_pca,
                "X_res_pca": X_res_pca,
                "explained_variance_ratio": pca.explained_variance_ratio_,
            }
        )

    if "full_columns_baseline" in methods:
        X_train, X_test, y_train, y_test = train_test_split(
            X_full,
            y_full,
            test_size=0.2,
            random_state=random_state,
            stratify=y_full,
        )
        models = _models_for_target(y_train, random_state=random_state)
        full_eval = run_train_test_models(X_train, y_train, X_test, y_test, models=models)
        path = (out / "mortalidad_full_columns_train_test.csv") if out is not None else None
        write_df(full_eval, path)
        exported["full_columns_baseline"] = path if path is not None else full_eval

    return exported
