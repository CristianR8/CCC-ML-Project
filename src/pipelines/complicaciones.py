"""Pipeline for the 'Complicaciones cardiovasculares' target."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

import config
from analysis.exploration import (
    isomap_before_after_sampling,
    pca_before_after_sampling,
)
from data.preprocessing import (
    build_full_columns_dataset,
    build_mean_imputed_dataset,
    prepare_complicaciones_dataframe,
)
from modeling.advanced import fuzzy_cmeans_severity, pso_feature_selection
from modeling.evaluation import (
    evaluate_top_models_on_full_dataset,
    run_gridsearch_by_completeness,
    run_kfold_models,
    run_train_test_models,
)
from modeling.feature_selection import (
    aggregate_permutation_importances,
    aggregate_tree_importances,
)
from modeling.model_registry import build_baseline_models, build_param_grids
from modeling.sampling import apply_sampling
from modeling.synthetic import gm_oversampling_train_test, gmm_oversampling_train_test
from pipelines.common import ensure_dir, write_df


DEFAULT_METHODS = [
    "gridsearch",
    "feature_tree",
    "feature_perm",
    "kfold",
    "kfold_sampling",
    "pso",
    "gaussian_copula",
    "gmm",
    "fuzzy",
    "pca_smote",
    "isomap_smote",
    "pca_gm",
    "pca_gmm",
    "mean_before_after_split",
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


def run_complicaciones_pipeline(
    data_path: str | Path,
    output_dir: str | Path | None = None,
    methods: list[str] | None = None,
    random_state: int = config.DEFAULT_RANDOM_STATE,
) -> dict[str, object]:
    """Run selected methods for Complicaciones target.

    If ``output_dir`` is ``None``, results are returned in-memory and nothing is written to disk.
    """
    if not methods or "all" in methods:
        methods = DEFAULT_METHODS

    out = ensure_dir(output_dir)
    exported: dict[str, object] = {}

    df = prepare_complicaciones_dataframe(data_path)
    X_full, y_full = build_full_columns_dataset(df, config.TARGET_COMPLICACIONES)
    X_mean, y_mean = build_mean_imputed_dataset(df, config.TARGET_COMPLICACIONES)

    if "gridsearch" in methods:
        grid = run_gridsearch_by_completeness(
            df,
            target=config.TARGET_COMPLICACIONES,
            completeness_levels=[1.0],
            param_grids=build_param_grids(),
            random_state=random_state,
        )
        path = (out / "complicaciones_gridsearch_completeness.csv") if out is not None else None
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
            path_all = (out / "complicaciones_tree_importances_all.csv") if out is not None else None
            path_avg = (out / "complicaciones_tree_importances_avg.csv") if out is not None else None
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
            path_all = (out / "complicaciones_permutation_importances_all.csv") if out is not None else None
            path_avg = (out / "complicaciones_permutation_importances_avg.csv") if out is not None else None
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
        avg_path = (out / "complicaciones_kfold_avg.csv") if out is not None else None
        fold_path = (out / "complicaciones_kfold_folds.csv") if out is not None else None
        top_path = (out / "complicaciones_kfold_top3_full_eval.csv") if out is not None else None
        full_path = (out / "complicaciones_kfold_full_dataset_avg.csv") if out is not None else None

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
            avg_path = (out / f"complicaciones_kfold_{sampling}_avg.csv") if out is not None else None
            fold_path = (out / f"complicaciones_kfold_{sampling}_folds.csv") if out is not None else None
            top_path = (out / f"complicaciones_kfold_{sampling}_top3_full_eval.csv") if out is not None else None
            full_path = (out / f"complicaciones_kfold_{sampling}_full_dataset_avg.csv") if out is not None else None

            write_df(avg_df.reset_index(), avg_path)
            write_df(fold_df, fold_path)
            top_eval = evaluate_top_models_on_full_dataset(X_mean, y_mean, top_models, top_n=3)
            write_df(top_eval, top_path)
            if full_df is not None:
                write_df(full_df.reset_index(), full_path)

            exported[f"kfold_{sampling}"] = avg_path if avg_path is not None else avg_df.reset_index()

    if "pso" in methods:
        pso_result = pso_feature_selection(X_mean, y_mean, random_state=random_state)
        path = (out / "complicaciones_pso_selection.json") if out is not None else None
        if path is not None:
            path.write_text(json.dumps(pso_result, indent=2), encoding="utf-8")
        exported["pso"] = path if path is not None else pso_result

    if "gaussian_copula" in methods:
        X_train, X_test, y_train, y_test = gm_oversampling_train_test(
            X_mean,
            y_mean,
            test_size=0.3,
            synth_multiplier=1.0,
            random_state=random_state,
        )
        models = _models_for_target(y_train, random_state=random_state)
        gm_eval = run_train_test_models(X_train, y_train, X_test, y_test, models=models)
        path = (out / "complicaciones_gaussian_copula_train_test.csv") if out is not None else None
        write_df(gm_eval, path)
        exported["gaussian_copula"] = path if path is not None else gm_eval

        if "pca_gm" in methods:
            X_pca, X_res_pca, pca = pca_before_after_sampling(X_mean, y_mean, X_train, y_train)
            pca_path = (out / "complicaciones_pca_gaussian_copula.npz") if out is not None else None
            _save_npz(
                pca_path,
                X_pca=X_pca,
                X_res_pca=X_res_pca,
                y_original=y_mean.to_numpy(),
                y_resampled=y_train.to_numpy(),
                explained_variance_ratio=pca.explained_variance_ratio_,
            )
            exported["pca_gm"] = (
                pca_path
                if pca_path is not None
                else {
                    "X_pca": X_pca,
                    "X_res_pca": X_res_pca,
                    "y_original": y_mean.to_numpy(),
                    "y_resampled": y_train.to_numpy(),
                    "explained_variance_ratio": pca.explained_variance_ratio_,
                }
            )

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
        path = (out / "complicaciones_gmm_train_test.csv") if out is not None else None
        write_df(gmm_eval, path)
        exported["gmm"] = path if path is not None else gmm_eval

        if "pca_gmm" in methods:
            X_pca, X_res_pca, pca = pca_before_after_sampling(X_mean, y_mean, X_train, y_train)
            pca_path = (out / "complicaciones_pca_gmm.npz") if out is not None else None
            _save_npz(
                pca_path,
                X_pca=X_pca,
                X_res_pca=X_res_pca,
                y_original=y_mean.to_numpy(),
                y_resampled=y_train.to_numpy(),
                explained_variance_ratio=pca.explained_variance_ratio_,
            )
            exported["pca_gmm"] = (
                pca_path
                if pca_path is not None
                else {
                    "X_pca": X_pca,
                    "X_res_pca": X_res_pca,
                    "y_original": y_mean.to_numpy(),
                    "y_resampled": y_train.to_numpy(),
                    "explained_variance_ratio": pca.explained_variance_ratio_,
                }
            )

    if "pca_gm" in methods and "gaussian_copula" not in methods:
        X_train, _, y_train, _ = gm_oversampling_train_test(
            X_mean,
            y_mean,
            test_size=0.3,
            synth_multiplier=1.0,
            random_state=random_state,
        )
        X_pca, X_res_pca, pca = pca_before_after_sampling(X_mean, y_mean, X_train, y_train)
        pca_path = (out / "complicaciones_pca_gaussian_copula.npz") if out is not None else None
        _save_npz(
            pca_path,
            X_pca=X_pca,
            X_res_pca=X_res_pca,
            y_original=y_mean.to_numpy(),
            y_resampled=y_train.to_numpy(),
            explained_variance_ratio=pca.explained_variance_ratio_,
        )
        exported["pca_gm"] = (
            pca_path
            if pca_path is not None
            else {
                "X_pca": X_pca,
                "X_res_pca": X_res_pca,
                "y_original": y_mean.to_numpy(),
                "y_resampled": y_train.to_numpy(),
                "explained_variance_ratio": pca.explained_variance_ratio_,
            }
        )

    if "pca_gmm" in methods and "gmm" not in methods:
        X_train, _, y_train, _ = gmm_oversampling_train_test(
            X_mean,
            y_mean,
            test_size=0.3,
            synth_multiplier=1.0,
            n_components=3,
            random_state=random_state,
        )
        X_pca, X_res_pca, pca = pca_before_after_sampling(X_mean, y_mean, X_train, y_train)
        pca_path = (out / "complicaciones_pca_gmm.npz") if out is not None else None
        _save_npz(
            pca_path,
            X_pca=X_pca,
            X_res_pca=X_res_pca,
            y_original=y_mean.to_numpy(),
            y_resampled=y_train.to_numpy(),
            explained_variance_ratio=pca.explained_variance_ratio_,
        )
        exported["pca_gmm"] = (
            pca_path
            if pca_path is not None
            else {
                "X_pca": X_pca,
                "X_res_pca": X_res_pca,
                "y_original": y_mean.to_numpy(),
                "y_resampled": y_train.to_numpy(),
                "explained_variance_ratio": pca.explained_variance_ratio_,
            }
        )

    if "fuzzy" in methods:
        fuzzy_df, fuzzy_meta = fuzzy_cmeans_severity(df, target=config.TARGET_COMPLICACIONES)
        data_path_out = (out / "complicaciones_fuzzy_scores.csv") if out is not None else None
        meta_path_out = (out / "complicaciones_fuzzy_meta.json") if out is not None else None
        write_df(fuzzy_df, data_path_out)
        if meta_path_out is not None:
            meta_path_out.write_text(json.dumps(fuzzy_meta, indent=2, default=str), encoding="utf-8")
        exported["fuzzy_scores"] = data_path_out if data_path_out is not None else fuzzy_df
        exported["fuzzy_meta"] = meta_path_out if meta_path_out is not None else fuzzy_meta

    if "pca_smote" in methods or "isomap_smote" in methods:
        X_res, y_res = apply_sampling(X_mean, y_mean, method="smote", random_state=random_state)

        if "pca_smote" in methods:
            X_pca, X_res_pca, pca = pca_before_after_sampling(X_mean, y_mean, X_res, y_res)
            path = (out / "complicaciones_pca_smote.npz") if out is not None else None
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

        if "isomap_smote" in methods:
            X_iso, X_res_iso, _ = isomap_before_after_sampling(X_mean, X_res)
            path = (out / "complicaciones_isomap_smote.npz") if out is not None else None
            _save_npz(path, X_iso=X_iso, X_res_iso=X_res_iso)
            exported["isomap_smote"] = (
                path if path is not None else {"X_iso": X_iso, "X_res_iso": X_res_iso}
            )

    if "mean_before_after_split" in methods:
        X_raw = df.drop(columns=[config.TARGET_COMPLICACIONES])

        X_train_a, X_test_a, y_train_a, y_test_a = train_test_split(
            X_mean,
            y_mean,
            test_size=0.3,
            random_state=random_state,
            stratify=y_mean,
        )
        models_a = _models_for_target(y_train_a, random_state=random_state)
        eval_before = run_train_test_models(X_train_a, y_train_a, X_test_a, y_test_a, models=models_a)

        X_train_b, X_test_b, y_train_b, y_test_b = train_test_split(
            X_raw,
            y_mean,
            test_size=0.3,
            random_state=random_state,
            stratify=y_mean,
        )
        means_b = X_train_b.mean(numeric_only=True)
        X_train_b = X_train_b.fillna(means_b)
        X_test_b = X_test_b.fillna(means_b)
        models_b = _models_for_target(y_train_b, random_state=random_state)
        eval_after = run_train_test_models(X_train_b, y_train_b, X_test_b, y_test_b, models=models_b)

        path_before = (out / "complicaciones_mean_before_split.csv") if out is not None else None
        path_after = (out / "complicaciones_mean_after_split.csv") if out is not None else None
        write_df(eval_before, path_before)
        write_df(eval_after, path_after)
        exported["mean_before"] = path_before if path_before is not None else eval_before
        exported["mean_after"] = path_after if path_after is not None else eval_after

    # Full-columns train/test snapshot.
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
        path = (out / "complicaciones_full_columns_train_test.csv") if out is not None else None
        write_df(full_eval, path)
        exported["full_columns_baseline"] = path if path is not None else full_eval

    return exported
