"""Pipeline for dataset exploration."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from analysis.exploration import (
    lda_projection,
    pca_before_after_sampling,
    pca_biplot_top_features,
    pca_variance_curve,
)
from data.loaders import load_excel_dataset
from data.preprocessing import drop_existing_columns, remove_ghost_row
from modeling.sampling import apply_sampling
from pipelines.common import ensure_dir, write_df


DROP_COLUMNS_EXPLORACION = [
    "Primera dosis",
    "segunda dosis",
    "Tercera dosis",
    "Cuarta dosis",
    "Vacunación COVID",
    "Tipo vacuna",
    "Tipo vacuna.1",
    "Tipo vacuna.2",
    "Tipo vacuna.3",
    "Mortalidad tiempo",
    "Fuma actualmente",
    "número de cigarrillos diarios",
    "Años de fumador",
    "IMCat",
    "cod",
    "fechaingreso",
    "anioingreso",
    "Fecha aplicación",
    "Fecha aplicación.1",
    "Fecha aplicación.2",
    "Fecha aplicación.3",
    "Fecha mortalidad",
    "Fecha trasplante",
    "Fecha asistencia",
]

DEFAULT_METHODS = ["completeness", "pca", "pca_smote", "lda"]


def _prepare_exploration_dataframe(data_path: str | Path) -> pd.DataFrame:
    df = load_excel_dataset(data_path)
    df = remove_ghost_row(df)
    if "IPA" in df.columns:
        df["IPA"] = df["IPA"].fillna(0)
    df = drop_existing_columns(df, DROP_COLUMNS_EXPLORACION)
    return df


def run_exploracion_pipeline(
    data_path: str | Path,
    output_dir: str | Path | None = None,
    methods: list[str] | None = None,
) -> dict[str, object]:
    if not methods or "all" in methods:
        methods = DEFAULT_METHODS

    out = ensure_dir(output_dir)
    exported: dict[str, object] = {}

    df = _prepare_exploration_dataframe(data_path)

    if "completeness" in methods:
        counts = df.count().sort_values(ascending=False)
        completeness_df = pd.DataFrame(
            {
                "variable": counts.index,
                "counts": counts.values,
                "completeness_pct": (counts.values / len(df)) * 100,
            }
        )
        path = (out / "exploracion_completeness.csv") if out is not None else None
        write_df(completeness_df, path)
        exported["completeness"] = path if path is not None else completeness_df

    # PCA uses whichever target is available in this priority.
    pca_target = None
    for candidate in ["tipodepaciente", "Complicaciones cardiovasculares", "Mortalidad menor a 2 años"]:
        if candidate in df.columns:
            pca_target = candidate
            break

    if pca_target is not None and ("pca" in methods or "pca_smote" in methods):
        X = df.drop(columns=[pca_target]).copy()
        X = X.select_dtypes(include=[np.number]).fillna(X.mean(numeric_only=True))
        y = df[pca_target]

        if "pca" in methods:
            curve = pca_variance_curve(X)
            X_pca, loadings, top_idx = pca_biplot_top_features(X, y, top_k=5)
            path_curve = (out / "exploracion_pca_curve.json") if out is not None else None
            path_proj = (out / "exploracion_pca_projection.npz") if out is not None else None
            if path_curve is not None:
                path_curve.write_text(json.dumps(curve, indent=2, default=lambda x: x.tolist()), encoding="utf-8")
            if path_proj is not None:
                np.savez(path_proj, X_pca=X_pca, loadings=loadings, top_idx=top_idx)
            exported["pca_curve"] = path_curve if path_curve is not None else curve
            exported["pca_projection"] = (
                path_proj
                if path_proj is not None
                else {"X_pca": X_pca, "loadings": loadings, "top_idx": top_idx}
            )

        if "pca_smote" in methods:
            X_res, y_res = apply_sampling(X, y, method="smote", random_state=42)
            X_pca, X_res_pca, pca = pca_before_after_sampling(X, y, X_res, y_res)
            path = (out / "exploracion_pca_smote.npz") if out is not None else None
            if path is not None:
                np.savez(path, X_pca=X_pca, X_res_pca=X_res_pca, explained_variance_ratio=pca.explained_variance_ratio_)
            exported["pca_smote"] = (
                path
                if path is not None
                else {
                    "X_pca": X_pca,
                    "X_res_pca": X_res_pca,
                    "explained_variance_ratio": pca.explained_variance_ratio_,
                }
            )

    if "lda" in methods and "tipodepaciente" in df.columns:
        hematology_cols = [
            "Globulos Rojos 1 x 10^6/u",
            "Leucocitos 1 ",
            "Heoglobina 1 ",
            "hematocrito",
            "Volumen Corpuscular Medio 1",
            "Hemoglobina Corpuscular Media 1",
            "Concentración de Hemoglobina Crospucular Media 1",
            "Recuento de plaquetas 1",
            "RDW 1",
            "Volumen Plaquetario 1",
            "Neutrofilos 1",
            "Linfocitos 1",
            "Monocitos 1",
            "Eosinofilos 1",
            "Basofilos 1",
            "Neutrofilos porcentaje",
            "Linfocitos porcentaje",
            "Monocitos porcentaje",
            "Eosinofilos porcentaje",
            "Basofilos porcentaje",
        ]
        existing = [col for col in hematology_cols if col in df.columns]
        if existing:
            X_lda = df[existing].fillna(df[existing].mean(numeric_only=True))
            y_lda = df["tipodepaciente"]
            projection = lda_projection(X_lda, y_lda)
            path = (out / "exploracion_lda_projection.npz") if out is not None else None
            if path is not None:
                np.savez(path, X_lda=projection, y=y_lda.to_numpy())
            exported["lda"] = path if path is not None else {"X_lda": projection, "y": y_lda.to_numpy()}

    return exported
