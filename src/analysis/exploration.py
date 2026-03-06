"""EDA and projection methods."""

from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.manifold import Isomap
from sklearn.preprocessing import StandardScaler


def plot_all_correlation_subplots(
    df: pd.DataFrame,
    vars_per_plot: int = 5,
    correlation: str = "pearson",
) -> list[plt.Figure]:
    """Create batches of correlation heatmaps grouping strongly related variables."""
    corr_matrix = df.corr(method=correlation)
    corr_unstacked = corr_matrix.unstack()
    corr_unstacked = corr_unstacked[
        corr_unstacked.index.get_level_values(0) != corr_unstacked.index.get_level_values(1)
    ]
    strong = corr_unstacked.abs().sort_values(ascending=False)

    used = set()
    groups: list[list[str]] = []

    while len(used) < len(df.columns):
        available_pairs = [
            (pair, corr)
            for pair, corr in strong.items()
            if pair[0] not in used and pair[1] not in used
        ]

        if not available_pairs:
            remaining = [var for var in df.columns if var not in used]
            for idx in range(0, len(remaining), vars_per_plot):
                group = remaining[idx : idx + vars_per_plot]
                if len(group) >= 2:
                    groups.append(group)
                    used.update(group)
            break

        best_pair, _ = available_pairs[0]
        group = [best_pair[0], best_pair[1]]

        for pair, _ in strong.items():
            if len(group) >= vars_per_plot:
                break
            if pair[0] in group and pair[1] not in group and pair[1] not in used:
                group.append(pair[1])
            elif pair[1] in group and pair[0] not in group and pair[0] not in used:
                group.append(pair[0])

        groups.append(group)
        used.update(group)

    figures: list[plt.Figure] = []
    for start in range(0, len(groups), 4):
        batch = groups[start : start + 4]
        rows = (len(batch) + 1) // 2
        cols = 2 if len(batch) > 1 else 1
        fig, axes = plt.subplots(rows, cols, figsize=(20, 7 * rows))

        if len(batch) == 1:
            axes = [axes]
        else:
            axes = np.array(axes).ravel().tolist()

        for idx, vars_group in enumerate(batch):
            heat = df[vars_group].corr(method=correlation)
            sns.heatmap(
                heat,
                annot=True,
                cmap="coolwarm",
                center=0,
                fmt=".2f",
                square=True,
                cbar_kws={"shrink": 0.8},
                ax=axes[idx],
            )
            axes[idx].set_title(f"Grupo {start + idx + 1}")

        for idx in range(len(batch), len(axes)):
            axes[idx].set_visible(False)

        fig.tight_layout()
        figures.append(fig)

    return figures


def plot_subset_correlation(
    df: pd.DataFrame,
    columns: list[str],
    figsize: tuple[int, int] = (8, 6),
    correlation: str = "pearson",
) -> plt.Figure:
    """Heatmap for a selected set of columns."""
    subset = df[columns]
    corr = subset.corr(method=correlation)
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(corr, annot=True, cmap="coolwarm", center=0, fmt=".2f", ax=ax)
    ax.set_title("Matriz de correlacion (subset)")
    return fig


def pca_variance_curve(X: pd.DataFrame, threshold: float = 0.9) -> dict[str, object]:
    """Compute explained-variance curve and minimum component count at threshold."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA()
    pca.fit(X_scaled)
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    n_components = int(np.argmax(cumulative >= threshold) + 1)

    return {
        "explained_variance_ratio": explained,
        "cumulative_variance": cumulative,
        "n_components_threshold": n_components,
        "variance_at_threshold": float(cumulative[n_components - 1]),
    }


def pca_biplot_top_features(
    X: pd.DataFrame,
    y: pd.Series,
    top_k: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return PCA projection and top loading vectors for biplot rendering."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    loadings = pca.components_.T * np.sqrt(pca.explained_variance_)

    magnitudes = np.sqrt(loadings[:, 0] ** 2 + loadings[:, 1] ** 2)
    top_idx = np.argsort(magnitudes)[-top_k:]
    return X_pca, loadings, top_idx


def lda_projection(X: pd.DataFrame, y: pd.Series) -> np.ndarray:
    """1D LDA projection for binary targets."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    lda = LinearDiscriminantAnalysis(n_components=1)
    return lda.fit_transform(X_scaled, y)


def pca_before_after_sampling(
    X_original: pd.DataFrame,
    y_original: pd.Series,
    X_resampled: pd.DataFrame,
    y_resampled: pd.Series,
) -> tuple[np.ndarray, np.ndarray, PCA]:
    """Project original and resampled datasets using the same PCA basis."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_original)
    X_res_scaled = scaler.transform(X_resampled)

    pca = PCA(n_components=2)
    pca.fit(X_scaled)
    return pca.transform(X_scaled), pca.transform(X_res_scaled), pca


def isomap_before_after_sampling(
    X_original: pd.DataFrame,
    X_resampled: pd.DataFrame,
    n_components: int = 2,
) -> tuple[np.ndarray, np.ndarray, Isomap]:
    """Isomap projection before and after resampling with shared manifold map."""
    iso = Isomap(n_components=n_components)
    X_iso = iso.fit_transform(X_original)
    X_res_iso = iso.transform(X_resampled)
    return X_iso, X_res_iso, iso
