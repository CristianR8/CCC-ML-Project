"""Advanced methods (PSO and Fuzzy C-Means)"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler


try:
    from pyswarms.single.global_best import GlobalBestPSO
except ImportError:  # pragma: no cover
    GlobalBestPSO = None

try:
    import skfuzzy as fuzz
except ImportError:  # pragma: no cover
    fuzz = None


def pso_feature_selection(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.2,
    random_state: int = 42,
    threshold: float = 0.7,
    n_particles: int = 30,
    iters: int = 50,
) -> dict[str, object]:
    """PSO feature selection with GaussianNB fitness."""
    if GlobalBestPSO is None:
        raise ImportError("pyswarms is required for PSO feature selection")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    train_means = X_train.mean(numeric_only=True)
    X_train = X_train.fillna(train_means)
    X_test = X_test.fillna(train_means)

    scaler = StandardScaler()
    X_train_np = scaler.fit_transform(X_train.values)
    X_test_np = scaler.transform(X_test.values)

    def fitness(particles: np.ndarray) -> np.ndarray:
        n = particles.shape[0]
        scores = np.zeros(n)
        for idx, particle in enumerate(particles):
            mask = particle > threshold
            if mask.sum() == 0:
                scores[idx] = 1e6
                continue

            model = GaussianNB()
            model.fit(X_train_np[:, mask], y_train)
            y_pred = model.predict(X_test_np[:, mask])

            f1_major = f1_score(y_test, y_pred, pos_label=0)
            f1_minor = f1_score(y_test, y_pred, pos_label=1)

            penalty = (1 / X_train_np.shape[1]) * (mask.sum() ** 2)
            scores[idx] = -(0.5 * f1_major + 0.5 * f1_minor) + penalty

        return scores

    optimizer = GlobalBestPSO(
        n_particles=n_particles,
        dimensions=X_train_np.shape[1],
        options={"c1": 1.8, "c2": 2.0, "w": 0.7},
        bounds=(np.zeros(X_train_np.shape[1]), np.ones(X_train_np.shape[1])),
    )

    best_cost, best_pos = optimizer.optimize(fitness, iters=iters)
    best_mask = best_pos > threshold
    selected_features = X.columns[best_mask].tolist()

    return {
        "best_cost": float(best_cost),
        "selected_features": selected_features,
        "selected_mask": best_mask,
    }


def fuzzy_cmeans_severity(
    df: pd.DataFrame,
    target: str,
    n_clusters: int = 2,
    m: float = 2.0,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Fuzzy C-Means severity score based on membership to high-risk cluster."""
    if fuzz is None:
        raise ImportError("scikit-fuzzy is required for fuzzy C-Means methods")

    if target not in df.columns:
        raise KeyError(f"Missing target column '{target}'")

    df_full = df.dropna(axis=1).copy()
    y = df_full[target].astype(float)
    X = df_full.drop(columns=[target]).copy()

    bool_cols = X.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        X[bool_cols] = X[bool_cols].astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_fcm = X_scaled.T

    cntr, u, _, _, _, _, fpc = fuzz.cluster.cmeans(
        X_fcm,
        c=n_clusters,
        m=m,
        error=1e-5,
        maxiter=1000,
        init=None,
        seed=random_state,
    )

    memberships = u.T
    risk_by_cluster = []
    y_array = y.values
    for cluster_idx in range(n_clusters):
        avg_risk = np.average(y_array, weights=memberships[:, cluster_idx])
        risk_by_cluster.append(float(avg_risk))

    high_risk_idx = int(np.argmax(risk_by_cluster))
    severity_score = memberships[:, high_risk_idx]

    result_df = df_full.copy()
    result_df["severity_score"] = severity_score

    meta = {
        "centroids": cntr,
        "fpc": float(fpc),
        "risk_by_cluster": risk_by_cluster,
        "high_risk_cluster": high_risk_idx,
    }
    return result_df, meta
