"""Synthetic-data methods (GM, GMM, DP-GNB, DataSynthesizer)."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

try:
    from sdv.metadata import SingleTableMetadata
    from sdv.single_table import GaussianCopulaSynthesizer
except ImportError:  # pragma: no cover
    SingleTableMetadata = None
    GaussianCopulaSynthesizer = None


def gm_oversampling_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.3,
    synth_multiplier: float = 1.0,
    random_state: int = 42,
    minority_label: int = 1,
    majority_label: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """GaussianCopula oversampling on training split only."""
    if GaussianCopulaSynthesizer is None or SingleTableMetadata is None:
        raise ImportError("sdv is required for GaussianCopulaSynthesizer methods")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    train_means = X_train.mean(numeric_only=True)
    X_train = X_train.fillna(train_means)
    X_test = X_test.fillna(train_means)

    df_train = X_train.copy()
    df_train["target"] = y_train

    df_min = df_train[df_train["target"] == minority_label].reset_index(drop=True)
    df_maj = df_train[df_train["target"] == majority_label].reset_index(drop=True)

    if len(df_min) < 10:
        return X_train, X_test, y_train, y_test

    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(df_min)

    gmodel = GaussianCopulaSynthesizer(metadata)
    gmodel.fit(df_min)

    target_size = int(len(df_maj) * synth_multiplier)
    n_to_generate = max(target_size - len(df_min), 0)

    if n_to_generate > 0:
        synthetic_min = gmodel.sample(n_to_generate)
        synthetic_min = synthetic_min[df_min.columns]
        synthetic_min["target"] = minority_label
    else:
        synthetic_min = pd.DataFrame(columns=df_train.columns)

    df_aug = pd.concat([df_maj, df_min, synthetic_min], ignore_index=True)
    df_aug = df_aug.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    X_train_bal = df_aug.drop(columns=["target"])
    y_train_bal = df_aug["target"]
    return X_train_bal, X_test, y_train_bal, y_test


def gmm_oversampling_train_test(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.3,
    synth_multiplier: float = 1.0,
    n_components: int = 3,
    random_state: int = 42,
    minority_label: int = 1,
    majority_label: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """GMM oversampling on training split only."""
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state,
    )

    bool_cols = X_train.select_dtypes(include=["bool"]).columns
    if len(bool_cols) > 0:
        X_train = X_train.copy()
        X_test = X_test.copy()
        X_train[bool_cols] = X_train[bool_cols].astype(int)
        X_test[bool_cols] = X_test[bool_cols].astype(int)

    train_means = X_train.mean(numeric_only=True)
    X_train = X_train.fillna(train_means)
    X_test = X_test.fillna(train_means)

    df_train = X_train.copy()
    df_train["target"] = y_train

    df_min = df_train[df_train["target"] == minority_label].reset_index(drop=True)
    df_maj = df_train[df_train["target"] == majority_label].reset_index(drop=True)

    if len(df_min) < n_components:
        return X_train, X_test, y_train, y_test

    gmm = GaussianMixture(n_components=n_components, covariance_type="full", random_state=random_state)
    gmm.fit(df_min.drop(columns=["target"]))

    target_size = int(len(df_maj) * synth_multiplier)
    n_to_generate = max(target_size - len(df_min), 0)

    if n_to_generate > 0:
        synthetic_features, _ = gmm.sample(n_to_generate)
        synthetic_min = pd.DataFrame(synthetic_features, columns=df_min.drop(columns=["target"]).columns)
        synthetic_min["target"] = minority_label
    else:
        synthetic_min = pd.DataFrame(columns=df_train.columns)

    df_aug = pd.concat([df_maj, df_min, synthetic_min], ignore_index=True)
    df_aug = df_aug.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    X_train_bal = df_aug.drop(columns=["target"])
    y_train_bal = df_aug["target"]
    return X_train_bal, X_test, y_train_bal, y_test


def gmm_generate_minority_samples(
    X: pd.DataFrame,
    y: pd.Series,
    n_components: int = 3,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic minority points (scaled) for visualization."""
    class_counts = Counter(y)
    minority_class = min(class_counts, key=class_counts.get)
    major_class = max(class_counts, key=class_counts.get)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_minority = X_scaled[y == minority_class]
    if len(X_minority) < n_components:
        n_components = len(X_minority)

    gmm = GaussianMixture(n_components=max(1, n_components), random_state=random_state)
    gmm.fit(X_minority)

    n_to_generate = max(class_counts[major_class] - class_counts[minority_class], 1)
    X_syn_minority = gmm.sample(n_to_generate)[0]
    return X_scaled, X_syn_minority


def private_gaussian_naive_bayes_params(
    X: np.ndarray,
    Y: np.ndarray,
    eps: float = 1.0,
    data_range: float = 1.0,
) -> tuple[dict[int, dict[int, dict[str, float]]], np.ndarray]:
    """Estimate DP-GNB parameters with Laplace mechanism."""
    d = X.shape[1]
    classes = np.unique(Y)

    eps_split = eps / 2.0
    sensitivity_sum = data_range
    sensitivity_sum_sq = data_range**2

    private_params: dict[int, dict[int, dict[str, float]]] = defaultdict(dict)

    for c in classes:
        X_c = X[Y == c]
        n_c = X_c.shape[0]
        if n_c == 0:
            continue

        for i in range(d):
            feature_i = X_c[:, i]
            sum_i = np.sum(feature_i)
            sum_sq_i = np.sum(feature_i**2)

            sum_i_noisy = sum_i + np.random.laplace(0, sensitivity_sum / eps_split)
            sum_sq_i_noisy = sum_sq_i + np.random.laplace(0, sensitivity_sum_sq / eps_split)

            mu_noisy = sum_i_noisy / n_c
            ex2_noisy = sum_sq_i_noisy / n_c
            var_noisy = max(1e-6, ex2_noisy - (mu_noisy**2))
            private_params[int(c)][i] = {"mu": float(mu_noisy), "var": float(var_noisy)}

    return dict(private_params), classes


def generate_minority_synthetic_data(
    private_params: dict[int, dict[int, dict[str, float]]],
    Y: np.ndarray,
    minority_class: int,
    multiplier: int = 3,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate DP-GNB minority synthetic samples."""
    n_minority_original = int(np.sum(Y == minority_class))
    n_minority_syn = n_minority_original * multiplier

    class_params = private_params[minority_class]
    d = len(class_params)
    X_minority_syn = np.zeros((n_minority_syn, d))

    for i in range(d):
        params = class_params[i]
        mu_noisy = params["mu"]
        std_noisy = float(np.sqrt(params["var"]))
        X_minority_syn[:, i] = np.random.normal(loc=mu_noisy, scale=std_noisy, size=n_minority_syn)

    y_minority_syn = np.repeat(minority_class, n_minority_syn)
    return X_minority_syn, y_minority_syn


def generate_with_datasynthesizer(
    input_csv: str | Path,
    output_csv: str | Path,
    mode: str = "independent_attribute_mode",
) -> None:
    """Optional DataSynthesizer wrapper for completeness of methods inventory."""
    try:
        from DataSynthesizer.DataDescriber import DataDescriber
        from DataSynthesizer.DataGenerator import DataGenerator
    except ImportError as exc:  # pragma: no cover
        raise ImportError("DataSynthesizer is not installed") from exc

    input_csv = str(input_csv)
    output_csv = str(output_csv)
    description_path = output_csv + ".description.json"

    describer = DataDescriber(category_threshold=20)
    describer.describe_dataset_in_independent_attribute_mode(dataset_file=input_csv)
    describer.save_dataset_description_to_file(description_path)

    generator = DataGenerator()
    generator.generate_dataset_in_independent_mode(
        n=1000,
        description_file=description_path,
        seed=42,
    )
    generator.save_synthetic_data(output_csv)
