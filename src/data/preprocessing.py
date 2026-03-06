"""Dataset preprocessing helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

import config
from data.loaders import load_excel_dataset


def _as_dataframe(data: pd.DataFrame | str | Path) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return load_excel_dataset(data)


def drop_existing_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    existing = [col for col in columns if col in df.columns]
    if existing:
        return df.drop(columns=existing)
    return df


def replace_echo_zeros_with_nan(df: pd.DataFrame) -> pd.DataFrame:
    for col in config.ECHO_ZERO_TO_NAN_COLUMNS:
        if col in df.columns:
            df.loc[df[col] == 0, col] = pd.NA
    return df


def remove_ghost_row(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) == 0:
        return df
    return df.iloc[:-1].copy()


def fill_zero_semantics(df: pd.DataFrame) -> pd.DataFrame:
    for col in config.ZERO_IMPUTE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    return df


def apply_one_hot_encoding(df: pd.DataFrame) -> pd.DataFrame:
    existing = [col for col in config.OHE_COLUMNS if col in df.columns]
    if not existing:
        return df
    return pd.get_dummies(df, columns=existing)


def build_full_columns_dataset(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Dataset variant with only fully complete columns."""
    full_df = df.dropna(axis=1)
    if target not in full_df.columns:
        raise KeyError(f"Target '{target}' not available after dropna(axis=1)")
    y = full_df[target].copy()
    X = full_df.drop(columns=[target]).copy()
    return X, y


def build_mean_imputed_dataset(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, pd.Series]:
    """Dataset variant with numeric mean imputation."""
    if target not in df.columns:
        raise KeyError(f"Target '{target}' was not found in the dataframe")
    y = df[target].copy()
    X = df.drop(columns=[target]).copy()
    X = X.fillna(X.mean(numeric_only=True))
    return X, y


def train_val_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.3,
    val_size_from_temp: float = 0.5,
    random_state: int = config.DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Stratified train / val / test split."""
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=val_size_from_temp,
        random_state=random_state,
        stratify=y_temp,
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def prepare_complicaciones_dataframe(data: pd.DataFrame | str | Path) -> pd.DataFrame:
    """Preprocessing path for Complicaciones and Entrenamiento."""
    df = _as_dataframe(data)
    df = replace_echo_zeros_with_nan(df)
    df = remove_ghost_row(df)
    df = fill_zero_semantics(df)
    df = apply_one_hot_encoding(df)
    df = drop_existing_columns(df, config.DROP_COMMON_COLUMNS)
    df = drop_existing_columns(df, config.DROP_FOR_COMPLICACIONES)

    if config.TARGET_COMPLICACIONES not in df.columns:
        raise KeyError(
            f"'{config.TARGET_COMPLICACIONES}' is required for this pipeline and was not found"
        )
    return df


def prepare_mortalidad_dataframe(data: pd.DataFrame | str | Path) -> pd.DataFrame:
    """Preprocessing path for Mortalidad_menor_a_2_anos."""
    df = _as_dataframe(data)
    df = replace_echo_zeros_with_nan(df)
    df = remove_ghost_row(df)
    df = fill_zero_semantics(df)
    df = apply_one_hot_encoding(df)
    df = drop_existing_columns(df, config.DROP_COMMON_COLUMNS)
    df = drop_existing_columns(df, config.DROP_FOR_MORTALIDAD)

    if config.TARGET_MORTALIDAD not in df.columns:
        raise KeyError(f"'{config.TARGET_MORTALIDAD}' is required for this pipeline")
    return df
