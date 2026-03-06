"""Cox regression helpers, including LASSO-Cox utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize column names by trimming spaces and line breaks."""
    normalized = df.copy()
    normalized.columns = (
        normalized.columns.astype(str)
        .str.strip()
        .str.replace("\n", " ", regex=False)
        .str.replace("  ", " ", regex=False)
    )
    return normalized


def coerce_binary(series: pd.Series) -> pd.Series:
    """Coerce common text/bool encodings into numeric 0/1."""
    out = series.copy()
    if out.dtype == "O":
        lowered = out.astype(str).str.strip().str.lower()
        lowered = lowered.replace(
            {
                "si": 1,
                "sí": 1,
                "s": 1,
                "true": 1,
                "1": 1,
                "no": 0,
                "n": 0,
                "false": 0,
                "0": 0,
            }
        )
        return pd.to_numeric(lowered, errors="coerce")
    return pd.to_numeric(out, errors="coerce")


def prepare_survival_dataframe(
    df: pd.DataFrame,
    time_col: str,
    event_col: str,
) -> pd.DataFrame:
    """Clean survival columns and keep valid time/event rows only."""
    out = df.copy()
    out[time_col] = pd.to_numeric(out[time_col], errors="coerce")
    out[event_col] = coerce_binary(out[event_col])

    out = out.loc[out[time_col].notna() & (out[time_col] > 0)]
    out = out.loc[out[event_col].isin([0, 1])]
    return out


def make_design_matrix(df: pd.DataFrame, predictors: list[str]) -> pd.DataFrame:
    """Build Cox design matrix using numeric coercion + one-hot for categoricals."""
    X = df[predictors].copy()

    categorical_cols = [col for col in X.columns if X[col].dtype == "O"]
    numeric_cols = [col for col in X.columns if col not in categorical_cols]

    for col in numeric_cols:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    if categorical_cols:
        X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)

    return X


def build_cox_dataframe(
    df: pd.DataFrame,
    predictors: list[str],
    time_col: str,
    event_col: str,
    dropna: bool = True,
) -> pd.DataFrame:
    """Build modeling dataframe with survival columns + encoded predictors."""
    X = make_design_matrix(df, predictors)
    cox_df = pd.concat(
        [
            df[[time_col, event_col]].reset_index(drop=True),
            X.reset_index(drop=True),
        ],
        axis=1,
    )
    if dropna:
        cox_df = cox_df.dropna()
    return cox_df


def fit_cox_model(
    cox_df: pd.DataFrame,
    time_col: str,
    event_col: str,
    penalizer: float = 0.0,
    l1_ratio: float = 0.0,
) -> CoxPHFitter:
    """Fit Cox model with configurable elastic-net penalty."""
    cph = CoxPHFitter(penalizer=penalizer, l1_ratio=l1_ratio)
    cph.fit(cox_df, duration_col=time_col, event_col=event_col)
    return cph


def fit_lasso_cox(
    cox_df: pd.DataFrame,
    time_col: str,
    event_col: str,
    penalizer: float = 0.01,
) -> CoxPHFitter:
    """Fit LASSO-Cox (L1-only) model."""
    return fit_cox_model(
        cox_df=cox_df,
        time_col=time_col,
        event_col=event_col,
        penalizer=penalizer,
        l1_ratio=1.0,
    )


def hr_table(cph: CoxPHFitter) -> pd.DataFrame:
    """Hazard ratio summary table sorted by p-value."""
    summary = cph.summary.copy()
    summary["HR"] = np.exp(summary["coef"])
    summary["HR_lower_95"] = np.exp(summary["coef lower 95%"])
    summary["HR_upper_95"] = np.exp(summary["coef upper 95%"])
    return summary[["HR", "HR_lower_95", "HR_upper_95", "p"]].sort_values("p")


def split_hr_by_risk(hr_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split HR table into risk-increasing and risk-decreasing predictors."""
    higher_risk = hr_df.loc[hr_df["HR"] > 1].sort_values("HR", ascending=False)
    lower_risk = hr_df.loc[hr_df["HR"] < 1].sort_values("HR", ascending=True)
    return higher_risk, lower_risk


def non_zero_lasso_coefficients(
    cph: CoxPHFitter,
    atol: float = 1e-8,
) -> pd.DataFrame:
    """Return non-zero coefficients from a penalized Cox model."""
    summary = cph.summary.copy()
    non_zero = summary.loc[~np.isclose(summary["coef"], 0.0, atol=atol)].copy()
    return non_zero.sort_values("coef", ascending=False)
