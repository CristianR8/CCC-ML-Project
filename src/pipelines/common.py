"""Shared pipeline helpers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def ensure_dir(path: str | Path | None) -> Path | None:
    if path is None:
        return None
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_df(df: pd.DataFrame, output_path: Path | None) -> None:
    if output_path is None:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
