"""Raw dataset readers."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_excel_dataset(path: str | Path) -> pd.DataFrame:
    """Read the source Excel file into a DataFrame."""
    return pd.read_excel(path)
