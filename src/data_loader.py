"""Load and validate the Telco Customer Churn dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import DATA_PATH, REQUIRED_COLUMNS


class DatasetError(Exception):
    """Raised when the customer dataset cannot be used."""


def dataset_exists(path: Path | None = None) -> bool:
    target = path or DATA_PATH
    return target.is_file() and target.stat().st_size > 0


def validate_columns(df: pd.DataFrame) -> list[str]:
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    return missing


def load_raw_dataset(path: Path | None = None) -> pd.DataFrame:
    target = path or DATA_PATH
    if not dataset_exists(target):
        raise DatasetError(
            "Dataset not found. Place the Telco Customer Churn CSV at "
            "data/customer_churn.csv or run: python3 scripts/setup_data.py"
        )
    try:
        df = pd.read_csv(target)
    except Exception as exc:
        raise DatasetError(f"Could not read the dataset file: {exc}") from exc

    df.columns = df.columns.str.strip()
    missing = validate_columns(df)
    if missing:
        raise DatasetError(
            "Dataset is missing required columns: " + ", ".join(missing)
        )
    return df
