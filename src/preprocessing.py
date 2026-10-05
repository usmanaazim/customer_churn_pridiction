"""Data cleaning that does not learn parameters from the full dataset."""

from __future__ import annotations

import pandas as pd

from src.config import ID_COLUMN, TARGET_COLUMN, TARGET_NEGATIVE, TARGET_POSITIVE


def strip_object_columns(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    object_cols = cleaned.select_dtypes(include=["object"]).columns
    for col in object_cols:
        cleaned[col] = cleaned[col].astype(str).str.strip()
        cleaned[col] = cleaned[col].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return cleaned


def convert_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    cleaned["TotalCharges"] = pd.to_numeric(cleaned["TotalCharges"], errors="coerce")
    return cleaned


def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    cleaned = df.copy()
    if "TotalCharges" in cleaned.columns and "MonthlyCharges" in cleaned.columns:
        missing_total = cleaned["TotalCharges"].isna()
        if "tenure" in cleaned.columns:
            zero_tenure = cleaned["tenure"].fillna(0).astype(float) == 0
            cleaned.loc[missing_total & zero_tenure, "TotalCharges"] = 0.0
            remaining = cleaned["TotalCharges"].isna()
            cleaned.loc[remaining, "TotalCharges"] = cleaned.loc[remaining, "MonthlyCharges"]
        else:
            cleaned["TotalCharges"] = cleaned["TotalCharges"].fillna(cleaned["MonthlyCharges"])

    numeric_cols = cleaned.select_dtypes(include=["number"]).columns
    for col in numeric_cols:
        if cleaned[col].isna().any():
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    object_cols = cleaned.select_dtypes(include=["object"]).columns
    for col in object_cols:
        if col == ID_COLUMN:
            continue
        if cleaned[col].isna().any():
            mode = cleaned[col].mode(dropna=True)
            fill_value = mode.iloc[0] if not mode.empty else "Unknown"
            cleaned[col] = cleaned[col].fillna(fill_value)
    return cleaned


def encode_target(series: pd.Series) -> pd.Series:
    mapping = {TARGET_POSITIVE: 1, TARGET_NEGATIVE: 0, "1": 1, "0": 0, 1: 1, 0: 0}
    encoded = series.map(mapping)
    if encoded.isna().any():
        raise ValueError("Target column contains values other than Yes/No.")
    return encoded.astype(int)


def clean_dataset(df: pd.DataFrame, drop_duplicates: bool = True) -> pd.DataFrame:
    cleaned = strip_object_columns(df)
    cleaned = convert_total_charges(cleaned)
    cleaned = handle_missing_values(cleaned)
    cleaned["SeniorCitizen"] = pd.to_numeric(cleaned["SeniorCitizen"], errors="coerce").fillna(0).astype(int)
    cleaned["tenure"] = pd.to_numeric(cleaned["tenure"], errors="coerce").fillna(0)
    cleaned["MonthlyCharges"] = pd.to_numeric(cleaned["MonthlyCharges"], errors="coerce")

    if drop_duplicates:
        subset = [col for col in cleaned.columns if col != ID_COLUMN]
        cleaned = cleaned.drop_duplicates(subset=subset, keep="first")

    cleaned = cleaned.reset_index(drop=True)
    return cleaned


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series | None]:
    work = df.copy()
    customer_ids = work[ID_COLUMN] if ID_COLUMN in work.columns else None
    y = encode_target(work[TARGET_COLUMN]) if TARGET_COLUMN in work.columns else None
    drop_cols = [col for col in [ID_COLUMN, TARGET_COLUMN] if col in work.columns]
    X = work.drop(columns=drop_cols)
    return X, y, customer_ids
