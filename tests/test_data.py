from pathlib import Path

import pandas as pd
import pytest

from src.config import DATA_PATH, REQUIRED_COLUMNS
from src.data_loader import DatasetError, load_raw_dataset, validate_columns
from src.preprocessing import clean_dataset, convert_total_charges, encode_target, handle_missing_values


def _mini_csv(path: Path) -> None:
    rows = {
        "customerID": ["0001", "0002"],
        "gender": ["Female", "Male"],
        "SeniorCitizen": [0, 1],
        "Partner": ["Yes", "No"],
        "Dependents": ["No", "No"],
        "tenure": [1, 24],
        "PhoneService": ["Yes", "Yes"],
        "MultipleLines": ["No", "Yes"],
        "InternetService": ["DSL", "Fiber optic"],
        "OnlineSecurity": ["No", "Yes"],
        "OnlineBackup": ["Yes", "No"],
        "DeviceProtection": ["No", "No"],
        "TechSupport": ["No", "Yes"],
        "StreamingTV": ["No", "Yes"],
        "StreamingMovies": ["No", "No"],
        "Contract": ["Month-to-month", "One year"],
        "PaperlessBilling": ["Yes", "No"],
        "PaymentMethod": ["Electronic check", "Mailed check"],
        "MonthlyCharges": [50.0, 80.0],
        "TotalCharges": ["50", "1920"],
        "Churn": ["Yes", "No"],
    }
    pd.DataFrame(rows).to_csv(path, index=False)


def test_validate_columns_detects_missing():
    df = pd.DataFrame({"customerID": ["a"], "Churn": ["Yes"]})
    missing = validate_columns(df)
    assert "tenure" in missing
    assert "MonthlyCharges" in missing


def test_load_raw_dataset_from_temp(tmp_path):
    path = tmp_path / "customer_churn.csv"
    _mini_csv(path)
    df = load_raw_dataset(path)
    assert list(df.columns)[:3] == ["customerID", "gender", "SeniorCitizen"]
    assert set(REQUIRED_COLUMNS).issubset(df.columns)


def test_load_raw_dataset_missing_file(tmp_path):
    with pytest.raises(DatasetError):
        load_raw_dataset(tmp_path / "missing.csv")


def test_total_charges_blank_becomes_numeric():
    df = pd.DataFrame({"TotalCharges": [" ", "10.5"], "MonthlyCharges": [20.0, 10.5], "tenure": [0, 1]})
    converted = convert_total_charges(df)
    filled = handle_missing_values(converted)
    assert filled.loc[0, "TotalCharges"] == 0.0
    assert filled.loc[1, "TotalCharges"] == 10.5


def test_encode_target():
    encoded = encode_target(pd.Series(["Yes", "No", "Yes"]))
    assert encoded.tolist() == [1, 0, 1]


def test_clean_dataset_mini(tmp_path):
    path = tmp_path / "customer_churn.csv"
    _mini_csv(path)
    cleaned = clean_dataset(load_raw_dataset(path))
    assert cleaned["TotalCharges"].dtype.kind in "fc"
    assert cleaned["tenure"].isna().sum() == 0


@pytest.mark.skipif(not DATA_PATH.is_file(), reason="Full Telco dataset not present")
def test_full_dataset_required_columns():
    df = load_raw_dataset()
    assert validate_columns(df) == []
    assert len(df) > 100
