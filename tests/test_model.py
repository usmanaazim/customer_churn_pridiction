import pandas as pd
import pytest

from src.config import FULL_PIPELINE_PATH
from src.prediction import (
    PredictionError,
    load_full_pipeline,
    prediction_label,
    predict_single,
    risk_from_probability,
)


def test_risk_thresholds():
    assert risk_from_probability(0.10) == "LOW"
    assert risk_from_probability(0.29) == "LOW"
    assert risk_from_probability(0.30) == "MEDIUM"
    assert risk_from_probability(0.59) == "MEDIUM"
    assert risk_from_probability(0.60) == "HIGH"


def test_prediction_label():
    assert prediction_label(1) == "Likely to Churn"
    assert prediction_label(0) == "Likely to Stay"


SAMPLE = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "No",
    "Dependents": "No",
    "tenure": 2,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "No",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 85.5,
    "TotalCharges": 171.0,
}


@pytest.mark.skipif(not FULL_PIPELINE_PATH.is_file(), reason="Model artifacts not trained yet")
def test_predict_single_returns_probability_and_risk():
    pipeline = load_full_pipeline()
    result = predict_single(pipeline, SAMPLE)
    assert result["prediction"] in (0, 1)
    assert 0.0 <= result["probability"] <= 1.0
    assert result["risk"] in {"LOW", "MEDIUM", "HIGH"}
    assert "AverageMonthlySpend" in result["engineered_features"]


def test_missing_pipeline_message(tmp_path, monkeypatch):
    from src import prediction as prediction_mod

    monkeypatch.setattr(prediction_mod, "FULL_PIPELINE_PATH", tmp_path / "missing.joblib")
    with pytest.raises(PredictionError, match="Model not trained"):
        load_full_pipeline(tmp_path / "missing.joblib")
