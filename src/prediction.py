"""Load the saved pipeline and produce churn predictions."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import (
    FULL_PIPELINE_PATH,
    RISK_LOW_MAX,
    RISK_MEDIUM_MAX,
)
from src.feature_engineering import FeatureEngineer


class PredictionError(Exception):
    """Raised when a prediction cannot be produced."""


def risk_from_probability(probability: float) -> str:
    if probability < RISK_LOW_MAX:
        return "LOW"
    if probability < RISK_MEDIUM_MAX:
        return "MEDIUM"
    return "HIGH"


def risk_explanation(level: str) -> str:
    messages = {
        "HIGH": (
            "Customer has a high predicted probability of churn. "
            "Immediate retention action is recommended."
        ),
        "MEDIUM": (
            "Customer shows moderate churn risk. "
            "Proactive engagement is recommended."
        ),
        "LOW": "Customer currently shows low predicted churn risk.",
    }
    return messages.get(level, messages["MEDIUM"])


def prediction_label(churn_flag: int) -> str:
    return "Likely to Churn" if int(churn_flag) == 1 else "Likely to Stay"


def load_full_pipeline(path: Path | None = None):
    target = path or FULL_PIPELINE_PATH
    if not target.is_file():
        raise PredictionError(
            "Model not trained. Run python3 scripts/train_model.py first."
        )
    try:
        pipeline = joblib.load(target)
    except Exception as exc:
        raise PredictionError(f"The saved model could not be loaded: {exc}") from exc
    return pipeline


def extract_engineered_row(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    engineer: FeatureEngineer = pipeline.named_steps["features"]
    engineered = engineer.transform(X)
    cols = [
        "AverageMonthlySpend",
        "TotalServices",
        "EstimatedLifetimeValue",
        "TenureGroup",
        "HighMonthlyCharge",
        "IsLongTermContract",
        "HasTechnicalSupport",
        "HasOnlineSecurity",
        "ServiceEngagementScore",
        "CustomerValueSegment",
    ]
    return engineered[cols]


def predict_dataframe(pipeline, X: pd.DataFrame) -> pd.DataFrame:
    if X.empty:
        raise PredictionError("No customer records were provided.")
    try:
        pred = pipeline.predict(X)
        if hasattr(pipeline, "predict_proba"):
            proba = pipeline.predict_proba(X)[:, 1]
        else:
            raise PredictionError("The trained model does not provide class probabilities.")
    except PredictionError:
        raise
    except Exception as exc:
        raise PredictionError(f"Prediction failed: {exc}") from exc

    result = X.copy()
    result["ChurnPrediction"] = pred.astype(int)
    result["ChurnProbability"] = np.asarray(proba, dtype=float)
    result["RiskLevel"] = [risk_from_probability(p) for p in result["ChurnProbability"]]
    result["PredictionLabel"] = [prediction_label(v) for v in result["ChurnPrediction"]]
    return result


def predict_single(pipeline, record: dict | pd.Series | pd.DataFrame) -> dict:
    if isinstance(record, pd.DataFrame):
        X = record.copy()
    else:
        X = pd.DataFrame([dict(record)])
    scored = predict_dataframe(pipeline, X)
    engineered = extract_engineered_row(pipeline, X).iloc[0].to_dict()
    row = scored.iloc[0]
    probability = float(row["ChurnProbability"])
    level = str(row["RiskLevel"])
    return {
        "prediction": int(row["ChurnPrediction"]),
        "prediction_label": str(row["PredictionLabel"]),
        "probability": probability,
        "probability_pct": round(probability * 100, 1),
        "risk": level,
        "explanation": risk_explanation(level),
        "engineered_features": engineered,
    }
