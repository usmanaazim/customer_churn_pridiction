#!/usr/bin/env python3
"""Predict churn for a sample customer from the command line."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.prediction import load_full_pipeline, predict_single
from src.recommendations import generate_recommendations


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict customer churn from a JSON record.")
    parser.add_argument(
        "--json",
        dest="json_path",
        help="Path to a JSON file with one customer object. Uses a sample profile if omitted.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    record = SAMPLE
    if args.json_path:
        record = json.loads(Path(args.json_path).read_text())

    pipeline = load_full_pipeline()
    result = predict_single(pipeline, record)
    recs = generate_recommendations(record, result["probability"], result["risk"])

    print("=" * 37)
    print("CUSTOMER CHURN PREDICTION")
    print("=" * 37)
    print(f"Prediction : {result['prediction_label']}")
    print(f"Probability: {result['probability_pct']:.1f}%")
    print(f"Risk       : {result['risk']}")
    print()
    print("Engineered features")
    for key, value in result["engineered_features"].items():
        print(f"  {key}: {value}")
    print()
    print("Rule-based retention recommendations")
    for item in recs:
        print(f"  - {item}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
