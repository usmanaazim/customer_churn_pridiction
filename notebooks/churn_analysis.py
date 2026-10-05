"""Exploratory analysis script for the Telco churn dataset.

Run from the project root:

    python3 notebooks/churn_analysis.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt
import seaborn as sns

from src.analytics import churn_rate_by, key_insights, kpi_summary, prepare_analytics_frame
from src.data_loader import load_raw_dataset
from src.preprocessing import clean_dataset

sns.set_theme(style="whitegrid")


def main() -> None:
    df = prepare_analytics_frame(clean_dataset(load_raw_dataset()))
    print("Shape:", df.shape)
    print("KPIs:", kpi_summary(df))
    print("\nChurn distribution")
    print(df["Churn"].value_counts(normalize=True))

    analyses = [
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "Contract",
        "InternetService",
        "PaymentMethod",
        "TechSupport",
        "OnlineSecurity",
        "TenureGroup",
    ]
    for col in analyses:
        print(f"\nChurn rate by {col}")
        print(churn_rate_by(df, col).to_string(index=False))

    print("\nTenure vs churn (means)")
    print(df.groupby("Churn")[["tenure", "MonthlyCharges", "TotalCharges"]].mean())

    print("\nInsights")
    for line in key_insights(df):
        print("-", line)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    sns.countplot(data=df, x="Churn", ax=axes[0], color="#3E7CB1")
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", ax=axes[1])
    fig.tight_layout()
    out = ROOT / "assets" / "eda_preview.png"
    fig.savefig(out, dpi=120)
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
