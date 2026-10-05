"""Dataset KPIs, dynamic insights, and scoring helpers."""

from __future__ import annotations

import pandas as pd

from src.feature_engineering import FeatureEngineer, tenure_group_from_values
from src.prediction import predict_dataframe


def prepare_analytics_frame(df: pd.DataFrame, engineer: FeatureEngineer | None = None) -> pd.DataFrame:
    work = df.copy()
    if "Churn" in work.columns:
        work["ChurnFlag"] = work["Churn"].astype(str).str.strip().str.lower().eq("yes").astype(int)
    if engineer is not None:
        engineered = engineer.transform(work.drop(columns=["Churn"], errors="ignore"))
        for col in [
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
        ]:
            work[col] = engineered[col].values
    else:
        work["TenureGroup"] = tenure_group_from_values(work["tenure"])
        work["EstimatedLifetimeValue"] = work["MonthlyCharges"] * work["tenure"]
        from src.feature_engineering import count_total_services, service_engagement_score

        work["TotalServices"] = count_total_services(work)
        work["ServiceEngagementScore"] = service_engagement_score(work, work["TotalServices"])
        work["AverageMonthlySpend"] = work["TotalCharges"] / work["tenure"].replace(0, pd.NA)
        work["AverageMonthlySpend"] = work["AverageMonthlySpend"].fillna(work["MonthlyCharges"])
    return work


def kpi_summary(df: pd.DataFrame) -> dict:
    n = len(df)
    churned = int(df["ChurnFlag"].sum()) if n else 0
    retained = n - churned
    high_risk = int((df["RiskLevel"] == "HIGH").sum()) if "RiskLevel" in df.columns else None
    return {
        "total_customers": n,
        "churned_customers": churned,
        "retained_customers": retained,
        "churn_rate": (churned / n) if n else 0.0,
        "high_risk_customers": high_risk,
        "avg_monthly_charges": float(df["MonthlyCharges"].mean()) if n else 0.0,
        "avg_tenure": float(df["tenure"].mean()) if n else 0.0,
        "avg_lifetime_value": float(df["EstimatedLifetimeValue"].mean()) if n else 0.0,
    }


def churn_rate_by(df: pd.DataFrame, column: str) -> pd.DataFrame:
    grouped = (
        df.groupby(column, dropna=False)["ChurnFlag"]
        .agg(["mean", "count"])
        .reset_index()
        .rename(columns={"mean": "churn_rate", "count": "customers"})
    )
    grouped["churn_rate"] = grouped["churn_rate"] * 100
    return grouped.sort_values("churn_rate", ascending=False)


def highest_churn_category(df: pd.DataFrame, column: str) -> tuple[str, float]:
    table = churn_rate_by(df, column)
    if table.empty:
        return "N/A", 0.0
    top = table.iloc[0]
    return str(top[column]), float(top["churn_rate"])


def key_insights(df: pd.DataFrame) -> list[str]:
    insights: list[str] = []
    if df.empty:
        return ["No customer records are available to compute insights."]

    contract, contract_rate = highest_churn_category(df, "Contract")
    insights.append(
        f"The highest churn rate by contract type is {contract} at {contract_rate:.1f}%."
    )

    internet, internet_rate = highest_churn_category(df, "InternetService")
    insights.append(
        f"The highest churn rate by internet service is {internet} at {internet_rate:.1f}%."
    )

    if "TenureGroup" in df.columns:
        tenure_g, tenure_rate = highest_churn_category(df, "TenureGroup")
        insights.append(
            f"The highest churn rate by tenure group is {tenure_g} customers at {tenure_rate:.1f}%."
        )

    payment, payment_rate = highest_churn_category(df, "PaymentMethod")
    insights.append(
        f"The highest churn rate by payment method is {payment} at {payment_rate:.1f}%."
    )

    if "MonthlyCharges" in df.columns and df["ChurnFlag"].nunique() == 2:
        churn_mean = df.loc[df["ChurnFlag"] == 1, "MonthlyCharges"].mean()
        stay_mean = df.loc[df["ChurnFlag"] == 0, "MonthlyCharges"].mean()
        if churn_mean > stay_mean:
            insights.append(
                f"Customers who churned have higher average monthly charges "
                f"(${churn_mean:.2f}) than retained customers (${stay_mean:.2f})."
            )
        else:
            insights.append(
                f"Customers who churned have lower average monthly charges "
                f"(${churn_mean:.2f}) than retained customers (${stay_mean:.2f})."
            )

    low_tenure = df[df["tenure"] <= 12]
    if len(low_tenure):
        low_rate = low_tenure["ChurnFlag"].mean() * 100
        overall = df["ChurnFlag"].mean() * 100
        if low_rate > overall:
            insights.append(
                f"Low-tenure customers (0-12 months) churn at {low_rate:.1f}%, "
                f"above the overall rate of {overall:.1f}%."
            )
        else:
            insights.append(
                f"Low-tenure customers (0-12 months) churn at {low_rate:.1f}%, "
                f"compared with an overall rate of {overall:.1f}%."
            )
    return insights[:5]


def score_customers(df: pd.DataFrame, pipeline) -> pd.DataFrame:
    from src.config import ORIGINAL_PREDICTORS

    feature_cols = [c for c in ORIGINAL_PREDICTORS if c in df.columns]
    scored = predict_dataframe(pipeline, df[feature_cols])
    merged = df.copy()
    merged["ChurnProbability"] = scored["ChurnProbability"].values
    merged["RiskLevel"] = scored["RiskLevel"].values
    merged["PredictionLabel"] = scored["PredictionLabel"].values
    return merged
