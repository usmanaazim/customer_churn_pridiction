"""Reusable feature engineering fitted only on training data."""

from __future__ import annotations

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import (
    ENGINEERED_FEATURES,
    LONG_TERM_CONTRACTS,
    SERVICE_COLUMNS,
    TENURE_BINS,
    TENURE_LABELS,
)


def _is_yes(series: pd.Series) -> pd.Series:
    """Return True for values representing 'Yes'."""
    return series.astype(str).str.strip().str.lower().eq("yes")


def tenure_group_from_values(tenure: pd.Series) -> pd.Series:
    """Convert numeric tenure into meaningful customer tenure groups."""
    return pd.cut(
        tenure.astype(float),
        bins=TENURE_BINS,
        labels=TENURE_LABELS,
        include_lowest=True,
        right=True,
    ).astype(str)


def count_total_services(df: pd.DataFrame) -> pd.Series:
    """Count active customer services."""
    total = pd.Series(0, index=df.index, dtype=int)

    service_columns = [
        "PhoneService",
        "MultipleLines",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
    ]

    for col in service_columns:
        if col in df.columns:
            total += _is_yes(df[col]).astype(int)

    return total


def service_engagement_score(
    df: pd.DataFrame,
    total_services: pd.Series,
) -> pd.Series:
    """
    Calculate a customer service engagement score from 0 to 100.
    """

    max_services = max(len(SERVICE_COLUMNS), 1)

    base = (total_services / max_services) * 70.0

    support = pd.Series(0.0, index=df.index)

    if "TechSupport" in df.columns:
        support += _is_yes(df["TechSupport"]).astype(float) * 10.0

    if "OnlineSecurity" in df.columns:
        support += _is_yes(df["OnlineSecurity"]).astype(float) * 10.0

    streaming = pd.Series(0.0, index=df.index)

    if "StreamingTV" in df.columns:
        streaming += _is_yes(df["StreamingTV"]).astype(float) * 5.0

    if "StreamingMovies" in df.columns:
        streaming += _is_yes(df["StreamingMovies"]).astype(float) * 5.0

    return (base + support + streaming).clip(0, 100)


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Adds engineered features.

    Important:
    Thresholds such as the monthly charge median and lifetime-value
    quantiles are learned only from the training data.
    """

    def __init__(self):
        self.monthly_median_ = None
        self.eltv_q33_ = None
        self.eltv_q66_ = None
        self.feature_names_in_ = None
        self.feature_names_out_ = None

    def fit(self, X, y=None):
        """
        Learn feature-engineering thresholds from the training data.

        IMPORTANT:
        We do not call the public transform() method here because
        scikit-learn may wrap transform() with pandas output handling,
        which can call get_feature_names_out() before fitting is complete.
        """

        df = pd.DataFrame(X).copy()

        # Store original feature names.
        self.feature_names_in_ = list(df.columns)

        # Learn MonthlyCharges threshold from training data only.
        self.monthly_median_ = float(
            pd.to_numeric(
                df["MonthlyCharges"],
                errors="coerce",
            ).median()
        )

        # Learn customer lifetime value thresholds from training data only.
        monthly = pd.to_numeric(
            df["MonthlyCharges"],
            errors="coerce",
        )

        tenure = pd.to_numeric(
            df["tenure"],
            errors="coerce",
        )

        eltv = monthly * tenure

        self.eltv_q33_ = float(eltv.quantile(0.33))
        self.eltv_q66_ = float(eltv.quantile(0.66))

        # Determine output feature names BEFORE transform is called.
        self.feature_names_out_ = list(self.feature_names_in_)

        for feature in ENGINEERED_FEATURES:
            if feature not in self.feature_names_out_:
                self.feature_names_out_.append(feature)

        return self

    def transform(self, X):
        """Apply learned feature engineering to new data."""

        if self.monthly_median_ is None:
            raise RuntimeError(
                "FeatureEngineer must be fitted before transform."
            )

        df = pd.DataFrame(X).copy()

        # Make sure required numeric columns are numeric.
        tenure = pd.to_numeric(
            df["tenure"],
            errors="coerce",
        )

        monthly = pd.to_numeric(
            df["MonthlyCharges"],
            errors="coerce",
        )

        total = pd.to_numeric(
            df["TotalCharges"],
            errors="coerce",
        )

        # ---------------------------------------------------------
        # 1. Average Monthly Spend
        # ---------------------------------------------------------

        avg_spend = total / tenure.replace(0, pd.NA)

        avg_spend = avg_spend.fillna(monthly)

        df["AverageMonthlySpend"] = avg_spend.astype(float)

        # ---------------------------------------------------------
        # 2. Total Services
        # ---------------------------------------------------------

        df["TotalServices"] = count_total_services(df)

        # ---------------------------------------------------------
        # 3. Estimated Lifetime Value
        # ---------------------------------------------------------

        df["EstimatedLifetimeValue"] = monthly * tenure

        # ---------------------------------------------------------
        # 4. Tenure Group
        # ---------------------------------------------------------

        df["TenureGroup"] = tenure_group_from_values(tenure)

        # ---------------------------------------------------------
        # 5. High Monthly Charge
        # ---------------------------------------------------------

        df["HighMonthlyCharge"] = (
            monthly >= self.monthly_median_
        ).astype(int)

        # ---------------------------------------------------------
        # 6. Long-Term Contract
        # ---------------------------------------------------------

        contract = df["Contract"].astype(str)

        df["IsLongTermContract"] = (
            contract.isin(LONG_TERM_CONTRACTS)
        ).astype(int)

        # ---------------------------------------------------------
        # 7. Technical Support
        # ---------------------------------------------------------

        if "TechSupport" in df.columns:
            df["HasTechnicalSupport"] = (
                _is_yes(df["TechSupport"])
            ).astype(int)
        else:
            df["HasTechnicalSupport"] = 0

        # ---------------------------------------------------------
        # 8. Online Security
        # ---------------------------------------------------------

        if "OnlineSecurity" in df.columns:
            df["HasOnlineSecurity"] = (
                _is_yes(df["OnlineSecurity"])
            ).astype(int)
        else:
            df["HasOnlineSecurity"] = 0

        # ---------------------------------------------------------
        # 9. Service Engagement Score
        # ---------------------------------------------------------

        df["ServiceEngagementScore"] = service_engagement_score(
            df,
            df["TotalServices"],
        )

        # ---------------------------------------------------------
        # 10. Customer Value Segment
        # ---------------------------------------------------------

        df["CustomerValueSegment"] = self._value_segment(
            df["EstimatedLifetimeValue"]
        )

        # Keep output columns consistent.
        # This is important for the sklearn pipeline.
        expected_columns = list(self.feature_names_out_)

        # Add any missing expected columns.
        for column in expected_columns:
            if column not in df.columns:
                df[column] = 0

        # Return in the exact same order used during training.
        return df[expected_columns]

    def _value_segment(self, eltv: pd.Series) -> pd.Series:
        """Create customer value segments using training thresholds."""

        labels = pd.Series(
            "Medium Value",
            index=eltv.index,
            dtype=object,
        )

        labels.loc[eltv <= self.eltv_q33_] = "Low Value"

        labels.loc[eltv > self.eltv_q66_] = "High Value"

        return labels

    def get_feature_names_out(self, input_features=None):
        """
        Return output feature names.

        This method is safe to call after fit().
        """

        if self.feature_names_out_ is None:
            raise RuntimeError(
                "FeatureEngineer is not fitted."
            )

        return self.feature_names_out_


def engineer_features(
    df: pd.DataFrame,
    engineer: FeatureEngineer | None = None,
) -> pd.DataFrame:
    """
    Apply the same fitted engineer used in training.

    If no engineer is provided, fit a new one.
    This behavior is useful for EDA.
    """

    if engineer is None:
        engineer = FeatureEngineer()

        return engineer.fit_transform(df)

    return engineer.transform(df)


def engineered_feature_summary(
    engineer: FeatureEngineer,
) -> dict:
    """Return metadata about engineered features."""

    return {
        "engineered_features": ENGINEERED_FEATURES,
        "monthly_median_threshold": engineer.monthly_median_,
        "lifetime_value_q33": engineer.eltv_q33_,
        "lifetime_value_q66": engineer.eltv_q66_,
    }