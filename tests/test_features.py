import pandas as pd

from src.feature_engineering import FeatureEngineer, count_total_services


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "gender": ["Female", "Male", "Female"],
            "SeniorCitizen": [0, 0, 1],
            "Partner": ["No", "Yes", "No"],
            "Dependents": ["No", "Yes", "No"],
            "tenure": [0, 18, 60],
            "PhoneService": ["Yes", "Yes", "No"],
            "MultipleLines": ["No", "Yes", "No phone service"],
            "InternetService": ["Fiber optic", "DSL", "No"],
            "OnlineSecurity": ["No", "Yes", "No internet service"],
            "OnlineBackup": ["No", "Yes", "No internet service"],
            "DeviceProtection": ["No", "No", "No internet service"],
            "TechSupport": ["No", "Yes", "No internet service"],
            "StreamingTV": ["Yes", "No", "No internet service"],
            "StreamingMovies": ["No", "Yes", "No internet service"],
            "Contract": ["Month-to-month", "One year", "Two year"],
            "PaperlessBilling": ["Yes", "No", "No"],
            "PaymentMethod": ["Electronic check", "Mailed check", "Credit card (automatic)"],
            "MonthlyCharges": [90.0, 40.0, 20.0],
            "TotalCharges": [0.0, 720.0, 1200.0],
        }
    )


def test_total_services_counts_yes_only():
    df = _frame()
    counts = count_total_services(df)
    assert counts.iloc[0] == 2
    assert counts.iloc[1] == 6
    assert counts.iloc[2] == 0


def test_feature_engineer_handles_zero_tenure():
    engineer = FeatureEngineer().fit(_frame())
    out = engineer.transform(_frame())
    assert out.loc[0, "AverageMonthlySpend"] == 90.0
    assert out.loc[1, "AverageMonthlySpend"] == 40.0
    assert out.loc[0, "TenureGroup"] == "New"
    assert out.loc[1, "TenureGroup"] == "Short"
    assert out.loc[2, "TenureGroup"] == "Long"
    assert out.loc[1, "IsLongTermContract"] == 1
    assert out.loc[0, "IsLongTermContract"] == 0
    assert out.loc[0, "HasTechnicalSupport"] == 0
    assert out.loc[1, "HasOnlineSecurity"] == 1
    assert out["ServiceEngagementScore"].between(0, 100).all()
    assert set(out["CustomerValueSegment"].unique()) <= {"Low Value", "Medium Value", "High Value"}


def test_high_monthly_charge_uses_training_median():
    engineer = FeatureEngineer().fit(_frame())
    assert engineer.monthly_median_ == 40.0
    out = engineer.transform(_frame())
    assert out.loc[0, "HighMonthlyCharge"] == 1
    assert out.loc[1, "HighMonthlyCharge"] == 1
    assert out.loc[2, "HighMonthlyCharge"] == 0


def test_same_transformer_is_reusable():
    engineer = FeatureEngineer().fit(_frame())
    first = engineer.transform(_frame().iloc[[0]])
    second = engineer.transform(_frame().iloc[[0]])
    assert first["AverageMonthlySpend"].iloc[0] == second["AverageMonthlySpend"].iloc[0]
    assert first["CustomerValueSegment"].iloc[0] == second["CustomerValueSegment"].iloc[0]
