"""Project configuration: paths, columns, thresholds, and model settings."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
ASSETS_DIR = PROJECT_ROOT / "assets"
DATA_PATH = DATA_DIR / "customer_churn.csv"

DATASET_URLS = [
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/treselle-systems/customer_churn_analysis/master/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    "https://raw.githubusercontent.com/dsrscientist/DSData/master/Telecom_customer_churn.csv",
]

REQUIRED_COLUMNS = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]

ID_COLUMN = "customerID"
TARGET_COLUMN = "Churn"
TARGET_POSITIVE = "Yes"
TARGET_NEGATIVE = "No"

ORIGINAL_PREDICTORS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]

SERVICE_COLUMNS = [
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
]

ENGINEERED_FEATURES = [
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

ENGINEERED_FEATURE_DESCRIPTIONS = {
    "AverageMonthlySpend": "TotalCharges / tenure (MonthlyCharges when tenure is 0).",
    "TotalServices": "Count of active services among phone, lines, security, backup, protection, support, and streaming.",
    "EstimatedLifetimeValue": "MonthlyCharges multiplied by tenure.",
    "TenureGroup": "Tenure buckets: New (0-12), Short (13-24), Medium (25-48), Long (49+).",
    "HighMonthlyCharge": "1 if MonthlyCharges is at or above the training-set median.",
    "IsLongTermContract": "1 if contract is One year or Two year, else 0.",
    "HasTechnicalSupport": "1 if TechSupport is Yes, else 0.",
    "HasOnlineSecurity": "1 if OnlineSecurity is Yes, else 0.",
    "ServiceEngagementScore": "0-100 score from active services, support add-ons, and streaming.",
    "CustomerValueSegment": "Low / Medium / High value from training-set lifetime-value tertiles.",
}

NUMERIC_FEATURES = [
    "SeniorCitizen",
    "tenure",
    "MonthlyCharges",
    "TotalCharges",
    "AverageMonthlySpend",
    "TotalServices",
    "EstimatedLifetimeValue",
    "HighMonthlyCharge",
    "IsLongTermContract",
    "HasTechnicalSupport",
    "HasOnlineSecurity",
    "ServiceEngagementScore",
]

CATEGORICAL_FEATURES = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "TenureGroup",
    "CustomerValueSegment",
]

DROPPED_FEATURES = [ID_COLUMN]

RANDOM_STATE = 42
TEST_SIZE = 0.20
PCA_VARIANCE = 0.95
CV_FOLDS = 5
SVM_CV_FOLDS = 3

RISK_LOW_MAX = 0.30
RISK_MEDIUM_MAX = 0.60

BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
PREPROCESSING_PATH = MODELS_DIR / "preprocessing_pipeline.joblib"
FULL_PIPELINE_PATH = MODELS_DIR / "full_churn_pipeline.joblib"
MODEL_METADATA_PATH = MODELS_DIR / "model_metadata.json"
PCA_METADATA_PATH = MODELS_DIR / "pca_metadata.json"
FEATURE_METADATA_PATH = MODELS_DIR / "feature_metadata.json"
COMPARISON_PATH = MODELS_DIR / "model_comparison.csv"
EVALUATION_CURVES_PATH = MODELS_DIR / "evaluation_curves.joblib"

TENURE_BINS = [0, 12, 24, 48, 10_000]
TENURE_LABELS = ["New", "Short", "Medium", "Long"]

LONG_TERM_CONTRACTS = {"One year", "Two year"}
