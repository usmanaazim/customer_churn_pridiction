# Customer Churn Prediction Using Feature Engineering and PCA

AI-powered **Customer Churn Intelligence** for a final-year academic project: predict whether a telecom customer will leave, estimate churn probability, classify risk, and suggest rule-based retention actions.

## Project overview

Customer churn means a customer stops using a company’s service. This system learns from the public Telco Customer Churn dataset, engineers extra customer features, optionally reduces dimensionality with PCA, and compares several scikit-learn classifiers. A Streamlit dashboard supports demonstration, viva, and portfolio use.

## Problem statement

Retention teams need an early signal of cancellation risk. Raw billing and service fields are noisy and collinear. The project asks: after careful cleaning and **feature engineering**, does **PCA** (retaining ~95% variance) help or hurt predictive quality, and which model should be deployed?

## Objectives

- Prepare the Telco dataset without fabricating customers.
- Engineer reusable features (spend, services, tenure, value).
- Encode categoricals with `OneHotEncoder(handle_unknown="ignore")` and scale numerics in a sklearn pipeline.
- Train models **with and without PCA** and compare Accuracy, Precision, Recall, F1, and ROC-AUC.
- Tune hyperparameters with cross-validated F1.
- Select the best model using **F1 then ROC-AUC** (not accuracy alone).
- Serve probability, risk (Low / Medium / High), and retention guidance in Streamlit.

## Features

- Dataset validation and cleaning (`TotalCharges` numeric handling, missing values).
- Exploratory analysis in Streamlit and `notebooks/churn_analysis.py`.
- Ten documented engineered features fitted only on training data.
- PCA at 95% explained variance, with metadata and plots.
- Logistic Regression, Random Forest, SVM, Gradient Boosting.
- Saved pipelines, metadata JSON, and comparison CSV.
- Prediction UI plus CLI (`scripts/predict_customer.py`).
- High-risk customer table and CSV download.
- Pytest coverage for data, features, risk, and artifacts.

## Architecture

```
Customer Dataset
        ↓
Data Cleaning
        ↓
EDA
        ↓
Feature Engineering
        ↓
Encoding
        ↓
Scaling
        ↓
PCA
        ↓
Machine Learning
        ↓
Model Evaluation
        ↓
Best Model
        ↓
Churn Probability
        ↓
Risk Classification
        ↓
Retention Recommendation
```

## Dataset

**Name:** Telco Customer Churn (IBM / community mirror)  
**Path:** `data/customer_churn.csv`  
**Target:** `Churn` (`Yes` = churned, `No` = stayed)

If the file is missing, the app shows a warning. Run `python3 scripts/setup_data.py` to download a public copy, or follow `data/README.md`.

## Feature engineering

Implemented in `src/feature_engineering.py` (`FeatureEngineer`) and reused in training, evaluation, and prediction:

| Feature | Definition |
| --- | --- |
| AverageMonthlySpend | `TotalCharges / tenure` (uses `MonthlyCharges` if tenure is 0) |
| TotalServices | Count of active listed services |
| EstimatedLifetimeValue | `MonthlyCharges * tenure` |
| TenureGroup | New 0–12, Short 13–24, Medium 25–48, Long 49+ |
| HighMonthlyCharge | 1 if monthly charge ≥ **training** median |
| IsLongTermContract | 1 for One year / Two year |
| HasTechnicalSupport | 1 if TechSupport = Yes |
| HasOnlineSecurity | 1 if OnlineSecurity = Yes |
| ServiceEngagementScore | 0–100 score from services and add-ons |
| CustomerValueSegment | Low / Medium / High from **training** LTV tertiles |

## PCA

After encoding and scaling, `PCA(n_components=0.95)` retains about 95% of variance. Actual component counts and explained-variance ratios are written to `models/pca_metadata.json` during training (not hardcoded).

PCA-based models act on principal components. Original-feature importance is reported from **non-PCA** Random Forest and Logistic Regression.

## Machine learning models

- Logistic Regression  
- Random Forest  
- Support Vector Machine (probability enabled)  
- Gradient Boosting (scikit-learn)

Each is trained **without PCA** and **with PCA**, then tuned with `GridSearchCV` or `RandomizedSearchCV` (`scoring="f1"`).

## Evaluation metrics

Accuracy, Precision, Recall, F1 Score, ROC-AUC, confusion matrix, ROC curve, precision-recall curve. Artifacts live under `models/`.

## Project structure

```
Customer_Churn_Prediction/
├── app.py
├── requirements.txt
├── README.md
├── data/
├── models/
├── scripts/
├── notebooks/
├── src/
├── pages/
├── assets/
└── tests/
```

## Installation (macOS / MacBook)

```bash
cd Customer_Churn_Prediction
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python3 scripts/setup_data.py
```

## Training

```bash
python3 scripts/train_model.py
```

This loads the CSV, engineers features, splits with `test_size=0.2`, `random_state=42`, `stratify=y`, trains baseline and PCA models, tunes them, selects the best, and writes:

- `models/full_churn_pipeline.joblib` — feature engineering + preprocess + optional PCA + classifier  
- `models/best_model.joblib` — classifier only  
- `models/preprocessing_pipeline.joblib`  
- `models/model_metadata.json`  
- `models/pca_metadata.json`  
- `models/feature_metadata.json`  
- `models/model_comparison.csv`  

Retrain any time by running the same command again (artifacts are overwritten).

## Streamlit

```bash
streamlit run app.py
```

The app uses a wide layout. Open Dashboard, Prediction, Analytics, Feature Engineering & PCA, Model Performance, Customer Data, and About.

## Command-line prediction

```bash
python3 scripts/predict_customer.py
```

Optional JSON customer file:

```bash
python3 scripts/predict_customer.py --json path/to/customer.json
```

Output includes prediction, probability, risk, engineered features, and rule-based recommendations.

## Tests

```bash
pytest
```

## Risk thresholds

Configured in `src/config.py`:

- Low: probability &lt; 0.30  
- Medium: 0.30 ≤ probability &lt; 0.60  
- High: probability ≥ 0.60  

## Limitations

- Historical patterns may not hold if pricing or products change.  
- PCA reduces direct interpretability of original variables.  
- Outputs are probabilities, not certainties.  
- Retention text is **rule-based**, not a second ML model.  
- One public telco table does not represent every business.

## Future scope

Real-time monitoring, feedback/sentiment, richer explainable AI, automated campaigns, CRM integration, continuous retraining, and deep learning on larger datasets.

## License / academic use

Intended for academic demonstration and portfolio. Cite the Telco Customer Churn dataset source if you publish results.
