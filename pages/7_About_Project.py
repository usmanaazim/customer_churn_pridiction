"""Academic project documentation for viva and HOD demonstration."""

from __future__ import annotations

import streamlit as st

from src.ui import hero, inject_css, load_metadata

st.set_page_config(page_title="About Project", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
inject_css()
hero("Customer Churn Prediction Using Feature Engineering and PCA", "Final-year academic machine learning system for customer retention")

meta = load_metadata() or {}

st.markdown("### Problem statement")
st.write(
    "Service businesses lose revenue when customers cancel. Historical account, service, and billing "
    "data can be used to estimate the probability that a customer will churn, so retention teams can act earlier."
)

st.markdown("### Why customer churn matters")
st.write(
    "Acquiring a new customer typically costs more than retaining an existing one. Even a modest reduction "
    "in churn improves lifetime value, forecast stability, and campaign efficiency."
)

st.markdown("### Project objectives")
st.markdown(
    """
- Clean and validate the Telco Customer Churn dataset without fabricating records.
- Engineer interpretable customer features that capture spend, tenure, and service engagement.
- Apply PCA to retain approximately 95% of encoded-feature variance.
- Compare Logistic Regression, Random Forest, SVM, and Gradient Boosting with and without PCA.
- Select a production model using F1 and ROC-AUC rather than accuracy alone.
- Deliver churn probability, risk class, and rule-based retention recommendations in Streamlit.
"""
)

st.markdown("### Methodology")
st.write(
    "Records are cleaned, split with stratification (test size 0.2, random state 42), then transformed with a "
    "fitted FeatureEngineer. Categorical fields are one-hot encoded (`handle_unknown='ignore'`). Numeric fields "
    "are imputed and scaled. PCA (`n_components=0.95`) is evaluated as an optional stage. Models are tuned with "
    "cross-validated F1 and compared on a held-out test set."
)

st.markdown("### Architecture")
st.markdown(
    """
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
"""
)

st.markdown("### Feature engineering")
st.write(
    "Ten derived fields (average monthly spend, service counts, lifetime value, tenure group, high-charge flag, "
    "long-term contract, support/security flags, engagement score, and value segment) are produced by one reusable "
    "transformer fitted only on training data so median and tertile thresholds do not leak from the test set."
)

st.markdown("### PCA")
st.write(
    "After encoding and scaling, principal components keep enough variance to reconstruct most of the signal while "
    "reducing collinearity and dimensionality. Component-level plots are not original feature names."
)

st.markdown("### Machine learning")
st.write(
    "Linear and tree ensembles provide complementary inductive biases. Class weights address churn imbalance. "
    "The stored `full_churn_pipeline.joblib` applies feature engineering, preprocessing, optional PCA, and the classifier."
)
if meta:
    st.success(f"Currently selected model: {meta.get('model_name')} ({meta.get('variant')})")

st.markdown("### Evaluation metrics")
st.write(
    "Accuracy, precision, recall, F1, and ROC-AUC are reported on the test set, with a confusion matrix, ROC curve, "
    "and precision-recall curve. F1 and ROC-AUC drive model selection because accuracy can hide poor churn detection."
)

st.markdown("### Business benefits")
st.write(
    "The dashboard highlights high-risk accounts, contract and payment segments with elevated churn, and suggested "
    "retention offers. Recommendations are explicitly rule-based."
)

st.markdown("### Limitations")
st.markdown(
    """
- Churn prediction depends on historical patterns and can drift if products or pricing change.
- PCA improves dimensionality reduction but reduces direct interpretability of original fields.
- Predictions are probabilities, not guarantees.
- Retention recommendations are rule-based, not learned policies.
- The public telco dataset may not represent every real-world business.
"""
)

st.markdown("### Future scope")
st.markdown(
    """
- Real-time churn monitoring
- Customer feedback and sentiment analysis
- Explainable AI beyond non-PCA importances
- Automated retention campaigns and CRM integration
- Continuous retraining and, for much larger data, deep learning
"""
)

st.markdown("### Technology stack")
st.write("Python, pandas, NumPy, scikit-learn, Matplotlib, Seaborn, Plotly, Joblib, Streamlit, pytest.")

st.markdown("### How to run")
st.code(
    "python3 -m venv venv\nsource venv/bin/activate\npip install -r requirements.txt\n"
    "python3 scripts/setup_data.py\npython3 scripts/train_model.py\nstreamlit run app.py\npytest",
    language="bash",
)
