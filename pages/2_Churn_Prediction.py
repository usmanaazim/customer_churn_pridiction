"""Interactive single-customer churn prediction."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from src.prediction import predict_single
from src.recommendations import generate_recommendations
from src.ui import hero, inject_css, kpi_card, require_pipeline, risk_badge

st.set_page_config(page_title="Churn Prediction", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
inject_css()
hero("Customer Churn Prediction", "Score a customer profile, inspect engineered features, and review retention actions")

pipeline = require_pipeline()
if pipeline is None:
    st.stop()

with st.form("predict_form"):
    st.markdown("### Customer profile")
    p1, p2, p3, p4, p5 = st.columns(5)
    gender = p1.selectbox("Gender", ["Female", "Male"])
    senior = p2.selectbox("Senior Citizen", [0, 1])
    partner = p3.selectbox("Partner", ["Yes", "No"])
    dependents = p4.selectbox("Dependents", ["Yes", "No"])
    tenure = p5.number_input("Tenure (months)", min_value=0, max_value=100, value=12)

    st.markdown("### Services")
    s1, s2, s3 = st.columns(3)
    phone = s1.selectbox("Phone Service", ["Yes", "No"])
    multiple = s2.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    internet = s3.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    s4, s5, s6, s7 = st.columns(4)
    online_sec = s4.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_bak = s5.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device = s6.selectbox("Device Protection", ["No", "Yes", "No internet service"])
    tech = s7.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    s8, s9 = st.columns(2)
    stream_tv = s8.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    stream_mv = s9.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])

    st.markdown("### Billing")
    b1, b2, b3 = st.columns(3)
    contract = b1.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless = b2.selectbox("Paperless Billing", ["Yes", "No"])
    payment = b3.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
    )
    b4, b5 = st.columns(2)
    monthly = b4.number_input("Monthly Charges", min_value=0.0, max_value=250.0, value=70.0, step=0.5)
    total = b5.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=840.0, step=1.0)
    submitted = st.form_submit_button("Predict Churn", use_container_width=True)

if not submitted:
    st.info("Complete the profile and click Predict Churn.")
    st.stop()

if tenure == 0 and total > monthly * 2:
    st.warning("Total charges look high for a new customer. The value will still be scored.")

record = {
    "gender": gender,
    "SeniorCitizen": int(senior),
    "Partner": partner,
    "Dependents": dependents,
    "tenure": int(tenure),
    "PhoneService": phone,
    "MultipleLines": multiple,
    "InternetService": internet,
    "OnlineSecurity": online_sec,
    "OnlineBackup": online_bak,
    "DeviceProtection": device,
    "TechSupport": tech,
    "StreamingTV": stream_tv,
    "StreamingMovies": stream_mv,
    "Contract": contract,
    "PaperlessBilling": paperless,
    "PaymentMethod": payment,
    "MonthlyCharges": float(monthly),
    "TotalCharges": float(total),
}

try:
    result = predict_single(pipeline, record)
except Exception:
    st.error("Prediction could not be completed. Check the inputs and retrained model artifacts.")
    st.stop()

st.markdown("### Prediction result")
m1, m2, m3 = st.columns(3)
with m1:
    kpi_card("Prediction", result["prediction_label"])
with m2:
    kpi_card("Churn Probability", f"{result['probability_pct']:.1f}%")
with m3:
    st.markdown("**Risk**")
    st.markdown(risk_badge(result["risk"]), unsafe_allow_html=True)

st.progress(min(max(result["probability"], 0.0), 1.0))
st.write(result["explanation"])

st.markdown("### Derived customer features")
eng = result["engineered_features"]
labels = [
    ("Average Monthly Spend", f"${float(eng['AverageMonthlySpend']):.2f}"),
    ("Total Services", str(int(eng["TotalServices"]))),
    ("Estimated Lifetime Value", f"${float(eng['EstimatedLifetimeValue']):.2f}"),
    ("Tenure Group", str(eng["TenureGroup"])),
    ("High Monthly Charge", "Yes" if int(eng["HighMonthlyCharge"]) else "No"),
    ("Long-Term Contract", "Yes" if int(eng["IsLongTermContract"]) else "No"),
    ("Technical Support", "Yes" if int(eng["HasTechnicalSupport"]) else "No"),
    ("Online Security", "Yes" if int(eng["HasOnlineSecurity"]) else "No"),
    ("Service Engagement Score", f"{float(eng['ServiceEngagementScore']):.1f}"),
    ("Customer Value Segment", str(eng["CustomerValueSegment"])),
]
cols = st.columns(5)
for i, (label, value) in enumerate(labels):
    with cols[i % 5]:
        kpi_card(label, value)

st.markdown("### Rule-based retention recommendations")
st.caption("These actions are generated by business rules, not by the machine learning model.")
for rec in generate_recommendations(record, result["probability"], result["risk"]):
    st.markdown(f"- {rec}")
