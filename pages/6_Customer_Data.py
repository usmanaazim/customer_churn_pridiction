"""Customer data explorer with search, filters, and CSV download."""

from __future__ import annotations

import streamlit as st

from src.ui import get_scored_data, hero, inject_css, require_dataset

st.set_page_config(page_title="Customer Data", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
inject_css()
hero("Customer Data Explorer", "Search, filter, and export customer records used by the churn models")

df = require_dataset()
if df is None:
    st.stop()

scored = get_scored_data()
frame = scored if scored is not None else df
if "TenureGroup" not in frame.columns:
    from src.analytics import prepare_analytics_frame

    frame = prepare_analytics_frame(df)

display_cols = [
    "customerID",
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "InternetService",
    "Contract",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
    "Churn",
]
optional = ["TenureGroup", "TotalServices", "EstimatedLifetimeValue", "ChurnProbability", "RiskLevel"]
for col in optional:
    if col in frame.columns:
        display_cols.append(col)

view = frame[display_cols].copy()
if "ChurnProbability" in view.columns:
    view["ChurnProbability"] = (view["ChurnProbability"] * 100).round(1)

search = st.text_input("Search customer ID")
f1, f2, f3 = st.columns(3)
churn_f = f1.multiselect("Churn", sorted(view["Churn"].unique().tolist()), default=sorted(view["Churn"].unique().tolist()))
contract_f = f2.multiselect("Contract", sorted(view["Contract"].unique().tolist()), default=sorted(view["Contract"].unique().tolist()))
internet_f = f3.multiselect("Internet Service", sorted(view["InternetService"].unique().tolist()), default=sorted(view["InternetService"].unique().tolist()))

filtered = view[view["Churn"].isin(churn_f) & view["Contract"].isin(contract_f) & view["InternetService"].isin(internet_f)]
if search.strip():
    filtered = filtered[filtered["customerID"].astype(str).str.contains(search.strip(), case=False, na=False)]

if "RiskLevel" in filtered.columns:
    risk_f = st.multiselect("Risk level", sorted(filtered["RiskLevel"].dropna().unique().tolist()), default=sorted(filtered["RiskLevel"].dropna().unique().tolist()))
    filtered = filtered[filtered["RiskLevel"].isin(risk_f)]

st.caption(f"{len(filtered):,} of {len(view):,} customers")
st.dataframe(filtered, use_container_width=True, hide_index=True)
st.download_button(
    "Download filtered CSV",
    data=filtered.to_csv(index=False).encode("utf-8"),
    file_name="filtered_customers.csv",
    mime="text/csv",
)
