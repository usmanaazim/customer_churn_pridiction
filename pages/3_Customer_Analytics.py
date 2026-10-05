"""Filtered customer analytics with interactive Plotly charts."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.analytics import kpi_summary
from src.ui import PLOTLY_LAYOUT, get_scored_data, hero, inject_css, kpi_card, require_dataset

st.set_page_config(page_title="Customer Analytics", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
inject_css()
hero("Customer Analytics", "Filter the base and inspect how churn changes across segments")

df = require_dataset()
if df is None:
    st.stop()

scored = get_scored_data()
frame = scored if scored is not None else df
if "ChurnFlag" not in frame.columns or "TenureGroup" not in frame.columns:
    from src.analytics import prepare_analytics_frame

    frame = prepare_analytics_frame(df)

st.sidebar.header("Filters")
churn_opt = st.sidebar.multiselect("Churn", sorted(frame["Churn"].dropna().unique().tolist()), default=sorted(frame["Churn"].dropna().unique().tolist()))
contract_opt = st.sidebar.multiselect("Contract", sorted(frame["Contract"].dropna().unique().tolist()), default=sorted(frame["Contract"].dropna().unique().tolist()))
internet_opt = st.sidebar.multiselect("Internet Service", sorted(frame["InternetService"].dropna().unique().tolist()), default=sorted(frame["InternetService"].dropna().unique().tolist()))
pay_opt = st.sidebar.multiselect("Payment Method", sorted(frame["PaymentMethod"].dropna().unique().tolist()), default=sorted(frame["PaymentMethod"].dropna().unique().tolist()))
gender_opt = st.sidebar.multiselect("Gender", sorted(frame["gender"].dropna().unique().tolist()), default=sorted(frame["gender"].dropna().unique().tolist()))
tenure_opt = st.sidebar.multiselect("Tenure Group", sorted(frame["TenureGroup"].dropna().unique().tolist()), default=sorted(frame["TenureGroup"].dropna().unique().tolist()))
senior_opt = st.sidebar.multiselect("Senior Citizen", sorted(frame["SeniorCitizen"].dropna().unique().tolist()), default=sorted(frame["SeniorCitizen"].dropna().unique().tolist()))

t_min, t_max = int(frame["tenure"].min()), int(frame["tenure"].max())
tenure_range = st.sidebar.slider("Tenure range", t_min, t_max, (t_min, t_max))
m_min, m_max = float(frame["MonthlyCharges"].min()), float(frame["MonthlyCharges"].max())
monthly_range = st.sidebar.slider("Monthly charges range", m_min, m_max, (m_min, m_max))
tot_min, tot_max = float(frame["TotalCharges"].min()), float(frame["TotalCharges"].max())
total_range = st.sidebar.slider("Total charges range", tot_min, tot_max, (tot_min, tot_max))

filtered = frame[
    frame["Churn"].isin(churn_opt)
    & frame["Contract"].isin(contract_opt)
    & frame["InternetService"].isin(internet_opt)
    & frame["PaymentMethod"].isin(pay_opt)
    & frame["gender"].isin(gender_opt)
    & frame["TenureGroup"].isin(tenure_opt)
    & frame["SeniorCitizen"].isin(senior_opt)
    & frame["tenure"].between(tenure_range[0], tenure_range[1])
    & frame["MonthlyCharges"].between(monthly_range[0], monthly_range[1])
    & frame["TotalCharges"].between(total_range[0], total_range[1])
].copy()

kpis = kpi_summary(filtered)
a, b = st.columns(2)
with a:
    kpi_card("Filtered Customers", f"{kpis['total_customers']:,}")
with b:
    kpi_card("Filtered Churn Rate", f"{kpis['churn_rate'] * 100:.1f}%")

if filtered.empty:
    st.warning("No customers match the current filters.")
    st.stop()

filtered["ChurnLabel"] = filtered["ChurnFlag"].map({0: "Stayed", 1: "Churned"})
color_map = {"Stayed": "#1F8A4C", "Churned": "#C0392B"}


def rate_chart(column: str, title: str):
    table = filtered.groupby(column)["ChurnFlag"].mean().reset_index()
    table["churn_rate"] = table["ChurnFlag"] * 100
    fig = px.bar(table, x=column, y="churn_rate", title=title, labels={"churn_rate": "Churn rate (%)"})
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig


c1, c2 = st.columns(2)
c1.plotly_chart(rate_chart("Contract", "Churn rate by contract"), use_container_width=True)
c2.plotly_chart(rate_chart("InternetService", "Churn rate by internet service"), use_container_width=True)

c3, c4 = st.columns(2)
c3.plotly_chart(rate_chart("TenureGroup", "Churn rate by tenure group"), use_container_width=True)
c4.plotly_chart(rate_chart("PaymentMethod", "Churn rate by payment method"), use_container_width=True)

c5, c6 = st.columns(2)
c5.plotly_chart(rate_chart("gender", "Churn rate by gender"), use_container_width=True)
c6.plotly_chart(rate_chart("SeniorCitizen", "Churn rate by senior citizen"), use_container_width=True)

h1, h2 = st.columns(2)
fig = px.histogram(filtered, x="MonthlyCharges", color="ChurnLabel", barmode="overlay", opacity=0.7, color_discrete_map=color_map, title="Monthly charges vs churn")
fig.update_layout(**PLOTLY_LAYOUT)
h1.plotly_chart(fig, use_container_width=True)
fig = px.histogram(filtered, x="tenure", color="ChurnLabel", barmode="overlay", opacity=0.7, color_discrete_map=color_map, title="Tenure vs churn")
fig.update_layout(**PLOTLY_LAYOUT)
h2.plotly_chart(fig, use_container_width=True)

fig = px.histogram(filtered, x="TotalCharges", color="ChurnLabel", barmode="overlay", opacity=0.7, color_discrete_map=color_map, title="Total charges vs churn")
fig.update_layout(**PLOTLY_LAYOUT)
st.plotly_chart(fig, use_container_width=True)

if "TotalServices" in filtered.columns:
    fig = px.box(filtered, x="ChurnLabel", y="TotalServices", color="ChurnLabel", color_discrete_map=color_map, title="Service count vs churn")
    fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)
