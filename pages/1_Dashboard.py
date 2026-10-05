"""Executive churn dashboard with KPIs, charts, insights, and high-risk table."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.analytics import kpi_summary, key_insights
from src.ui import (
    PLOTLY_LAYOUT,
    get_scored_data,
    hero,
    inject_css,
    kpi_card,
    require_dataset,
)

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
inject_css()
hero("Customer Churn Intelligence", "AI-powered customer retention and churn analytics")

df = require_dataset()
if df is None:
    st.stop()

scored = get_scored_data()
frame = scored if scored is not None else df
if "ChurnFlag" not in frame.columns or "TenureGroup" not in frame.columns:
    from src.analytics import prepare_analytics_frame

    frame = prepare_analytics_frame(df)

frame = frame.copy()
frame["ChurnLabel"] = frame["ChurnFlag"].map({0: "Stayed", 1: "Churned"})

kpis = kpi_summary(frame)
r1 = st.columns(4)
values = [
    ("Total Customers", f"{kpis['total_customers']:,}"),
    ("Churned Customers", f"{kpis['churned_customers']:,}"),
    ("Retained Customers", f"{kpis['retained_customers']:,}"),
    ("Overall Churn Rate", f"{kpis['churn_rate'] * 100:.1f}%"),
]
for col, (label, val) in zip(r1, values):
    with col:
        kpi_card(label, val)

r2 = st.columns(4)
high_risk = kpis["high_risk_customers"]
values2 = [
    ("High-Risk Customers", "Train model to score" if high_risk is None else f"{high_risk:,}"),
    ("Avg Monthly Charges", f"${kpis['avg_monthly_charges']:.2f}"),
    ("Average Tenure", f"{kpis['avg_tenure']:.1f} months"),
    ("Avg Customer LTV", f"${kpis['avg_lifetime_value']:.0f}"),
]
for col, (label, val) in zip(r2, values2):
    with col:
        kpi_card(label, val)

color_map = {"Stayed": "#1F8A4C", "Churned": "#C0392B"}


def style(fig, title):
    fig.update_layout(**PLOTLY_LAYOUT, title=title)
    return fig


c1, c2 = st.columns(2)
with c1:
    pie = px.pie(
        frame,
        names="ChurnLabel",
        hole=0.58,
        color="ChurnLabel",
        color_discrete_map=color_map,
    )
    st.plotly_chart(style(pie, "Churn vs no churn"), use_container_width=True)
with c2:
    contract = (
        frame.groupby(["Contract", "ChurnLabel"]).size().reset_index(name="customers")
    )
    bar = px.bar(contract, x="Contract", y="customers", color="ChurnLabel", barmode="group", color_discrete_map=color_map)
    st.plotly_chart(style(bar, "Churn by contract"), use_container_width=True)

c3, c4 = st.columns(2)
with c3:
    tenure = frame.groupby(["TenureGroup", "ChurnLabel"]).size().reset_index(name="customers")
    bar = px.bar(tenure, x="TenureGroup", y="customers", color="ChurnLabel", barmode="group", color_discrete_map=color_map, category_orders={"TenureGroup": ["New", "Short", "Medium", "Long"]})
    st.plotly_chart(style(bar, "Churn by tenure group"), use_container_width=True)
with c4:
    internet = frame.groupby(["InternetService", "ChurnLabel"]).size().reset_index(name="customers")
    bar = px.bar(internet, x="InternetService", y="customers", color="ChurnLabel", barmode="group", color_discrete_map=color_map)
    st.plotly_chart(style(bar, "Churn by internet service"), use_container_width=True)

c5, c6 = st.columns(2)
with c5:
    pay = frame.groupby(["PaymentMethod", "ChurnLabel"]).size().reset_index(name="customers")
    bar = px.bar(pay, x="PaymentMethod", y="customers", color="ChurnLabel", barmode="group", color_discrete_map=color_map)
    bar.update_xaxes(tickangle=-20)
    st.plotly_chart(style(bar, "Churn by payment method"), use_container_width=True)
with c6:
    tech = frame.groupby(["TechSupport", "ChurnLabel"]).size().reset_index(name="customers")
    bar = px.bar(tech, x="TechSupport", y="customers", color="ChurnLabel", barmode="group", color_discrete_map=color_map)
    st.plotly_chart(style(bar, "Churn by tech support"), use_container_width=True)

c7, c8 = st.columns(2)
with c7:
    hist = px.histogram(frame, x="MonthlyCharges", color="ChurnLabel", nbins=40, color_discrete_map=color_map, barmode="overlay", opacity=0.7)
    st.plotly_chart(style(hist, "Monthly charges distribution"), use_container_width=True)
with c8:
    hist = px.histogram(frame, x="tenure", color="ChurnLabel", nbins=36, color_discrete_map=color_map, barmode="overlay", opacity=0.7)
    st.plotly_chart(style(hist, "Tenure distribution"), use_container_width=True)

scatter = px.scatter(
    frame.sample(min(len(frame), 2500), random_state=42),
    x="tenure",
    y="MonthlyCharges",
    color="ChurnLabel",
    color_discrete_map=color_map,
    opacity=0.55,
)
st.plotly_chart(style(scatter, "Monthly charges vs tenure"), use_container_width=True)

c9, c10 = st.columns(2)
with c9:
    sc = px.scatter(
        frame.sample(min(len(frame), 2500), random_state=42),
        x="MonthlyCharges",
        y="TotalCharges",
        color="ChurnLabel",
        color_discrete_map=color_map,
        opacity=0.55,
    )
    st.plotly_chart(style(sc, "Total charges vs monthly charges"), use_container_width=True)
with c10:
    if "TotalServices" in frame.columns:
        svc = frame.groupby(["TotalServices", "ChurnLabel"]).size().reset_index(name="customers")
        line = px.line(svc, x="TotalServices", y="customers", color="ChurnLabel", color_discrete_map=color_map, markers=True)
        st.plotly_chart(style(line, "Service count vs churn"), use_container_width=True)

if "CustomerValueSegment" in frame.columns:
    seg = frame.groupby(["CustomerValueSegment", "ChurnLabel"]).size().reset_index(name="customers")
    bar = px.bar(seg, x="CustomerValueSegment", y="customers", color="ChurnLabel", barmode="group", color_discrete_map=color_map)
    st.plotly_chart(style(bar, "Customer value segment vs churn"), use_container_width=True)

st.markdown("### Churn summary")
s1, s2, s3 = st.columns(3)
with s1:
    st.metric("Churned share", f"{kpis['churn_rate'] * 100:.1f}%")
with s2:
    fiber = frame[frame["InternetService"] == "Fiber optic"]
    st.metric("Fiber optic churn", f"{fiber['ChurnFlag'].mean() * 100:.1f}%" if len(fiber) else "—")
with s3:
    m2m = frame[frame["Contract"] == "Month-to-month"]
    st.metric("Month-to-month churn", f"{m2m['ChurnFlag'].mean() * 100:.1f}%" if len(m2m) else "—")

st.markdown("### Key insights")
for insight in key_insights(frame):
    st.markdown(f'<div class="insight">{insight}</div>', unsafe_allow_html=True)

st.markdown("### High-risk customers")
if scored is None or "RiskLevel" not in frame.columns:
    st.info("Train the model to generate churn probabilities and risk levels for every customer.")
else:
    high = (
        frame[frame["RiskLevel"] == "HIGH"]
        .sort_values("ChurnProbability", ascending=False)
        .loc[:, ["customerID", "tenure", "MonthlyCharges", "Contract", "InternetService", "ChurnProbability", "RiskLevel"]]
        .head(50)
    )
    high = high.rename(columns={"customerID": "Customer ID", "ChurnProbability": "Churn Probability", "RiskLevel": "Risk"})
    high["Churn Probability"] = (high["Churn Probability"] * 100).round(1)
    st.dataframe(high, use_container_width=True, hide_index=True)
    st.download_button(
        "Download High-Risk Customers CSV",
        data=high.to_csv(index=False).encode("utf-8"),
        file_name="high_risk_customers.csv",
        mime="text/csv",
    )
