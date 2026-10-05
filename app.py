"""Customer Churn Intelligence — Streamlit entry page."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from src.analytics import kpi_summary, key_insights
from src.config import DATA_PATH, FULL_PIPELINE_PATH
from src.ui import (
    PLOTLY_LAYOUT,
    get_scored_data,
    hero,
    inject_css,
    kpi_card,
    load_metadata,
    require_dataset,
)

st.set_page_config(
    page_title="Customer Churn Intelligence",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

inject_css()
hero("Customer Churn Intelligence", "AI-powered customer retention and churn analytics")

st.sidebar.markdown("**Navigate** using the pages list above.")
st.sidebar.caption("Dataset, feature engineering, PCA, and trained models power every view.")

df = require_dataset()
if df is None:
    st.stop()

scored = get_scored_data()
frame = scored if scored is not None else df
if "ChurnFlag" not in frame.columns:
    from src.analytics import prepare_analytics_frame

    frame = prepare_analytics_frame(df)

kpis = kpi_summary(frame)
c1, c2, c3, c4 = st.columns(4)
with c1:
    kpi_card("Total Customers", f"{kpis['total_customers']:,}")
with c2:
    kpi_card("Churned Customers", f"{kpis['churned_customers']:,}")
with c3:
    kpi_card("Retained Customers", f"{kpis['retained_customers']:,}")
with c4:
    kpi_card("Overall Churn Rate", f"{kpis['churn_rate'] * 100:.1f}%")

c5, c6, c7, c8 = st.columns(4)
with c5:
    high_risk = kpis["high_risk_customers"]
    kpi_card("High-Risk Customers", "—" if high_risk is None else f"{high_risk:,}")
with c6:
    kpi_card("Avg Monthly Charges", f"${kpis['avg_monthly_charges']:.2f}")
with c7:
    kpi_card("Average Tenure", f"{kpis['avg_tenure']:.1f} mo")
with c8:
    kpi_card("Avg Lifetime Value", f"${kpis['avg_lifetime_value']:.0f}")

st.markdown("### Project pipeline")
st.markdown(
    """
Customer Dataset → Data Cleaning → EDA → **Feature Engineering** → Encoding → Scaling
→ **PCA** → Machine Learning → Evaluation → Best Model → Churn Probability
→ Risk Classification → Retention Recommendation
"""
)

left, right = st.columns((1.2, 1))
with left:
    churn_counts = frame.groupby(frame["ChurnFlag"].map({0: "Stayed", 1: "Churned"})).size().reset_index(name="customers")
    fig = px.pie(
        churn_counts,
        names=churn_counts.columns[0],
        values="customers",
        hole=0.55,
        color=churn_counts.columns[0],
        color_discrete_map={"Stayed": "#1F8A4C", "Churned": "#C0392B"},
        title="Churn vs retained",
    )
    fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.markdown("#### Status")
    st.write(f"Dataset file: `{DATA_PATH.name}`")
    if FULL_PIPELINE_PATH.is_file():
        meta = load_metadata() or {}
        st.success("Trained model is available.")
        st.write(f"Best model: **{meta.get('model_name', '—')}** ({meta.get('variant', '')})")
        st.write(f"Test F1: **{meta.get('f1_score', 0):.3f}** · ROC-AUC: **{meta.get('roc_auc', 0):.3f}**")
    else:
        st.warning("Model not trained. Run python3 scripts/train_model.py first.")
    st.markdown("#### Open next")
    st.page_link("pages/1_Dashboard.py", label="Full analytics dashboard")
    st.page_link("pages/2_Churn_Prediction.py", label="Predict a customer")
    st.page_link("pages/4_Feature_Engineering_PCA.py", label="Feature engineering & PCA")

st.markdown("### Key insights")
for insight in key_insights(frame):
    st.markdown(f'<div class="insight">{insight}</div>', unsafe_allow_html=True)
