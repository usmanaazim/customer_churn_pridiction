"""Feature engineering documentation and PCA visualizations."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.config import ENGINEERED_FEATURE_DESCRIPTIONS, ORIGINAL_PREDICTORS
from src.ui import (
    PLOTLY_LAYOUT,
    get_scored_data,
    hero,
    inject_css,
    kpi_card,
    load_feature_metadata,
    load_metadata,
    load_pca_metadata,
    require_dataset,
    require_pipeline,
)

st.set_page_config(page_title="Feature Engineering & PCA", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
inject_css()
hero("Feature Engineering and PCA", "How original billing and service fields become a compact, predictive representation")

st.markdown(
    "PCA transforms the original feature space into a smaller number of principal components "
    "while retaining most of the information represented by the original variables."
)

feat_meta = load_feature_metadata()
pca_meta = load_pca_metadata()
model_meta = load_metadata()

if pca_meta:
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        kpi_card("Original encoded features", str(pca_meta.get("original_feature_count", "—")))
    with k2:
        kpi_card("PCA components", str(pca_meta.get("pca_components", "—")))
    with k3:
        kpi_card("Variance retained", f"{float(pca_meta.get('variance_retained', 0)) * 100:.1f}%")
    with k4:
        kpi_card("Dimensionality reduction", f"{pca_meta.get('dimensionality_reduction_percentage', 0)}%")
else:
    st.warning("PCA metadata not found. Run python3 scripts/train_model.py first.")

st.markdown("### Original features → engineered features")
left, mid, right = st.columns([1, 0.2, 1])
with left:
    st.markdown("**Original**")
    st.markdown("\n".join(f"- `{c}`" for c in ORIGINAL_PREDICTORS[:8]))
    with st.expander("All original predictors"):
        st.write(ORIGINAL_PREDICTORS)
with mid:
    st.markdown("### ↓")
with right:
    st.markdown("**Engineered**")
    st.markdown("\n".join(f"- `{c}`" for c in ENGINEERED_FEATURE_DESCRIPTIONS))

rows = [{"Feature": k, "Description": v, "Type": "Engineered"} for k, v in ENGINEERED_FEATURE_DESCRIPTIONS.items()]
st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

if feat_meta:
    with st.expander("Feature metadata from training"):
        st.json(feat_meta)

if pca_meta and pca_meta.get("explained_variance_ratio"):
    ratio = pca_meta["explained_variance_ratio"]
    cumulative = pca_meta.get("cumulative_explained_variance") or list(np.cumsum(ratio))
    idx = list(range(1, len(ratio) + 1))
    c1, c2 = st.columns(2)
    fig = px.bar(x=idx, y=ratio, labels={"x": "Principal component", "y": "Explained variance ratio"}, title="Explained variance ratio")
    fig.update_layout(**PLOTLY_LAYOUT)
    c1.plotly_chart(fig, use_container_width=True)
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=idx, y=cumulative, mode="lines+markers", name="Cumulative"))
    fig2.add_hline(y=0.95, line_dash="dash", annotation_text="95%")
    fig2.update_layout(**PLOTLY_LAYOUT, title="Cumulative explained variance", xaxis_title="Component", yaxis_title="Cumulative variance")
    c2.plotly_chart(fig2, use_container_width=True)

    contrib = pd.DataFrame({"component": [f"PC{i}" for i in idx], "explained_variance": ratio})
    fig3 = px.line(contrib, x="component", y="explained_variance", markers=True, title="PCA component contribution")
    fig3.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig3, use_container_width=True)

pipeline = require_pipeline()
df = require_dataset()
if pipeline is not None and df is not None and "pca" in pipeline.named_steps:
    st.markdown("### PC1 vs PC2")
    sample = df.sample(min(len(df), 1500), random_state=42)
    X = sample.drop(columns=["customerID", "Churn"], errors="ignore")
    feats = pipeline.named_steps["features"].transform(X)
    encoded = pipeline.named_steps["encode_scale"].transform(feats)
    pcs = pipeline.named_steps["pca"].transform(encoded)
    if hasattr(pcs, "iloc"):
        pc1 = pcs.iloc[:, 0]
        pc2 = pcs.iloc[:, 1] if pcs.shape[1] > 1 else pcs.iloc[:, 0]
    else:
        pc1 = pcs[:, 0]
        pc2 = pcs[:, 1] if pcs.shape[1] > 1 else pcs[:, 0]
    plot_df = pd.DataFrame({"PC1": np.asarray(pc1), "PC2": np.asarray(pc2), "Churn": sample["Churn"].values})
    fig = px.scatter(plot_df, x="PC1", y="PC2", color="Churn", color_discrete_map={"Yes": "#C0392B", "No": "#1F8A4C"}, opacity=0.6, title="Customers in the first two principal components")
    fig.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)
elif pipeline is not None and "pca" not in pipeline.named_steps:
    st.info("The selected best model does not include PCA. The charts above still use the PCA transformer fitted during comparison training.")
    scored = get_scored_data()
    _ = scored

st.markdown("### Feature importance (non-PCA models)")
st.caption(
    "PCA-based models operate on transformed principal components, so original feature "
    "interpretation is shown separately using the non-PCA model."
)
if model_meta and model_meta.get("non_pca_feature_importance", {}).get("available"):
    imp = model_meta["non_pca_feature_importance"]
    tabs = st.tabs(["Random Forest", "Logistic Regression"])
    with tabs[0]:
        data = pd.DataFrame(imp.get("random_forest") or [])
        if data.empty:
            st.write("Not available.")
        else:
            fig = px.bar(data, x="importance", y="feature", orientation="h", title="Random Forest importance (before PCA)")
            fig.update_layout(**PLOTLY_LAYOUT, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
    with tabs[1]:
        data = pd.DataFrame(imp.get("logistic_regression") or [])
        if data.empty:
            st.write("Not available.")
        else:
            fig = px.bar(data, x="importance", y="feature", orientation="h", title="Logistic Regression |coefficient| (before PCA)")
            fig.update_layout(**PLOTLY_LAYOUT, yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Train models to populate feature importance from the non-PCA Random Forest and Logistic Regression estimators.")
