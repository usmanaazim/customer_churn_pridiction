"""Model comparison, PCA vs no-PCA, and evaluation curves."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from src.ui import (
    PLOTLY_LAYOUT,
    hero,
    inject_css,
    kpi_card,
    load_curves,
    load_metadata,
    load_model_comparison,
    load_pca_metadata,
)

st.set_page_config(page_title="Model Performance", page_icon="📊", layout="wide", initial_sidebar_state="expanded")
inject_css()
hero("Model Performance", "Test-set metrics, PCA comparison, and the selected production model")

meta = load_metadata()
comparison = load_model_comparison()
curves = load_curves()
pca_meta = load_pca_metadata()

if meta is None or comparison is None:
    st.warning("Model not trained. Run python3 scripts/train_model.py first.")
    st.stop()

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    kpi_card("Accuracy", f"{meta['accuracy']:.3f}")
with k2:
    kpi_card("Precision", f"{meta['precision']:.3f}")
with k3:
    kpi_card("Recall", f"{meta['recall']:.3f}")
with k4:
    kpi_card("F1 Score", f"{meta['f1_score']:.3f}")
with k5:
    kpi_card("ROC-AUC", f"{meta['roc_auc']:.3f}")

st.markdown(f"### Best model: **{meta['model_name']}** ({meta.get('variant')})")
st.caption(meta.get("selection_rule", "Selected using F1 then ROC-AUC on the held-out test set."))
st.write(
    f"Training samples: {meta.get('training_samples')} · Test samples: {meta.get('test_samples')} · "
    f"Encoded features before PCA: {meta.get('feature_count_before_pca')} · "
    f"After PCA: {meta.get('feature_count_after_pca')}"
)
if meta.get("best_parameters"):
    st.json(meta["best_parameters"])

st.markdown("### Model comparison")
core = comparison.copy()
core["model_short"] = core["model_name"] + " · " + core["variant"]
metric_cols = ["accuracy", "precision", "recall", "f1_score", "roc_auc"]
long = core.melt(id_vars=["model_short"], value_vars=metric_cols, var_name="metric", value_name="score")
fig = px.bar(long, x="model_short", y="score", color="metric", barmode="group", title="Accuracy, precision, recall, F1, ROC-AUC")
fig.update_layout(**PLOTLY_LAYOUT, xaxis_tickangle=-25)
st.plotly_chart(fig, use_container_width=True)

st.markdown("### PCA vs no PCA")
pca_cmp = comparison.copy()
pca_cmp["pca_flag"] = pca_cmp["variant"].apply(lambda v: "PCA" if "PCA" in str(v) and "No PCA" not in str(v) else "No PCA")
# Fix: "PCA Tuned" contains PCA and not "No PCA". "No PCA Tuned" contains "No PCA".
grouped = pca_cmp.groupby(["model_name", "pca_flag"])["f1_score"].max().reset_index()
fig = px.bar(grouped, x="model_name", y="f1_score", color="pca_flag", barmode="group", title="Best F1 by algorithm: PCA vs no PCA")
fig.update_layout(**PLOTLY_LAYOUT)
st.plotly_chart(fig, use_container_width=True)

if pca_meta:
    st.caption(
        f"PCA retained {float(pca_meta.get('variance_retained', 0))*100:.1f}% variance using "
        f"{pca_meta.get('pca_components')} components from {pca_meta.get('original_feature_count')} encoded features "
        f"({pca_meta.get('dimensionality_reduction_percentage')}% reduction)."
    )

if curves:
    cm = curves.get("confusion_matrix")
    roc = curves.get("roc", {})
    pr = curves.get("pr", {})
    c1, c2 = st.columns(2)
    if cm:
        z = cm
        fig = go.Figure(
            data=go.Heatmap(
                z=z,
                x=["Pred Stay", "Pred Churn"],
                y=["Actual Stay", "Actual Churn"],
                colorscale="Blues",
                text=z,
                texttemplate="%{text}",
            )
        )
        fig.update_layout(**PLOTLY_LAYOUT, title="Confusion matrix")
        c1.plotly_chart(fig, use_container_width=True)
    if roc:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=roc["fpr"], y=roc["tpr"], mode="lines", name="ROC"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(dash="dash")))
        fig.update_layout(**PLOTLY_LAYOUT, title="ROC curve", xaxis_title="False positive rate", yaxis_title="True positive rate")
        c2.plotly_chart(fig, use_container_width=True)
    if pr:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=pr["recall"], y=pr["precision"], mode="lines", name="PR"))
        fig.update_layout(**PLOTLY_LAYOUT, title="Precision-recall curve", xaxis_title="Recall", yaxis_title="Precision")
        st.plotly_chart(fig, use_container_width=True)

st.dataframe(comparison.drop(columns=["label"], errors="ignore"), use_container_width=True, hide_index=True)
st.download_button(
    "Download model comparison CSV",
    data=comparison.to_csv(index=False).encode("utf-8"),
    file_name="model_comparison.csv",
    mime="text/csv",
)
