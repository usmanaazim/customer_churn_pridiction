"""Shared Streamlit loading, messaging, and visual helpers."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

from src.analytics import prepare_analytics_frame, score_customers
from src.config import (
    COMPARISON_PATH,
    DATA_PATH,
    EVALUATION_CURVES_PATH,
    FEATURE_METADATA_PATH,
    FULL_PIPELINE_PATH,
    MODEL_METADATA_PATH,
    PCA_METADATA_PATH,
)
from src.data_loader import DatasetError, dataset_exists, load_raw_dataset
from src.prediction import PredictionError, load_full_pipeline
from src.preprocessing import clean_dataset

COLORS = {
    "navy": "#0F2C59",
    "blue": "#1B4F8A",
    "accent": "#3E7CB1",
    "bg": "#F4F7FB",
    "card": "#FFFFFF",
    "low": "#1F8A4C",
    "medium": "#D97706",
    "high": "#C0392B",
    "muted": "#5C6B7A",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, Segoe UI, sans-serif", color="#1F2933"),
    margin=dict(l=16, r=16, t=48, b=16),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
)


def inject_css() -> None:
    st.markdown(
        """
        <style>
        .block-container {padding-top: 1.4rem; padding-bottom: 2rem; max-width: 1400px;}
        .cci-hero {background: #0F2C59; color: #fff; border-radius: 16px; padding: 1.4rem 1.6rem; margin-bottom: 1.1rem;}
        .cci-hero h1 {color: #fff !important; font-size: 1.85rem; margin: 0 0 .25rem 0;}
        .cci-hero p {color: #D6E4F0; margin: 0; font-size: 1.02rem;}
        .kpi-card {background: #fff; border: 1px solid #E3EAF2; border-radius: 14px; padding: 0.95rem 1rem; box-shadow: 0 1px 2px rgba(15,44,89,.04);}
        .kpi-label {font-size: .78rem; color: #5C6B7A; letter-spacing: .04em; text-transform: uppercase; font-weight: 600;}
        .kpi-value {font-size: 1.55rem; color: #0F2C59; font-weight: 700; margin-top: .2rem;}
        .risk-low {background:#E8F7EE; color:#1F8A4C; padding:.2rem .55rem; border-radius:999px; font-weight:700; font-size:.8rem;}
        .risk-medium {background:#FEF3C7; color:#B45309; padding:.2rem .55rem; border-radius:999px; font-weight:700; font-size:.8rem;}
        .risk-high {background:#FDE8E6; color:#C0392B; padding:.2rem .55rem; border-radius:999px; font-weight:700; font-size:.8rem;}
        .insight {background:#fff; border-left:4px solid #3E7CB1; padding:.7rem .9rem; margin-bottom:.55rem; border-radius: 0 10px 10px 0;}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str) -> None:
    st.markdown(
        f'<div class="cci-hero"><h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str) -> None:
    st.markdown(
        f'<div class="kpi-card"><div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


def risk_badge(level: str) -> str:
    cls = {"LOW": "risk-low", "MEDIUM": "risk-medium", "HIGH": "risk-high"}.get(level, "risk-medium")
    return f'<span class="{cls}">{level}</span>'


def friendly_error(message: str) -> None:
    st.warning(message)


@st.cache_data(show_spinner=False)
def load_clean_data() -> pd.DataFrame:
    raw = load_raw_dataset()
    return clean_dataset(raw)


@st.cache_resource(show_spinner=False)
def load_pipeline():
    return load_full_pipeline()


@st.cache_data(show_spinner=False)
def load_json(path_str: str, mtime: float) -> dict:
    return json.loads(Path(path_str).read_text())


@st.cache_data(show_spinner=False)
def load_comparison(path_str: str, mtime: float) -> pd.DataFrame:
    return pd.read_csv(path_str)


def artifact_mtime(path: Path) -> float:
    return path.stat().st_mtime if path.is_file() else 0.0


def load_metadata() -> dict | None:
    if not MODEL_METADATA_PATH.is_file():
        return None
    return load_json(str(MODEL_METADATA_PATH), artifact_mtime(MODEL_METADATA_PATH))


def load_pca_metadata() -> dict | None:
    if not PCA_METADATA_PATH.is_file():
        return None
    return load_json(str(PCA_METADATA_PATH), artifact_mtime(PCA_METADATA_PATH))


def load_feature_metadata() -> dict | None:
    if not FEATURE_METADATA_PATH.is_file():
        return None
    return load_json(str(FEATURE_METADATA_PATH), artifact_mtime(FEATURE_METADATA_PATH))


def load_model_comparison() -> pd.DataFrame | None:
    if not COMPARISON_PATH.is_file():
        return None
    return load_comparison(str(COMPARISON_PATH), artifact_mtime(COMPARISON_PATH))


@st.cache_resource(show_spinner=False)
def load_curves():
    if not EVALUATION_CURVES_PATH.is_file():
        return None
    import joblib

    return joblib.load(EVALUATION_CURVES_PATH)


def require_dataset() -> pd.DataFrame | None:
    if not dataset_exists():
        friendly_error(
            "The Telco Customer Churn dataset was not found at data/customer_churn.csv. "
            "See data/README.md or run: python3 scripts/setup_data.py"
        )
        return None
    try:
        return load_clean_data()
    except DatasetError as exc:
        friendly_error(str(exc))
        return None
    except Exception:
        friendly_error("The dataset could not be loaded. Please verify data/customer_churn.csv.")
        return None


def require_pipeline():
    if not FULL_PIPELINE_PATH.is_file():
        friendly_error("Model not trained. Run python3 scripts/train_model.py first.")
        return None
    try:
        return load_pipeline()
    except PredictionError as exc:
        friendly_error(str(exc))
        return None
    except Exception:
        friendly_error("The saved model appears to be corrupt. Retrain with python3 scripts/train_model.py.")
        return None


@st.cache_data(show_spinner="Scoring customers with the trained model...")
def load_scored_frame(_model_mtime: float, _data_mtime: float) -> pd.DataFrame:
    df = load_clean_data()
    pipeline = load_full_pipeline()
    analytics = prepare_analytics_frame(df, pipeline.named_steps["features"])
    return score_customers(analytics, pipeline)


def get_scored_data() -> pd.DataFrame | None:
    if not DATA_PATH.is_file() or not FULL_PIPELINE_PATH.is_file():
        return None
    try:
        return load_scored_frame(artifact_mtime(FULL_PIPELINE_PATH), artifact_mtime(DATA_PATH))
    except Exception:
        friendly_error("Customer scoring failed. Retrain the model or verify the dataset.")
        return None
