import json

import joblib
import pytest

from src.config import (
    BEST_MODEL_PATH,
    FEATURE_METADATA_PATH,
    FULL_PIPELINE_PATH,
    MODEL_METADATA_PATH,
    PCA_METADATA_PATH,
    PREPROCESSING_PATH,
)


@pytest.mark.skipif(not FULL_PIPELINE_PATH.is_file(), reason="Model artifacts not trained yet")
def test_artifacts_load():
    pipeline = joblib.load(FULL_PIPELINE_PATH)
    model = joblib.load(BEST_MODEL_PATH)
    pre = joblib.load(PREPROCESSING_PATH)
    assert hasattr(pipeline, "predict")
    assert hasattr(model, "predict")
    assert hasattr(pre, "transform")
    assert "features" in pipeline.named_steps
    assert "model" in pipeline.named_steps


@pytest.mark.skipif(not MODEL_METADATA_PATH.is_file(), reason="Metadata not trained yet")
def test_metadata_has_required_fields():
    meta = json.loads(MODEL_METADATA_PATH.read_text())
    for key in [
        "model_name",
        "training_date",
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc",
        "best_parameters",
        "training_samples",
        "test_samples",
        "feature_count_before_pca",
        "feature_count_after_pca",
    ]:
        assert key in meta
    assert 0 <= meta["accuracy"] <= 1
    assert 0 <= meta["f1_score"] <= 1
    assert 0 <= meta["roc_auc"] <= 1


@pytest.mark.skipif(not PCA_METADATA_PATH.is_file(), reason="PCA metadata not trained yet")
def test_pca_metadata_is_numeric():
    pca = json.loads(PCA_METADATA_PATH.read_text())
    assert pca.get("original_feature_count", 0) > 0
    assert pca.get("pca_components", 0) > 0
    assert 0 < float(pca.get("variance_retained", 0)) <= 1.0001


@pytest.mark.skipif(not FEATURE_METADATA_PATH.is_file(), reason="Feature metadata not trained yet")
def test_feature_metadata_lists():
    feat = json.loads(FEATURE_METADATA_PATH.read_text())
    assert "AverageMonthlySpend" in feat["engineered_features"]
    assert "customerID" in feat["dropped_features"]
    assert feat["target_feature"] == "Churn"
