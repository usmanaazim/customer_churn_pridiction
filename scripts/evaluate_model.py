#!/usr/bin/env python3
"""Print evaluation metrics from saved training artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import COMPARISON_PATH, MODEL_METADATA_PATH, PCA_METADATA_PATH
import pandas as pd


def main() -> int:
    if not MODEL_METADATA_PATH.is_file():
        print("Model not trained. Run python3 scripts/train_model.py first.")
        return 1

    metadata = json.loads(MODEL_METADATA_PATH.read_text())
    print("=" * 37)
    print("MODEL EVALUATION")
    print("=" * 37)
    print(f"Best model: {metadata['model_name']} ({metadata.get('variant')})")
    print(f"Trained: {metadata.get('training_date')}")
    print()
    print(f"Accuracy : {metadata['accuracy']:.4f}")
    print(f"Precision: {metadata['precision']:.4f}")
    print(f"Recall   : {metadata['recall']:.4f}")
    print(f"F1 Score : {metadata['f1_score']:.4f}")
    print(f"ROC-AUC  : {metadata['roc_auc']:.4f}")
    print()
    print(f"Training samples: {metadata['training_samples']}")
    print(f"Test samples    : {metadata['test_samples']}")
    print(f"Features before PCA: {metadata['feature_count_before_pca']}")
    print(f"Features after PCA : {metadata['feature_count_after_pca']}")
    print(f"Best parameters: {metadata.get('best_parameters')}")

    if PCA_METADATA_PATH.is_file():
        pca = json.loads(PCA_METADATA_PATH.read_text())
        print()
        print("PCA")
        print(f"  Original encoded features: {pca.get('original_feature_count')}")
        print(f"  Components: {pca.get('pca_components')}")
        print(f"  Variance retained: {pca.get('variance_retained')}")
        print(f"  Dimensionality reduction: {pca.get('dimensionality_reduction_percentage')}%")

    if COMPARISON_PATH.is_file():
        print()
        print("Comparison table")
        print(pd.read_csv(COMPARISON_PATH).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
