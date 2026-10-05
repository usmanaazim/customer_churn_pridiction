#!/usr/bin/env python3
"""Train churn models, compare PCA vs no PCA, tune, and save artifacts."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.model_selection import train_test_split

from src.config import RANDOM_STATE, TEST_SIZE
from src.data_loader import load_raw_dataset
from src.model_training import train_and_compare
from src.preprocessing import clean_dataset, split_features_target


def main() -> int:
    print("=" * 37)
    print("CUSTOMER CHURN MODEL TRAINING")
    print("=" * 37)
    print()

    df = load_raw_dataset()
    print(f"Dataset loaded: {len(df)} rows")
    cleaned = clean_dataset(df)
    print(f"After cleaning: {len(cleaned)} rows")

    X, y, _ids = split_features_target(cleaned)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )
    print("Feature engineering completed")
    print(f"Training samples: {len(X_train)}  Test samples: {len(X_test)}")
    print()
    print("Training models...")

    result = train_and_compare(X_train, y_train, X_test, y_test)
    comparison = result["comparison"]
    metadata = result["metadata"]
    pca_meta = result["pca_metadata"] or {}

    print()
    print("-" * 37)
    print("TEST RESULTS BY MODEL")
    print("-" * 37)
    for _, row in comparison.iterrows():
        print()
        print(row["label"])
        print(f"Accuracy: {row['accuracy']:.4f}")
        print(f"F1: {row['f1_score']:.4f}")
        print(f"ROC-AUC: {row['roc_auc']:.4f}")

    print()
    print("=" * 37)
    print("BEST MODEL")
    print("=" * 37)
    print()
    print(f"Model: {metadata['model_name']} ({metadata['variant']})")
    print(f"F1: {metadata['f1_score']:.4f}")
    print(f"ROC-AUC: {metadata['roc_auc']:.4f}")
    print(f"Accuracy: {metadata['accuracy']:.4f}")
    print(f"Precision: {metadata['precision']:.4f}")
    print(f"Recall: {metadata['recall']:.4f}")
    print()
    print("PCA:")
    print(f"Original features: {pca_meta.get('original_feature_count', metadata.get('feature_count_before_pca'))}")
    print(f"Components: {pca_meta.get('pca_components', 'N/A (best model may not use PCA)')}")
    print(f"Variance retained: {pca_meta.get('variance_retained', metadata.get('pca_variance_retained'))}")
    print()
    print("Model saved successfully.")
    print("=" * 37)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
