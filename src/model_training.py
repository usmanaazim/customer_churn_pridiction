"""Model training, PCA pipelines, comparison, and artifact saving."""

from __future__ import annotations

import json
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn import set_config
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

from src.config import (
    BEST_MODEL_PATH,
    CATEGORICAL_FEATURES,
    COMPARISON_PATH,
    CV_FOLDS,
    DROPPED_FEATURES,
    ENGINEERED_FEATURES,
    EVALUATION_CURVES_PATH,
    FEATURE_METADATA_PATH,
    FULL_PIPELINE_PATH,
    MODEL_METADATA_PATH,
    MODELS_DIR,
    NUMERIC_FEATURES,
    ORIGINAL_PREDICTORS,
    PCA_METADATA_PATH,
    PCA_VARIANCE,
    PREPROCESSING_PATH,
    RANDOM_STATE,
    SVM_CV_FOLDS,
    TARGET_COLUMN,
)
from src.feature_engineering import FeatureEngineer
from src.model_evaluation import classification_metrics, curve_payload, select_best_row

set_config(transform_output="pandas")


def build_column_transformer() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_preprocessor_only() -> Pipeline:
    return Pipeline(
        steps=[
            ("features", FeatureEngineer()),
            ("encode_scale", build_column_transformer()),
        ]
    )


def build_model_pipeline(estimator, use_pca: bool) -> Pipeline:
    steps = [
        ("features", FeatureEngineer()),
        ("encode_scale", build_column_transformer()),
    ]
    if use_pca:
        steps.append(("pca", PCA(n_components=PCA_VARIANCE, random_state=RANDOM_STATE)))
    steps.append(("model", estimator))
    return Pipeline(steps)


def default_estimators() -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "SVM": SVC(
            kernel="rbf",
            probability=True,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def search_spaces() -> dict:
    return {
        "Logistic Regression": {
            "model__C": [0.1, 1.0, 10.0],
        },
        "Random Forest": {
            "model__n_estimators": [150, 250],
            "model__max_depth": [10, 20, None],
            "model__min_samples_split": [2, 5],
        },
        "SVM": {
            "model__C": [0.5, 1.0, 5.0],
            "model__gamma": ["scale", 0.05],
            "model__kernel": ["rbf"],
        },
        "Gradient Boosting": {
            "model__n_estimators": [100, 150],
            "model__learning_rate": [0.05, 0.1],
            "model__max_depth": [2, 3],
        },
    }


def _proba_and_pred(pipeline: Pipeline, X) -> tuple[np.ndarray, np.ndarray]:
    if hasattr(pipeline, "predict_proba"):
        proba = pipeline.predict_proba(X)[:, 1]
    else:
        scores = pipeline.decision_function(X)
        proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
    pred = (proba >= 0.5).astype(int)
    model_pred = pipeline.predict(X)
    if model_pred is not None:
        pred = np.asarray(model_pred).astype(int)
    return pred, proba


def evaluate_pipeline(pipeline: Pipeline, X, y) -> dict:
    pred, proba = _proba_and_pred(pipeline, X)
    return classification_metrics(y, pred, proba)


def pca_details_from_pipeline(pipeline: Pipeline) -> dict | None:
    if "pca" not in pipeline.named_steps:
        return None
    pca: PCA = pipeline.named_steps["pca"]
    ratio = pca.explained_variance_ratio_.tolist()
    cumulative = np.cumsum(pca.explained_variance_ratio_).tolist()
    original = int(pca.n_features_in_)
    components = int(pca.n_components_)
    retained = float(np.sum(pca.explained_variance_ratio_))
    reduction = float((1 - components / original) * 100) if original else 0.0
    return {
        "original_feature_count": original,
        "pca_components": components,
        "variance_retained": round(retained, 6),
        "dimensionality_reduction_percentage": round(reduction, 2),
        "explained_variance_ratio": ratio,
        "cumulative_explained_variance": cumulative,
        "n_components_setting": PCA_VARIANCE,
    }


def encoded_width(fitted_pipeline: Pipeline, X_sample: pd.DataFrame) -> int:
    features = fitted_pipeline.named_steps["features"].transform(X_sample.iloc[:1])
    encoded = fitted_pipeline.named_steps["encode_scale"].transform(features)
    return int(encoded.shape[1])


def tune_pipeline(base_pipeline: Pipeline, name: str, X_train, y_train, use_pca: bool) -> Pipeline:
    space = search_spaces()[name]
    cv_folds = SVM_CV_FOLDS if name == "SVM" else CV_FOLDS
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)
    if name == "SVM":
        search = RandomizedSearchCV(
            base_pipeline,
            space,
            n_iter=4,
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            refit=True,
        )
    else:
        search = GridSearchCV(
            base_pipeline,
            space,
            scoring="f1",
            cv=cv,
            n_jobs=-1,
            refit=True,
        )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, float(search.best_score_)


def train_and_compare(X_train, y_train, X_test, y_test) -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    estimators = default_estimators()
    rows = []
    fitted = {}

    print("\nTraining baseline models (without PCA) and PCA models...\n")
    for name, estimator in estimators.items():
        for use_pca in (False, True):
            label = f"{name} | {'PCA' if use_pca else 'No PCA'}"
            print(f"Fitting {label} ...")
            pipeline = build_model_pipeline(estimator, use_pca=use_pca)
            pipeline.fit(X_train, y_train)
            metrics = evaluate_pipeline(pipeline, X_test, y_test)
            row = {
                "model_name": name,
                "variant": "PCA" if use_pca else "No PCA",
                "label": label,
                **metrics,
                "tuned": False,
            }
            rows.append(row)
            fitted[label] = pipeline
            print(
                f"  Accuracy: {metrics['accuracy']:.4f}  "
                f"Precision: {metrics['precision']:.4f}  "
                f"Recall: {metrics['recall']:.4f}  "
                f"F1: {metrics['f1_score']:.4f}  "
                f"ROC-AUC: {metrics['roc_auc']:.4f}"
            )

    comparison = pd.DataFrame(rows)
    print("\nHyperparameter tuning on each algorithm (No PCA and PCA, compact grids)...\n")
    tuned_fitted = {}
    tuned_params = {}
    for name in estimators:
        for use_pca in (False, True):
            label = f"{name} | {'PCA' if use_pca else 'No PCA'} | Tuned"
            print(f"Tuning {label} ...")
            pipeline = build_model_pipeline(default_estimators()[name], use_pca=use_pca)
            best_est, params, cv_f1 = tune_pipeline(pipeline, name, X_train, y_train, use_pca)
            metrics = evaluate_pipeline(best_est, X_test, y_test)
            row = {
                "model_name": name,
                "variant": ("PCA" if use_pca else "No PCA") + " Tuned",
                "label": label,
                **metrics,
                "tuned": True,
                "cv_f1": cv_f1,
            }
            rows.append(row)
            tuned_fitted[label] = best_est
            tuned_params[label] = params
            print(
                f"  CV F1: {cv_f1:.4f}  Test F1: {metrics['f1_score']:.4f}  "
                f"ROC-AUC: {metrics['roc_auc']:.4f}  params: {params}"
            )

    comparison = pd.DataFrame(rows)
    best_row = select_best_row(comparison)
    best_label = best_row["label"]
    if best_label in tuned_fitted:
        best_pipeline = tuned_fitted[best_label]
        best_parameters = tuned_params.get(best_label, {})
    else:
        best_pipeline = fitted[best_label]
        best_parameters = {}

    pred, proba = _proba_and_pred(best_pipeline, X_test)
    curves = curve_payload(y_test, pred, proba)
    pca_meta = pca_details_from_pipeline(best_pipeline)
    pca_reference = pca_meta
    if pca_reference is None:
        pca_candidate = next(
            (pipe for pipe in list(fitted.values()) + list(tuned_fitted.values()) if "pca" in pipe.named_steps),
            None,
        )
        pca_reference = pca_details_from_pipeline(pca_candidate) if pca_candidate is not None else {}

    feature_count_before = encoded_width(best_pipeline, X_train)
    feature_count_after = (
        pca_meta["pca_components"] if pca_meta else feature_count_before
    )

    metadata = {
        "model_name": best_row["model_name"],
        "variant": best_row["variant"],
        "label": best_label,
        "uses_pca": pca_meta is not None,
        "training_date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "accuracy": float(best_row["accuracy"]),
        "precision": float(best_row["precision"]),
        "recall": float(best_row["recall"]),
        "f1_score": float(best_row["f1_score"]),
        "roc_auc": float(best_row["roc_auc"]),
        "best_parameters": _json_ready(best_parameters),
        "training_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "feature_count_before_pca": feature_count_before,
        "feature_count_after_pca": int(feature_count_after),
        "pca_variance_retained": float(pca_meta["variance_retained"]) if pca_meta else None,
        "random_state": RANDOM_STATE,
        "test_size": 0.2,
        "selection_rule": "Highest test F1, then highest ROC-AUC",
    }

    feature_metadata = {
        "original_features": ORIGINAL_PREDICTORS,
        "engineered_features": ENGINEERED_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "dropped_features": DROPPED_FEATURES,
        "target_feature": TARGET_COLUMN,
        "training_thresholds": {
            "monthly_median": best_pipeline.named_steps["features"].monthly_median_,
            "lifetime_value_q33": best_pipeline.named_steps["features"].eltv_q33_,
            "lifetime_value_q66": best_pipeline.named_steps["features"].eltv_q66_,
        },
    }

    importance = extract_non_pca_importance(fitted, X_train)
    metadata["explainability_note"] = (
        "PCA-based models operate on transformed principal components, so original "
        "feature interpretation is shown separately using the non-PCA model."
    )
    metadata["non_pca_feature_importance"] = importance

    save_artifacts(
        best_pipeline=best_pipeline,
        comparison=comparison,
        metadata=metadata,
        pca_metadata=pca_reference,
        feature_metadata=feature_metadata,
        curves=curves,
    )
    return {
        "best_pipeline": best_pipeline,
        "comparison": comparison,
        "metadata": metadata,
        "pca_metadata": pca_reference,
        "curves": curves,
    }


def extract_non_pca_importance(fitted: dict, X_train: pd.DataFrame) -> dict:
    rf_pipe = fitted.get("Random Forest | No PCA")
    lr_pipe = fitted.get("Logistic Regression | No PCA")
    payload = {"available": False}
    if rf_pipe is None and lr_pipe is None:
        return payload
    try:
        names = _encoded_feature_names(rf_pipe or lr_pipe)
    except Exception:
        names = []
    if rf_pipe is not None and names:
        importances = rf_pipe.named_steps["model"].feature_importances_
        payload["random_forest"] = _top_pairs(names, importances, 15)
        payload["available"] = True
    if lr_pipe is not None and names:
        coefs = np.abs(lr_pipe.named_steps["model"].coef_[0])
        payload["logistic_regression"] = _top_pairs(names, coefs, 15)
        payload["available"] = True
    return payload


def _encoded_feature_names(pipeline: Pipeline) -> list[str]:
    transformer: ColumnTransformer = pipeline.named_steps["encode_scale"]
    return transformer.get_feature_names_out().tolist()


def _top_pairs(names, values, k: int) -> list[dict]:
    order = np.argsort(values)[::-1][:k]
    return [{"feature": names[i], "importance": float(values[i])} for i in order]


def _json_ready(obj):
    if isinstance(obj, dict):
        return {str(k).replace("model__", ""): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if obj is None:
        return None
    return obj


def save_artifacts(best_pipeline, comparison, metadata, pca_metadata, feature_metadata, curves):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, FULL_PIPELINE_PATH)
    joblib.dump(best_pipeline.named_steps["model"], BEST_MODEL_PATH)
    preprocessing = Pipeline(
        [
            ("features", best_pipeline.named_steps["features"]),
            ("encode_scale", best_pipeline.named_steps["encode_scale"]),
        ]
    )
    if "pca" in best_pipeline.named_steps:
        preprocessing.steps.append(("pca", best_pipeline.named_steps["pca"]))
    joblib.dump(preprocessing, PREPROCESSING_PATH)
    joblib.dump(curves, EVALUATION_CURVES_PATH)
    comparison.to_csv(COMPARISON_PATH, index=False)
    MODEL_METADATA_PATH.write_text(json.dumps(metadata, indent=2))
    PCA_METADATA_PATH.write_text(json.dumps(pca_metadata, indent=2))
    FEATURE_METADATA_PATH.write_text(json.dumps(feature_metadata, indent=2))
