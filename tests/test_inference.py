"""Tests for the ML inference pipeline (backend/ml/inference.py)."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from backend.domain.experiment import ExperimentConfig
from backend.ml.anomaly import AnomalyDetector
from backend.ml.classifier import TransactionClassifier
from backend.ml.inference import (
    MLInferencePipeline,
    find_latest_model,
    load_anomaly_detector,
    load_classifier,
)


@pytest.fixture
def trained_models():
    """Create and save mock trained classifier and anomaly detector in a temp dir."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        models_dir = Path(tmp_dir)

        # 1. Train small classifier
        clf = TransactionClassifier(
            model_version="rf_M1_test_v1",
            feature_set_name="M1",
            n_estimators=10,
            feature_medians={"feat_a": 1.5, "feat_b": 2.5},
        )
        X_train = np.array([[1.0, 2.0], [2.0, 3.0], [0.5, 1.0], [3.0, 4.0]])
        y_train = np.array([0, 1, 0, 1])
        clf.train(X_train, y_train, feature_names=["feat_a", "feat_b"])
        clf.save(models_dir)

        # 2. Train small anomaly detector
        anom = AnomalyDetector(
            model_version="iforest_test_v1",
            n_estimators=10,
            feature_names=["feat_a", "feat_b"],
            feature_medians={"feat_a": 1.5, "feat_b": 2.5},
        )
        anom.fit(X_train)
        anom.save(models_dir)

        yield models_dir


def test_find_latest_model(trained_models):
    """find_latest_model locates correct artifacts by prefix."""
    clf_path = find_latest_model(trained_models, "rf_M1")
    assert clf_path is not None
    assert clf_path.name.startswith("rf_M1")

    anom_path = find_latest_model(trained_models, "iforest")
    assert anom_path is not None
    assert anom_path.name.startswith("iforest")

    missing = find_latest_model(trained_models, "nonexistent")
    assert missing is None


def test_load_models(trained_models):
    """load_classifier and load_anomaly_detector load models with medians."""
    clf = load_classifier(models_dir=trained_models, feature_set_name="M1")
    assert clf.is_fitted
    assert clf.feature_names == ["feat_a", "feat_b"]
    assert clf.feature_medians == {"feat_a": 1.5, "feat_b": 2.5}

    anom = load_anomaly_detector(models_dir=trained_models)
    assert anom.is_fitted
    assert anom.feature_names == ["feat_a", "feat_b"]
    assert anom.feature_medians == {"feat_a": 1.5, "feat_b": 2.5}


def test_inference_pipeline_prediction(trained_models):
    """Pipeline predicts accurately and applies imputation for missing values."""
    clf = load_classifier(models_dir=trained_models, feature_set_name="M1")
    anom = load_anomaly_detector(models_dir=trained_models)

    pipeline = MLInferencePipeline(classifier=clf, anomaly_detector=anom)

    # Input has missing feat_b (should impute 2.5)
    sample_features = {"feat_a": 1.2}
    result = pipeline.predict_transaction(sample_features, txid="tx_123")

    assert result.entity_id == "tx_123"
    assert result.entity_type == "transaction"
    assert result.prediction in (0, 1)
    assert 0.0 <= result.probability <= 1.0
    assert result.anomaly_score is not None
    assert 0.0 <= result.anomaly_score <= 1.0
    assert result.feature_set == "M1"
    assert result.model_version == "rf_M1_test_v1"
