"""Tests for temporal evaluation, anomaly detection, and graph feature evaluation.

Phase 5, Steps 9–11. Uses synthetic in-memory data.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from backend.graph.builder import ChainTraceGraph
from backend.domain.graph import TxTxEdge, AddrTxEdge, TxAddrEdge
from backend.ml.anomaly import AnomalyDetector
from backend.ml.classifier import TransactionClassifier
from backend.ml.evaluation import (
    TemporalEvaluation,
    evaluate_temporal,
    summarize_evaluation,
)
from backend.ml.feature_registry import GRAPH_FEATURES
from backend.ml.graph_features import extract_graph_feature_df, merge_graph_features


# ======================================================================
# Helpers
# ======================================================================


def _make_trained_classifier(
    n_samples: int = 200, n_features: int = 5, seed: int = 42
) -> tuple[TransactionClassifier, np.ndarray, np.ndarray, np.ndarray]:
    """Create a trained classifier with matching test data and time_steps.

    Returns (classifier, X_test, y_test, time_steps_test).
    """
    rng = np.random.RandomState(seed)

    # Train data
    X_train = rng.randn(n_samples, n_features)
    y_train = (X_train[:, 0] + rng.randn(n_samples) * 0.3 > 0).astype(int)

    # Test data with time_steps
    n_test = n_samples // 4
    X_test = rng.randn(n_test, n_features)
    y_test = (X_test[:, 0] + rng.randn(n_test) * 0.3 > 0).astype(int)
    # 3 distinct test time steps
    time_steps = rng.choice([35, 40, 49], size=n_test)

    clf = TransactionClassifier(model_version="eval_test", feature_set_name="M1")
    clf.train(X_train, y_train)

    return clf, X_test, y_test, time_steps


def _make_simple_graph() -> ChainTraceGraph:
    """Create a small graph for graph feature testing.

    Structure:
        tx_1 → tx_2 → tx_3
        addr_A → tx_1
        tx_3 → addr_B
    """
    graph = ChainTraceGraph()

    graph.add_tx_tx_edges([
        TxTxEdge(source_txid=1, target_txid=2),
        TxTxEdge(source_txid=2, target_txid=3),
    ])
    graph.add_addr_tx_edges([
        AddrTxEdge(input_address="addr_A", txid=1),
    ])
    graph.add_tx_addr_edges([
        TxAddrEdge(txid=3, output_address="addr_B"),
    ])

    return graph


# ======================================================================
# Temporal Evaluation Tests (Step 9)
# ======================================================================


class TestTemporalEvaluation:
    """Tests for evaluate_temporal and summarize_evaluation."""

    def test_basic_evaluation(self):
        """Temporal evaluation returns correct structure."""
        clf, X_test, y_test, time_steps = _make_trained_classifier()

        result = evaluate_temporal(clf, X_test, y_test, time_steps)

        assert isinstance(result, TemporalEvaluation)
        assert result.model_version == "eval_test"
        assert result.feature_set_name == "M1"
        assert len(result.overall_metrics) > 0
        assert "accuracy" in result.overall_metrics
        assert "f1" in result.overall_metrics

    def test_per_step_metrics(self):
        """Per-step metrics are computed for each unique time_step."""
        clf, X_test, y_test, time_steps = _make_trained_classifier()

        result = evaluate_temporal(clf, X_test, y_test, time_steps)

        unique_steps = set(time_steps)
        assert len(result.per_step_metrics) == len(unique_steps)
        for step in unique_steps:
            assert int(step) in result.per_step_metrics

    def test_per_step_sample_counts(self):
        """Per-step sample counts match actual data."""
        clf, X_test, y_test, time_steps = _make_trained_classifier()

        result = evaluate_temporal(clf, X_test, y_test, time_steps)

        total_samples = sum(m.n_samples for m in result.per_step_metrics.values())
        assert total_samples == len(y_test)

    def test_per_step_class_distribution(self):
        """Per-step class distribution is tracked."""
        clf, X_test, y_test, time_steps = _make_trained_classifier()

        result = evaluate_temporal(clf, X_test, y_test, time_steps)

        for step, metrics in result.per_step_metrics.items():
            assert metrics.n_illicit >= 0
            assert metrics.n_licit >= 0
            assert metrics.n_illicit + metrics.n_licit == metrics.n_samples

    def test_step_metrics_in_valid_range(self):
        """Per-step accuracy/precision/recall/f1 are in [0, 1]."""
        clf, X_test, y_test, time_steps = _make_trained_classifier()

        result = evaluate_temporal(clf, X_test, y_test, time_steps)

        for step, metrics in result.per_step_metrics.items():
            if metrics.accuracy is not None:
                assert 0.0 <= metrics.accuracy <= 1.0
            if metrics.precision is not None:
                assert 0.0 <= metrics.precision <= 1.0
            if metrics.recall is not None:
                assert 0.0 <= metrics.recall <= 1.0
            if metrics.f1 is not None:
                assert 0.0 <= metrics.f1 <= 1.0

    def test_single_class_step_handled(self):
        """Steps with only one class don't crash (ROC-AUC = None)."""
        rng = np.random.RandomState(99)
        clf = TransactionClassifier(model_version="single_class_test")
        X_train = rng.randn(100, 3)
        y_train = (X_train[:, 0] > 0).astype(int)
        clf.train(X_train, y_train)

        # All samples in step 35 are licit (y=0)
        X_test = rng.randn(20, 3)
        y_test = np.zeros(20, dtype=int)
        time_steps = np.full(20, 35)

        result = evaluate_temporal(clf, X_test, y_test, time_steps)

        step_metrics = result.per_step_metrics[35]
        assert step_metrics.n_illicit == 0
        assert step_metrics.roc_auc is None  # undefined with single class

    def test_time_steps_evaluated_sorted(self):
        """time_steps_evaluated is sorted."""
        clf, X_test, y_test, time_steps = _make_trained_classifier()

        result = evaluate_temporal(clf, X_test, y_test, time_steps)

        assert result.time_steps_evaluated == sorted(result.time_steps_evaluated)

    def test_empty_data_raises(self):
        """Empty test data raises ValueError."""
        clf, _, _, _ = _make_trained_classifier()

        with pytest.raises(ValueError, match="empty"):
            evaluate_temporal(
                clf, np.empty((0, 5)), np.empty(0), np.empty(0, dtype=int)
            )

    def test_shape_mismatch_raises(self):
        """Mismatched X/y shapes raise ValueError."""
        clf, X_test, _, time_steps = _make_trained_classifier()

        with pytest.raises(ValueError, match="Shape mismatch"):
            evaluate_temporal(clf, X_test, np.array([0, 1]), time_steps)

    def test_time_steps_mismatch_raises(self):
        """Mismatched X/time_steps shapes raise ValueError."""
        clf, X_test, y_test, _ = _make_trained_classifier()

        with pytest.raises(ValueError, match="Shape mismatch"):
            evaluate_temporal(clf, X_test, y_test, np.array([35, 40]))

    def test_summarize_is_json_serializable(self):
        """summarize_evaluation returns JSON-serializable output."""
        clf, X_test, y_test, time_steps = _make_trained_classifier()

        result = evaluate_temporal(clf, X_test, y_test, time_steps)
        summary = summarize_evaluation(result)

        # Must not raise
        json_str = json.dumps(summary)
        assert len(json_str) > 0

    def test_summarize_contains_required_fields(self):
        """Summary has model_version, overall_metrics, per_step_metrics."""
        clf, X_test, y_test, time_steps = _make_trained_classifier()

        result = evaluate_temporal(clf, X_test, y_test, time_steps)
        summary = summarize_evaluation(result)

        assert summary["model_version"] == "eval_test"
        assert summary["feature_set_name"] == "M1"
        assert "overall_metrics" in summary
        assert "per_step_metrics" in summary
        assert summary["n_steps"] == len(result.time_steps_evaluated)


# ======================================================================
# Anomaly Detection Tests (Step 10)
# ======================================================================


class TestAnomalyDetector:
    """Tests for AnomalyDetector."""

    def test_fit_predict(self):
        """Basic fit + predict works."""
        rng = np.random.RandomState(42)
        X = rng.randn(200, 10)

        detector = AnomalyDetector(model_version="test_ad")
        detector.fit(X)

        preds = detector.predict(X)
        assert preds.shape == (200,)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_scores_in_range(self):
        """Anomaly scores are in [0, 1]."""
        rng = np.random.RandomState(42)
        X = rng.randn(200, 10)

        detector = AnomalyDetector()
        detector.fit(X)

        scores = detector.score(X)
        assert scores.min() >= 0.0
        assert scores.max() <= 1.0

    def test_outliers_scored_higher(self):
        """Injected outliers should have higher anomaly scores on average."""
        rng = np.random.RandomState(42)
        # Normal data
        X_normal = rng.randn(300, 5)
        # Outliers: far from normal distribution
        X_outliers = rng.randn(20, 5) * 10 + 15

        detector = AnomalyDetector(contamination=0.05, random_state=42)
        detector.fit(X_normal)

        scores_normal = detector.score(X_normal)
        scores_outliers = detector.score(X_outliers)

        # Outliers should have higher mean score
        assert scores_outliers.mean() > scores_normal.mean()

    def test_is_fitted_flag(self):
        """is_fitted is False before fit, True after."""
        detector = AnomalyDetector()
        assert detector.is_fitted is False

        X = np.random.randn(50, 3)
        detector.fit(X)
        assert detector.is_fitted is True

    def test_predict_before_fit_raises(self):
        """Predict before fit raises RuntimeError."""
        detector = AnomalyDetector()

        with pytest.raises(RuntimeError, match="not been fitted"):
            detector.predict(np.random.randn(5, 3))

    def test_score_before_fit_raises(self):
        """Score before fit raises RuntimeError."""
        detector = AnomalyDetector()

        with pytest.raises(RuntimeError, match="not been fitted"):
            detector.score(np.random.randn(5, 3))

    def test_fit_empty_raises(self):
        """Fitting on empty data raises ValueError."""
        detector = AnomalyDetector()

        with pytest.raises(ValueError, match="empty"):
            detector.fit(np.empty((0, 5)))

    def test_explain_score_categories(self):
        """explain_score returns human-readable strings for different score levels."""
        assert "Highly anomalous" in AnomalyDetector.explain_score(0.95)
        assert "Moderately anomalous" in AnomalyDetector.explain_score(0.65)
        assert "Mildly anomalous" in AnomalyDetector.explain_score(0.45)
        assert "Normal" in AnomalyDetector.explain_score(0.1)

    def test_explain_score_no_feature_names(self):
        """Explanations must not contain anonymized feature references."""
        for score in [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]:
            explanation = AnomalyDetector.explain_score(score)
            assert "Local_feature" not in explanation
            assert "Aggregate_feature" not in explanation

    def test_explain_score_boundary(self):
        """Edge cases: 0.0 and 1.0."""
        assert "Normal" in AnomalyDetector.explain_score(0.0)
        assert "Highly anomalous" in AnomalyDetector.explain_score(1.0)


class TestAnomalyPersistence:
    """Tests for anomaly detector save/load."""

    def test_save_creates_files(self, tmp_path: Path):
        """Save creates both .joblib and _metadata.json."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)

        detector = AnomalyDetector(model_version="ad_save_test")
        detector.fit(X)

        model_path = detector.save(tmp_path)
        assert model_path.exists()
        assert (tmp_path / "ad_save_test_metadata.json").exists()

    def test_metadata_fields(self, tmp_path: Path):
        """Metadata contains expected fields."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)

        detector = AnomalyDetector(
            model_version="ad_meta_test", contamination=0.15
        )
        detector.fit(X)
        detector.save(tmp_path)

        with open(tmp_path / "ad_meta_test_metadata.json") as f:
            metadata = json.load(f)

        assert metadata["model_version"] == "ad_meta_test"
        assert metadata["model_type"] == "IsolationForest"
        assert metadata["contamination"] == 0.15
        assert "saved_at" in metadata

    def test_load_roundtrip(self, tmp_path: Path):
        """Load reproduces the same scores."""
        rng = np.random.RandomState(42)
        X = rng.randn(100, 5)

        detector = AnomalyDetector(model_version="ad_roundtrip")
        detector.fit(X)
        original_scores = detector.score(X[:10])

        model_path = detector.save(tmp_path)
        loaded = AnomalyDetector.load(model_path)

        assert loaded.model_version == "ad_roundtrip"
        assert loaded.is_fitted is True

        loaded_scores = loaded.score(X[:10])
        np.testing.assert_array_almost_equal(original_scores, loaded_scores)

    def test_save_before_fit_raises(self, tmp_path: Path):
        """Save before fit raises RuntimeError."""
        detector = AnomalyDetector()

        with pytest.raises(RuntimeError, match="not been fitted"):
            detector.save(tmp_path)

    def test_load_missing_raises(self, tmp_path: Path):
        """Loading nonexistent file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            AnomalyDetector.load(tmp_path / "nonexistent.joblib")


# ======================================================================
# Graph Feature Evaluation Tests (Step 11)
# ======================================================================


class TestGraphFeatureExtraction:
    """Tests for extract_graph_feature_df and merge_graph_features."""

    def test_extract_returns_correct_columns(self):
        """Extraction produces txId + 8 graph feature columns."""
        graph = _make_simple_graph()
        df = extract_graph_feature_df(graph)

        expected_cols = ["txId"] + GRAPH_FEATURES
        assert list(df.columns) == expected_cols

    def test_extract_correct_row_count(self):
        """Extraction returns one row per transaction node."""
        graph = _make_simple_graph()
        df = extract_graph_feature_df(graph)

        # Graph has 3 transaction nodes: 1, 2, 3
        assert len(df) == 3

    def test_extract_with_specific_ids(self):
        """Extracting for specific IDs returns only those rows."""
        graph = _make_simple_graph()
        df = extract_graph_feature_df(graph, transaction_ids=[1, 2])

        assert len(df) == 2
        assert set(df["txId"]) == {1, 2}

    def test_extract_missing_ids_excluded(self):
        """IDs not in graph are silently excluded."""
        graph = _make_simple_graph()
        df = extract_graph_feature_df(graph, transaction_ids=[1, 999])

        assert len(df) == 1
        assert df["txId"].iloc[0] == 1

    def test_extract_empty_graph(self):
        """Empty graph returns empty DataFrame with correct columns."""
        graph = ChainTraceGraph()
        df = extract_graph_feature_df(graph)

        assert len(df) == 0
        assert list(df.columns) == ["txId"] + GRAPH_FEATURES

    def test_degree_features_correct(self):
        """Degree features are computed correctly for known graph."""
        graph = _make_simple_graph()
        df = extract_graph_feature_df(graph)

        # tx_2 has: in from tx_1, out to tx_3 → in_degree >= 1, out_degree >= 1
        tx2_row = df[df["txId"] == 2].iloc[0]
        assert tx2_row["in_degree"] >= 1
        assert tx2_row["out_degree"] >= 1
        assert tx2_row["total_degree"] == tx2_row["in_degree"] + tx2_row["out_degree"]

    def test_pagerank_nonnegative(self):
        """PageRank values are non-negative."""
        graph = _make_simple_graph()
        df = extract_graph_feature_df(graph)

        assert (df["pagerank"] >= 0).all()

    def test_merge_preserves_all_transactions(self):
        """Left merge preserves all transaction rows."""
        graph = _make_simple_graph()
        graph_df = extract_graph_feature_df(graph)

        # Transaction DataFrame has more IDs than the graph
        tx_df = pd.DataFrame({
            "txId": [1, 2, 3, 4, 5],
            "feat_a": [0.1, 0.2, 0.3, 0.4, 0.5],
        })

        merged = merge_graph_features(tx_df, graph_df)

        assert len(merged) == 5  # All original rows preserved

    def test_merge_fills_missing_with_zero(self):
        """Transactions not in graph get 0 for graph features."""
        graph = _make_simple_graph()
        graph_df = extract_graph_feature_df(graph)

        tx_df = pd.DataFrame({
            "txId": [1, 999],  # 999 not in graph
            "feat_a": [0.1, 0.2],
        })

        merged = merge_graph_features(tx_df, graph_df)

        row_999 = merged[merged["txId"] == 999].iloc[0]
        for feat in GRAPH_FEATURES:
            assert row_999[feat] == 0.0

    def test_merge_missing_txid_raises(self):
        """Missing txId column raises ValueError."""
        graph_df = pd.DataFrame({"id": [1], "in_degree": [2]})
        tx_df = pd.DataFrame({"txId": [1]})

        with pytest.raises(ValueError, match="txId"):
            merge_graph_features(tx_df, graph_df)

    def test_m2_training_integration(self):
        """M2 features (blockchain + graph) can train a classifier."""
        rng = np.random.RandomState(42)
        n_features_blockchain = 5
        n_features_graph = len(GRAPH_FEATURES)
        n_total = n_features_blockchain + n_features_graph

        X = rng.randn(100, n_total)
        y = (X[:, 0] > 0).astype(int)

        clf = TransactionClassifier(
            model_version="m2_test", feature_set_name="M2"
        )
        metrics = clf.train(X, y)

        assert metrics["train_accuracy"] > 0.0
        assert clf.is_fitted

        preds = clf.predict(X[:10])
        assert preds.shape == (10,)
