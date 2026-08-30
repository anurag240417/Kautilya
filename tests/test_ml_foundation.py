"""Tests for ML foundation: feature registry, splitting, and leakage audit."""

import pandas as pd
import pytest

from backend.ml.feature_registry import (
    ALL_TX_FEATURES,
    ALL_WALLET_FEATURES,
    ANONYMIZED_AGGREGATE_FEATURES,
    ANONYMIZED_LOCAL_FEATURES,
    ANONYMIZED_TX_FEATURES,
    BINARY_TARGET_MAP,
    FEATURE_SETS,
    FORBIDDEN_FEATURE_COLUMNS,
    GRAPH_FEATURES,
    INTERPRETABLE_TX_FEATURES,
    LABEL_ILLICIT,
    LABEL_LICIT,
    LABEL_UNKNOWN,
    NETWORK_FEATURES,
    TRAIN_LABELS,
    WALLET_SCALAR_FEATURES,
    WALLET_STATS_FEATURES,
)
from backend.ml.leakage import audit_leakage
from backend.ml.splitting import temporal_train_test_split

# ======================================================================
# Feature Registry Tests
# ======================================================================


class TestFeatureRegistry:
    """Tests for feature count and composition correctness."""

    def test_interpretable_tx_feature_count(self):
        """There are exactly 17 interpretable transaction features."""
        assert len(INTERPRETABLE_TX_FEATURES) == 17

    def test_anonymized_local_feature_count(self):
        """There are exactly 93 local features."""
        assert len(ANONYMIZED_LOCAL_FEATURES) == 93

    def test_anonymized_aggregate_feature_count(self):
        """There are exactly 72 aggregate features."""
        assert len(ANONYMIZED_AGGREGATE_FEATURES) == 72

    def test_anonymized_tx_features_total(self):
        """93 + 72 = 165 anonymized transaction features."""
        assert len(ANONYMIZED_TX_FEATURES) == 165

    def test_all_tx_features_total(self):
        """165 anonymized + 17 interpretable = 182 total tx features."""
        assert len(ALL_TX_FEATURES) == 182

    def test_wallet_scalar_feature_count(self):
        """There are exactly 10 scalar wallet features."""
        assert len(WALLET_SCALAR_FEATURES) == 10

    def test_wallet_stats_feature_count(self):
        """9 groups × 5 suffixes = 45 wallet stats features."""
        assert len(WALLET_STATS_FEATURES) == 45

    def test_all_wallet_features_total(self):
        """10 + 45 = 55 total wallet features."""
        assert len(ALL_WALLET_FEATURES) == 55

    def test_graph_feature_count(self):
        """There are 8 graph-derived features."""
        assert len(GRAPH_FEATURES) == 8

    def test_network_feature_count(self):
        """There are 9 network-derived features."""
        assert len(NETWORK_FEATURES) == 9

    def test_no_duplicate_features(self):
        """No feature name should appear twice in any combined list."""
        assert len(ALL_TX_FEATURES) == len(set(ALL_TX_FEATURES))
        assert len(ALL_WALLET_FEATURES) == len(set(ALL_WALLET_FEATURES))

    def test_feature_set_m1_is_blockchain_only(self):
        """M1 contains only blockchain features (no graph, no network)."""
        m1 = FEATURE_SETS["M1"]
        assert m1 == ALL_TX_FEATURES
        for feat in GRAPH_FEATURES:
            assert feat not in m1
        for feat in NETWORK_FEATURES:
            assert feat not in m1

    def test_feature_set_m2_includes_graph(self):
        """M2 = M1 + graph features."""
        m2 = FEATURE_SETS["M2"]
        for feat in ALL_TX_FEATURES:
            assert feat in m2
        for feat in GRAPH_FEATURES:
            assert feat in m2
        for feat in NETWORK_FEATURES:
            assert feat not in m2

    def test_feature_set_m3_includes_all(self):
        """M3 = M1 + graph + network features."""
        m3 = FEATURE_SETS["M3"]
        for feat in ALL_TX_FEATURES:
            assert feat in m3
        for feat in GRAPH_FEATURES:
            assert feat in m3
        for feat in NETWORK_FEATURES:
            assert feat in m3

    def test_label_policy(self):
        """Label constants match the Elliptic++ encoding."""
        assert LABEL_ILLICIT == 1
        assert LABEL_LICIT == 2
        assert LABEL_UNKNOWN == 3
        assert frozenset({1, 2}) == TRAIN_LABELS

    def test_binary_target_map(self):
        """Illicit maps to 1 (positive), Licit maps to 0 (negative)."""
        assert BINARY_TARGET_MAP[LABEL_ILLICIT] == 1
        assert BINARY_TARGET_MAP[LABEL_LICIT] == 0
        assert LABEL_UNKNOWN not in BINARY_TARGET_MAP

    def test_forbidden_columns(self):
        """Key identity/target columns are forbidden from feature matrix."""
        assert "txId" in FORBIDDEN_FEATURE_COLUMNS
        assert "label" in FORBIDDEN_FEATURE_COLUMNS
        assert "class" in FORBIDDEN_FEATURE_COLUMNS
        assert "time_step" in FORBIDDEN_FEATURE_COLUMNS


# ======================================================================
# Temporal Split Tests
# ======================================================================


def _make_mock_df(steps: list[int], labels: list[int]) -> pd.DataFrame:
    """Create a mock DataFrame for split testing."""
    return pd.DataFrame(
        {
            "txId": range(len(steps)),
            "time_step": steps,
            "label": labels,
            "feat1": [0.5] * len(steps),
            "feat2": [1.0] * len(steps),
        }
    )


class TestTemporalSplit:
    """Tests for temporal_train_test_split."""

    def test_basic_split(self):
        """Rows are split correctly by time_step cutoff."""
        df = _make_mock_df(
            steps=[1, 2, 10, 20, 34, 35, 40, 49],
            labels=[1, 2, 1, 2, 1, 2, 1, 2],
        )
        train, test, config = temporal_train_test_split(df, train_cutoff=34)

        assert len(train) == 5
        assert len(test) == 3
        assert train["time_step"].max() <= 34
        assert test["time_step"].min() >= 35

    def test_binary_target_added(self):
        """A 'target' column is added with correct binary mapping."""
        df = _make_mock_df(
            steps=[1, 2, 35, 36],
            labels=[1, 2, 1, 2],
        )
        train, test, _ = temporal_train_test_split(df, train_cutoff=34)

        # Illicit (1) → target 1, Licit (2) → target 0
        assert train.loc[train["label"] == 1, "target"].iloc[0] == 1
        assert train.loc[train["label"] == 2, "target"].iloc[0] == 0

    def test_unknown_labels_excluded(self):
        """Unknown (3) labels are dropped from both train and test."""
        df = _make_mock_df(
            steps=[1, 2, 3, 35, 36, 37],
            labels=[1, 2, 3, 1, 2, 3],
        )
        train, test, _ = temporal_train_test_split(df, train_cutoff=34)

        assert 3 not in train["label"].values
        assert 3 not in test["label"].values
        assert len(train) == 2
        assert len(test) == 2

    def test_experiment_config_populated(self):
        """ExperimentConfig is correctly populated."""
        df = _make_mock_df(
            steps=[1, 10, 35, 49],
            labels=[1, 2, 1, 2],
        )
        _, _, config = temporal_train_test_split(
            df,
            train_cutoff=34,
            experiment_id="test_exp",
            feature_configuration="M2",
        )

        assert config.experiment_id == "test_exp"
        assert config.split_strategy == "temporal"
        assert config.feature_configuration == "M2"
        assert config.train_time_steps == list(range(1, 35))
        assert config.test_time_steps == list(range(35, 50))

    def test_missing_time_col_raises(self):
        """Missing time_step column raises ValueError."""
        df = pd.DataFrame({"label": [1, 2], "feat1": [0.5, 0.5]})
        with pytest.raises(ValueError, match="time_step"):
            temporal_train_test_split(df)

    def test_missing_label_col_raises(self):
        """Missing label column raises ValueError."""
        df = pd.DataFrame({"time_step": [1, 35], "feat1": [0.5, 0.5]})
        with pytest.raises(ValueError, match="label"):
            temporal_train_test_split(df)

    def test_all_unknown_raises(self):
        """All-unknown labels raises ValueError."""
        df = _make_mock_df(steps=[1, 35], labels=[3, 3])
        with pytest.raises(ValueError, match="No rows with supervised labels"):
            temporal_train_test_split(df)


# ======================================================================
# Leakage Audit Tests
# ======================================================================


class TestLeakageAudit:
    """Tests for audit_leakage."""

    def test_clean_audit_passes(self):
        """A properly split dataset passes all checks."""
        train = pd.DataFrame(
            {
                "time_step": [1, 2, 3],
                "target": [0, 1, 0],
                "feat1": [0.1, 0.2, 0.3],
            }
        )
        test = pd.DataFrame(
            {
                "time_step": [35, 36],
                "target": [1, 0],
                "feat1": [0.4, 0.5],
            }
        )
        result = audit_leakage(train, test, feature_columns=["feat1"])
        assert result.passed is True
        assert len(result.failures) == 0

    def test_temporal_overlap_fails(self):
        """Overlapping time_steps between train and test fails."""
        train = pd.DataFrame(
            {
                "time_step": [1, 2, 35],
                "target": [0, 1, 0],
                "feat1": [0.1, 0.2, 0.3],
            }
        )
        test = pd.DataFrame(
            {
                "time_step": [35, 36],
                "target": [1, 0],
                "feat1": [0.4, 0.5],
            }
        )
        result = audit_leakage(train, test, feature_columns=["feat1"])
        assert result.passed is False
        assert any("Temporal leakage" in f for f in result.failures)

    def test_forbidden_feature_column_fails(self):
        """Using 'label' as a feature column fails."""
        train = pd.DataFrame(
            {
                "time_step": [1, 2],
                "target": [0, 1],
                "label": [2, 1],
            }
        )
        test = pd.DataFrame(
            {
                "time_step": [35, 36],
                "target": [1, 0],
                "label": [1, 2],
            }
        )
        result = audit_leakage(train, test, feature_columns=["label"])
        assert result.passed is False
        assert any("forbidden columns" in f for f in result.failures)

    def test_invalid_target_values_fails(self):
        """Target with non-binary values (e.g. 3 for Unknown) fails."""
        train = pd.DataFrame(
            {
                "time_step": [1, 2, 3],
                "target": [0, 1, 3],
                "feat1": [0.1, 0.2, 0.3],
            }
        )
        test = pd.DataFrame(
            {
                "time_step": [35],
                "target": [0],
                "feat1": [0.5],
            }
        )
        result = audit_leakage(train, test, feature_columns=["feat1"])
        assert result.passed is False
        assert any("invalid values" in f for f in result.failures)

    def test_null_target_fails(self):
        """Null values in the training target fail."""
        train = pd.DataFrame(
            {
                "time_step": [1, 2],
                "target": [0, None],
                "feat1": [0.1, 0.2],
            }
        )
        test = pd.DataFrame(
            {
                "time_step": [35],
                "target": [1],
                "feat1": [0.5],
            }
        )
        result = audit_leakage(train, test, feature_columns=["feat1"])
        assert result.passed is False
        assert any("null values" in f for f in result.failures)
