"""Tests for Kautilya domain models.

Validates model creation, field constraints, enum values,
is_synthetic provenance, and error handling for invalid data.
"""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from backend.domain import (
    EntityClass,
    Evidence,
    EvidenceType,
    ExperimentConfig,
    GeneratorConfig,
    MLScore,
    NetworkObservation,
    RiskScore,
    ScriptType,
    StatsSummary,
    Transaction,
    TransactionFeatures,
    TxTxEdge,
    Wallet,
)

# --- EntityClass enum ---


class TestEntityClass:
    def test_illicit_value(self):
        assert EntityClass.ILLICIT == 1

    def test_licit_value(self):
        assert EntityClass.LICIT == 2

    def test_unknown_value(self):
        assert EntityClass.UNKNOWN == 3

    def test_all_values_covered(self):
        assert set(EntityClass) == {
            EntityClass.ILLICIT,
            EntityClass.LICIT,
            EntityClass.UNKNOWN,
        }


# --- ScriptType enum ---


class TestScriptType:
    def test_all_required_types_present(self):
        """SIH requires P2PKH, P2SH, P2WPKH, and Taproot."""
        expected = {"P2PKH", "P2SH", "P2WPKH", "TAPROOT"}
        actual = {s.value for s in ScriptType}
        assert actual == expected


# --- EvidenceType enum ---


class TestEvidenceType:
    def test_all_forensic_categories(self):
        """Must cover all four evidence categories from CONTEXT.md."""
        expected = {"observation", "correlation", "model_prediction", "risk_assessment"}
        actual = {e.value for e in EvidenceType}
        assert actual == expected


# --- Transaction ---


class TestTransaction:
    def test_valid_transaction(self, sample_transaction):
        assert sample_transaction.txid == 3321
        assert sample_transaction.time_step == 1
        assert sample_transaction.label == EntityClass.UNKNOWN

    def test_minimal_transaction(self):
        tx = Transaction(txid=100, time_step=1)
        assert tx.label is None
        assert tx.total_btc is None

    def test_all_interpretable_features(self):
        tx = Transaction(
            txid=1,
            time_step=1,
            total_btc=0.5,
            fees=0.001,
            size=250.0,
            num_input_addresses=2.0,
            num_output_addresses=3.0,
            in_txs_degree=5.0,
            out_txs_degree=3.0,
            in_btc_min=0.1,
            in_btc_max=0.4,
            in_btc_mean=0.25,
            in_btc_median=0.25,
            in_btc_total=0.5,
            out_btc_min=0.05,
            out_btc_max=0.3,
            out_btc_mean=0.15,
            out_btc_median=0.1,
            out_btc_total=0.45,
        )
        assert tx.total_btc == 0.5
        assert tx.out_btc_total == 0.45

    def test_invalid_time_step_too_low(self):
        with pytest.raises(ValidationError):
            Transaction(txid=1, time_step=0)

    def test_invalid_time_step_too_high(self):
        with pytest.raises(ValidationError):
            Transaction(txid=1, time_step=50)

    def test_boundary_time_steps(self):
        tx_min = Transaction(txid=1, time_step=1)
        tx_max = Transaction(txid=2, time_step=49)
        assert tx_min.time_step == 1
        assert tx_max.time_step == 49


# --- TransactionFeatures ---


class TestTransactionFeatures:
    def test_valid_features(self, sample_transaction_features):
        assert len(sample_transaction_features.local_features) == 93
        assert len(sample_transaction_features.aggregate_features) == 72

    def test_wrong_local_feature_count(self):
        """Should still create (no length validation in schema), but
        normalization code will validate counts."""
        tf = TransactionFeatures(
            txid=1,
            local_features=[0.0] * 10,
            aggregate_features=[0.0] * 72,
        )
        assert len(tf.local_features) == 10  # schema allows any length

    def test_empty_features(self):
        tf = TransactionFeatures(
            txid=1,
            local_features=[],
            aggregate_features=[],
        )
        assert tf.local_features == []


# --- Wallet ---


class TestWallet:
    def test_valid_wallet(self, sample_wallet):
        assert sample_wallet.address == "14YRXHHof4BY1TVxN5FqYPcEdpmXiYT78a"
        assert sample_wallet.time_step == 25
        assert sample_wallet.label == EntityClass.LICIT

    def test_wallet_with_stats_summary(self, sample_wallet):
        assert sample_wallet.btc_transacted is not None
        assert sample_wallet.btc_transacted.total == 1.5
        assert sample_wallet.btc_transacted.min == 0.01

    def test_minimal_wallet(self):
        w = Wallet(address="test_addr", time_step=1)
        assert w.label is None
        assert w.btc_transacted is None

    def test_wallet_no_label(self):
        """~445K addresses in the dataset lack labels."""
        w = Wallet(address="unlabeled_addr", time_step=10)
        assert w.label is None

    def test_stats_summary_creation(self):
        stats = StatsSummary(total=10.0, min=0.5, max=5.0, mean=2.5, median=2.0)
        assert stats.total == 10.0


# --- NetworkObservation ---


class TestNetworkObservation:
    def test_always_synthetic(self, sample_network_observation):
        """Network layer is always synthetic by design."""
        assert sample_network_observation.is_synthetic is True

    def test_cannot_set_synthetic_false(self):
        """Even if explicitly set to False, the model allows it but the
        invariant should be enforced by generator code."""
        obs = NetworkObservation(
            txid=1,
            src_ip="1.2.3.4",
            dst_ip="5.6.7.8",
            src_port=8333,
            dst_port=8333,
            timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            is_synthetic=False,  # violates invariant
        )
        # Model allows it — enforcement is at the generator level
        assert obs.is_synthetic is False

    def test_provenance_fields(self, sample_network_observation):
        assert sample_network_observation.generation_run_id == "run_001"
        assert sample_network_observation.scenario_id == "normal_propagation"
        assert sample_network_observation.generator_version == "0.1.0"

    def test_script_type_assignment(self, sample_network_observation):
        assert sample_network_observation.script_type == ScriptType.P2PKH

    def test_invalid_port(self):
        with pytest.raises(ValidationError):
            NetworkObservation(
                txid=1,
                src_ip="1.2.3.4",
                dst_ip="5.6.7.8",
                src_port=70000,  # invalid
                dst_port=8333,
                timestamp=datetime(2024, 1, 1, tzinfo=UTC),
            )


# --- Edge models ---


class TestEdgeModels:
    def test_tx_tx_edge(self, sample_edges):
        edge = sample_edges["tx_tx"]
        assert edge.source_txid == 230425980
        assert edge.target_txid == 5530458
        assert edge.is_synthetic is False

    def test_addr_tx_edge(self, sample_edges):
        edge = sample_edges["addr_tx"]
        assert edge.input_address == "14YRXHHof4BY1TVxN5FqYPcEdpmXiYT78a"
        assert edge.txid == 230325127
        assert edge.is_synthetic is False

    def test_tx_addr_edge(self, sample_edges):
        edge = sample_edges["tx_addr"]
        assert edge.txid == 230325127
        assert edge.output_address == "1GASxu5nMntiRKdVtTVRvEbP965G51bhHH"
        assert edge.is_synthetic is False

    def test_addr_addr_edge(self, sample_edges):
        edge = sample_edges["addr_addr"]
        assert edge.input_address == "14YRXHHof4BY1TVxN5FqYPcEdpmXiYT78a"
        assert edge.output_address == "1GASxu5nMntiRKdVtTVRvEbP965G51bhHH"
        assert edge.is_synthetic is False

    def test_synthetic_edge(self):
        edge = TxTxEdge(source_txid=1, target_txid=2, is_synthetic=True)
        assert edge.is_synthetic is True


# --- MLScore ---


class TestMLScore:
    def test_valid_ml_score(self, sample_ml_score):
        assert sample_ml_score.prediction == 1
        assert sample_ml_score.probability == 0.87
        assert sample_ml_score.model_version == "rf_v1"

    def test_probability_bounds(self):
        with pytest.raises(ValidationError):
            MLScore(
                entity_id="1",
                entity_type="transaction",
                prediction=1,
                probability=1.5,  # > 1.0
                model_version="v1",
            )

    def test_synthetic_input_tracking(self):
        score = MLScore(
            entity_id="1",
            entity_type="transaction",
            prediction=2,
            model_version="v1",
            contains_synthetic_input=True,
        )
        assert score.contains_synthetic_input is True


# --- RiskScore ---


class TestRiskScore:
    def test_valid_risk_score(self, sample_risk_score):
        assert sample_risk_score.score == 0.75
        assert sample_risk_score.behavioral_signal == 0.8
        assert sample_risk_score.explanation is not None

    def test_minimal_risk_score(self):
        rs = RiskScore(entity_id="1", entity_type="wallet", score=0.5)
        assert rs.behavioral_signal is None
        assert rs.contains_synthetic_input is False


# --- Evidence ---


class TestEvidence:
    def test_valid_evidence(self, sample_evidence):
        assert sample_evidence.evidence_type == EvidenceType.OBSERVATION
        assert sample_evidence.is_synthetic is False

    def test_synthetic_evidence(self):
        ev = Evidence(
            evidence_type=EvidenceType.CORRELATION,
            source_entity_id="1",
            source_entity_type="network_event",
            description="Temporal correlation with network event",
            confidence=0.6,
            is_synthetic=True,
        )
        assert ev.is_synthetic is True

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            Evidence(
                evidence_type=EvidenceType.OBSERVATION,
                source_entity_id="1",
                source_entity_type="transaction",
                description="test",
                confidence=2.0,  # > 1.0
            )


# --- ExperimentConfig ---


class TestExperimentConfig:
    def test_valid_config(self, sample_experiment_config):
        assert sample_experiment_config.split_strategy == "temporal"
        assert len(sample_experiment_config.train_time_steps) == 34
        assert len(sample_experiment_config.test_time_steps) == 15

    def test_missing_required_fields(self):
        with pytest.raises(ValidationError):
            ExperimentConfig(
                experiment_id="exp_001",
                # missing required fields
            )


# --- GeneratorConfig ---


class TestGeneratorConfig:
    def test_valid_config(self, sample_generator_config):
        assert sample_generator_config.random_seed == 42
        assert sample_generator_config.node_pool_size == 1000

    def test_empty_scenario_configs(self):
        cfg = GeneratorConfig(
            random_seed=1,
            generator_version="0.1.0",
        )
        assert cfg.scenario_configs == {}
        assert cfg.node_pool_size is None
