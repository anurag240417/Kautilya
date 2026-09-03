"""Tests for Phase 7 (Steps 1–3) Risk Engine.

Verifies:
1. Extensible SignalInput model (core signals + open-ended custom signals).
2. Multi-signal synthesis without arbitrary unvalidated weights.
3. Principled corroboration boost and anomaly safety capping (Anomaly ≠ Illicitness).
4. Priority tier assignment (Critical, High, Medium, Low).
5. Structured Evidence Ledger across all 5+ categories.
6. Strict explainability compliance:
   - ZERO occurrences of `Local_feature_*` or `Aggregate_feature_*` in human explanations.
   - Anomaly explanations clearly disclaim confirmed illicitness.
   - Graph explanations disclaim centrality as proof of guilt.
   - Synthetic network/temporal evidence carry `is_synthetic=True` and non-attribution caveats.
"""

from datetime import UTC, datetime, timedelta

import pytest

from backend.domain.correlation import (
    CandidateIPRole,
    TemporalCorrelation,
    TransactionIPCorrelation,
)
from backend.domain.ml import MLScore
from backend.domain.risk import (
    EvidenceCategory,
    EvidenceLedger,
    SignalInput,
)
from backend.domain.transaction import Transaction
from backend.domain.types import EntityClass, PriorityTier
from backend.risk.evidence import (
    build_evidence_ledger,
    generate_narrative_explanation,
)
from backend.risk.scorer import (
    DEFAULT_SYNTHESIS_CONFIG,
    determine_priority_tier,
    synthesize_risk_score,
)


class TestSignalInput:
    """Tests for extensible SignalInput domain model."""

    def test_valid_signal_input(self):
        signals = SignalInput(
            entity_id="tx_123",
            entity_type="transaction",
            illicit_probability=0.85,
            predicted_label=EntityClass.ILLICIT,
            anomaly_score=0.60,
            graph_signal=0.70,
            correlation_confidence=0.90,
            known_indicator_signal=None,
            contains_synthetic_input=True,
            time_step=10,
        )
        assert signals.entity_id == "tx_123"
        assert signals.illicit_probability == 0.85
        assert signals.predicted_label == EntityClass.ILLICIT
        assert signals.contains_synthetic_input is True

    def test_signal_bounds_validation(self):
        with pytest.raises(ValueError):
            SignalInput(entity_id="tx_bad", entity_type="transaction", illicit_probability=1.5)
        with pytest.raises(ValueError):
            SignalInput(entity_id="tx_bad", entity_type="transaction", anomaly_score=-0.1)

    def test_open_ended_custom_signals(self):
        signals = SignalInput(
            entity_id="wallet_xyz",
            entity_type="wallet",
            custom_signals={"mixer_interaction_depth": 0.75, "darknet_mention_score": 0.90},
        )
        assert signals.custom_signals["mixer_interaction_depth"] == 0.75
        assert signals.custom_signals["darknet_mention_score"] == 0.90


class TestMultiSignalSynthesis:
    """Tests for multi-signal synthesis and priority tiering."""

    def test_empty_signals_returns_zero_low_tier(self):
        signals = SignalInput(entity_id="tx_empty", entity_type="transaction")
        risk = synthesize_risk_score(signals)
        assert risk.score == 0.0
        assert risk.priority_tier == PriorityTier.LOW
        assert risk.active_signals == []
        assert "No investigative signals" in risk.explanation

    def test_dynamic_normalization_single_signal(self):
        # A single strong supervised signal should not be suppressed by missing signals
        signals = SignalInput(
            entity_id="tx_single",
            entity_type="transaction",
            illicit_probability=0.85,
        )
        risk = synthesize_risk_score(signals)
        # Should be approximately 85.0 (scaled to 100)
        assert 84.0 <= risk.score <= 86.0
        assert risk.priority_tier == PriorityTier.CRITICAL
        assert risk.active_signals == ["illicit_probability"]

    def test_corroborated_tier_boost(self):
        # Supervised illicit probability + anomaly + graph corroboration
        signals = SignalInput(
            entity_id="tx_corroborated",
            entity_type="transaction",
            illicit_probability=0.75,
            anomaly_score=0.70,
            graph_signal=0.80,
        )
        risk = synthesize_risk_score(signals)
        # Baseline ~ 75.3 * 1.15 boost = ~86.6 -> Critical tier
        assert risk.score >= 80.0
        assert risk.priority_tier == PriorityTier.CRITICAL
        assert len(risk.active_signals) == 3

    def test_anomaly_safety_capping(self):
        # High anomaly deviance ALONE without supervised illicitness must not reach Critical tier
        signals = SignalInput(
            entity_id="tx_novelty",
            entity_type="transaction",
            anomaly_score=0.95,  # Very high statistical deviance
            illicit_probability=None,
            graph_signal=None,
        )
        risk = synthesize_risk_score(signals)
        # Must be capped at uncorroborated_anomaly_cap (60.0)
        assert risk.score <= DEFAULT_SYNTHESIS_CONFIG.uncorroborated_anomaly_cap
        assert risk.priority_tier in (PriorityTier.MEDIUM, PriorityTier.HIGH)
        assert risk.priority_tier != PriorityTier.CRITICAL
        assert "statistical anomaly deviance without confirmed illicit" in risk.explanation

    def test_synthetic_input_tracking(self):
        signals = SignalInput(
            entity_id="tx_synth",
            entity_type="transaction",
            illicit_probability=0.50,
            correlation_confidence=0.80,  # network layer is synthetic
        )
        risk = synthesize_risk_score(signals)
        assert risk.contains_synthetic_input is True
        assert "synthetic" in risk.explanation.lower()

    def test_priority_tier_thresholds(self):
        cfg = DEFAULT_SYNTHESIS_CONFIG
        assert determine_priority_tier(85.0, cfg) == PriorityTier.CRITICAL
        assert determine_priority_tier(80.0, cfg) == PriorityTier.CRITICAL
        assert determine_priority_tier(75.0, cfg) == PriorityTier.HIGH
        assert determine_priority_tier(60.0, cfg) == PriorityTier.HIGH
        assert determine_priority_tier(45.0, cfg) == PriorityTier.MEDIUM
        assert determine_priority_tier(40.0, cfg) == PriorityTier.MEDIUM
        assert determine_priority_tier(35.0, cfg) == PriorityTier.LOW
        assert determine_priority_tier(0.0, cfg) == PriorityTier.LOW


class TestStructuredEvidenceLedger:
    """Tests for structured Evidence Ledger compilation and natural explainability."""

    @pytest.fixture
    def sample_evidence_context(self):
        tx = Transaction(
            txid=98765,
            time_step=12,
            total_btc=124.50,
            fees=0.005,
            num_input_addresses=3,
            num_output_addresses=2,
            in_txs_degree=4,
            out_txs_degree=8,
        )
        ml_score = MLScore(
            entity_id="98765",
            entity_type="transaction",
            prediction=1,
            probability=0.88,
            model_version="rf_M2_v1",
        )
        t_start = datetime(2019, 6, 17, 0, 0, 0, tzinfo=UTC)
        temp_corr = TemporalCorrelation(
            txid=98765,
            time_step=12,
            observation_timestamp=t_start + timedelta(days=2),
            window_start=t_start,
            window_end=t_start + timedelta(days=14),
            delta_seconds=0.0,
            temporal_proximity=1.0,
            is_synthetic=True,
        )
        tx_ip_corr = TransactionIPCorrelation(
            txid=98765,
            ip="203.0.113.45",
            role=CandidateIPRole.ORIGIN,
            correlation_confidence=0.92,
            metrics={
                "temporal_proximity": 1.0,
                "propagation_consistency": 1.0,
                "topology_consistency": 0.8,
                "composite_confidence": 0.92,
            },
            observations_count=3,
            earliest_observation=t_start + timedelta(days=2),
            latest_observation=t_start + timedelta(days=2, seconds=15),
            asn=13335,
            country="US",
            is_synthetic=True,
        )
        return tx, ml_score, temp_corr, tx_ip_corr

    def test_build_full_evidence_ledger(self, sample_evidence_context):
        tx, ml_score, temp_corr, tx_ip_corr = sample_evidence_context

        ledger = build_evidence_ledger(
            entity_id="98765",
            entity_type="transaction",
            tx=tx,
            ml_score=ml_score,
            anomaly_score=0.82,
            graph_metrics={"pagerank": 0.0042, "betweenness": 0.015},
            temporal_correlations=[temp_corr],
            tx_ip_correlations=[tx_ip_corr],
            known_indicator_match="Large high-frequency cluster heuristic",
        )

        assert isinstance(ledger, EvidenceLedger)
        assert len(ledger.records) == 6
        assert ledger.has_synthetic_evidence() is True

        # Check categories
        categories = {r.category for r in ledger.records}
        assert categories == {
            EvidenceCategory.ML_BEHAVIORAL,
            EvidenceCategory.ANOMALY,
            EvidenceCategory.GRAPH_STRUCTURAL,
            EvidenceCategory.TEMPORAL,
            EvidenceCategory.SYNTHETIC_NETWORK,
            EvidenceCategory.KNOWN_INDICATOR,
        }

    def test_explainability_no_anonymized_feature_leak(self, sample_evidence_context):
        """Crucial test: Ensure ZERO occurrences of Local_feature_* or Aggregate_feature_*."""
        tx, ml_score, temp_corr, tx_ip_corr = sample_evidence_context

        # Supply raw features dictionary containing anonymized columns
        raw_features = {
            "total_btc": 124.50,
            "Local_feature_1": 0.45,
            "Local_feature_93": 1.22,
            "Aggregate_feature_1": -0.88,
        }

        ledger = build_evidence_ledger(
            entity_id="98765",
            entity_type="transaction",
            tx=tx,
            interpretable_features=raw_features,
            ml_score=ml_score,
            anomaly_score=0.78,
            graph_metrics={"pagerank": 0.005},
        )

        narrative = generate_narrative_explanation(ledger)

        # Assert no anonymized feature names leak into narrative or records
        assert "Local_feature" not in narrative
        assert "Aggregate_feature" not in narrative

        for r in ledger.records:
            assert "Local_feature" not in r.headline
            assert "Local_feature" not in r.description
            assert "Aggregate_feature" not in r.headline
            assert "Aggregate_feature" not in r.description
            for k in r.supporting_metrics:
                assert not k.startswith("Local_feature_")
                assert not k.startswith("Aggregate_feature_")

    def test_anomaly_explanation_disclaims_illicitness(self):
        ledger = build_evidence_ledger(
            entity_id="tx_111",
            entity_type="transaction",
            anomaly_score=0.91,
        )
        anom_records = ledger.get_by_category(EvidenceCategory.ANOMALY)
        assert len(anom_records) == 1
        desc = anom_records[0].description
        assert "statistical novelty" in desc.lower() or "statistical deviance" in desc.lower()
        assert "not confirmed illicit" in desc.lower()

    def test_synthetic_network_evidence_caveats(self, sample_evidence_context):
        tx, _, _, tx_ip_corr = sample_evidence_context
        ledger = build_evidence_ledger(
            entity_id="98765",
            entity_type="transaction",
            tx_ip_correlations=[tx_ip_corr],
        )
        net_records = ledger.get_by_category(EvidenceCategory.SYNTHETIC_NETWORK)
        assert len(net_records) == 1
        record = net_records[0]
        assert record.is_synthetic is True
        assert "ownership" in record.description.lower()

    def test_to_human_summary(self, sample_evidence_context):
        tx, ml_score, _, _ = sample_evidence_context
        ledger = build_evidence_ledger(
            entity_id="98765",
            entity_type="transaction",
            tx=tx,
            ml_score=ml_score,
        )
        summary = ledger.to_human_summary()
        assert len(summary) >= 1
        assert any("ML_BEHAVIORAL" in s for s in summary)
        assert any("124.50" in s for s in summary)
