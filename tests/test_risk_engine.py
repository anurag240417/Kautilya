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
    RiskScore,
    SignalInput,
)
from backend.domain.transaction import Transaction
from backend.domain.types import EntityClass, ExplanationType, PriorityTier
from backend.risk import (
    DEFAULT_SYNTHESIS_CONFIG,
    DEFAULT_TIER_CONFIG,
    AggregationMethod,
    PriorityTierConfig,
    PriorityTierDefinition,
    SynthesisConfig,
    aggregate_transaction_scores,
    build_evidence_ledger,
    determine_priority_tier,
    generate_narrative_explanation,
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


@pytest.fixture
def sample_evidence_context():
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


class TestStructuredEvidenceLedger:
    """Tests for structured Evidence Ledger compilation and natural explainability."""

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


class TestExplanationTypeDistinction:
    """Tests distinguishing classifier explanations from anomaly explanations (AGENTS.md §3.5)."""

    def test_classifier_vs_anomaly_explanation_tagging(self, sample_evidence_context):
        tx, ml_score, _, _ = sample_evidence_context
        ledger = build_evidence_ledger(
            entity_id="tx_test_tagging",
            entity_type="transaction",
            tx=tx,
            ml_score=ml_score,
            anomaly_score=0.82,
        )

        ml_records = ledger.get_by_category(EvidenceCategory.ML_BEHAVIORAL)
        assert len(ml_records) == 1
        assert ml_records[0].explanation_type == ExplanationType.CLASSIFIER_EXPLANATION
        assert "Supervised model" in ml_records[0].description
        assert "illicit probability" in ml_records[0].description

        anom_records = ledger.get_by_category(EvidenceCategory.ANOMALY)
        assert len(anom_records) == 1
        assert anom_records[0].explanation_type == ExplanationType.ANOMALY_EXPLANATION
        assert "statistical novelty" in anom_records[0].description.lower()
        assert "not confirmed illicit" in anom_records[0].description.lower()

    def test_classifier_explanation_rationale(self):
        # High probability rationale
        ml_high = MLScore(
            entity_id="tx_high",
            entity_type="transaction",
            prediction=1,
            probability=0.92,
            model_version="xgb_v1",
        )
        ledger_high = build_evidence_ledger(
            entity_id="tx_high",
            ml_score=ml_high,
        )
        rec_high = ledger_high.get_by_category(EvidenceCategory.ML_BEHAVIORAL)[0]
        assert "high illicit probability" in rec_high.description

        # Low probability rationale
        ml_low = MLScore(
            entity_id="tx_low",
            entity_type="transaction",
            prediction=2,
            probability=0.08,
            model_version="xgb_v1",
        )
        ledger_low = build_evidence_ledger(
            entity_id="tx_low",
            ml_score=ml_low,
        )
        rec_low = ledger_low.get_by_category(EvidenceCategory.ML_BEHAVIORAL)[0]
        assert "low illicit probability" in rec_low.description

    def test_anomaly_explanation_never_implies_illicitness(self):
        ledger = build_evidence_ledger(
            entity_id="tx_novelty",
            anomaly_score=0.95,
        )
        rec = ledger.get_by_category(EvidenceCategory.ANOMALY)[0]
        assert rec.explanation_type == ExplanationType.ANOMALY_EXPLANATION
        # Explicit disclaimer required per AGENTS.md §3.5
        assert "not confirmed illicit" in rec.description.lower()
        assert "unusual behavior, not guilt" in rec.description.lower()

    def test_narrative_explanation_structural_separation(self, sample_evidence_context):
        tx, ml_score, _, _ = sample_evidence_context
        ledger = build_evidence_ledger(
            entity_id="tx_dual",
            entity_type="transaction",
            tx=tx,
            ml_score=ml_score,
            anomaly_score=0.75,
        )
        narrative = generate_narrative_explanation(ledger)
        # Verify clear separation of classifier finding vs anomaly deviance
        assert "Classifier Findings (why the model assigned this label)" in narrative
        assert "Anomaly Findings (statistical novelty, NOT proof of illicit activity)" in narrative
        assert "statistical novelty" in narrative.lower()


class TestConfigurablePriorityTiers:
    """Tests for configurable investigative priority tiers."""

    def test_default_tier_config_thresholds(self):
        cfg = DEFAULT_TIER_CONFIG
        assert cfg.determine_tier(95.0).tier == PriorityTier.CRITICAL
        assert cfg.determine_tier(80.0).tier == PriorityTier.CRITICAL
        assert cfg.determine_tier(79.9).tier == PriorityTier.HIGH
        assert cfg.determine_tier(60.0).tier == PriorityTier.HIGH
        assert cfg.determine_tier(59.9).tier == PriorityTier.MEDIUM
        assert cfg.determine_tier(40.0).tier == PriorityTier.MEDIUM
        assert cfg.determine_tier(39.9).tier == PriorityTier.LOW
        assert cfg.determine_tier(0.0).tier == PriorityTier.LOW

    def test_custom_priority_tier_config(self):
        # Strict custom tiers: CRITICAL ≥ 90, HIGH ≥ 75, MEDIUM ≥ 50, LOW < 50
        custom_tier_config = PriorityTierConfig(
            tiers=(
                PriorityTierDefinition(
                    tier=PriorityTier.CRITICAL,
                    min_score=90.0,
                    description="Immediate action required (strict threshold)",
                ),
                PriorityTierDefinition(
                    tier=PriorityTier.HIGH,
                    min_score=75.0,
                    description="Elevated priority review",
                ),
                PriorityTierDefinition(
                    tier=PriorityTier.MEDIUM,
                    min_score=50.0,
                    description="Secondary queue review",
                ),
                PriorityTierDefinition(
                    tier=PriorityTier.LOW,
                    min_score=0.0,
                    description="Informational only",
                ),
            )
        )

        assert custom_tier_config.determine_tier(85.0).tier == PriorityTier.HIGH
        assert custom_tier_config.determine_tier(91.0).tier == PriorityTier.CRITICAL
        assert (
            custom_tier_config.determine_tier(91.0).description
            == "Immediate action required (strict threshold)"
        )

        synth_cfg = SynthesisConfig(tier_config=custom_tier_config)
        assert determine_priority_tier(85.0, synth_cfg) == PriorityTier.HIGH
        assert determine_priority_tier(91.0, synth_cfg) == PriorityTier.CRITICAL

    def test_synthesize_with_custom_tier_config_and_description(self):
        custom_tier_config = PriorityTierConfig(
            tiers=(
                PriorityTierDefinition(
                    tier=PriorityTier.CRITICAL,
                    min_score=95.0,
                    description="Highest echelon critical incident",
                ),
                PriorityTierDefinition(
                    tier=PriorityTier.HIGH,
                    min_score=70.0,
                    description="High priority triage required",
                ),
                PriorityTierDefinition(
                    tier=PriorityTier.LOW,
                    min_score=0.0,
                    description="Baseline routine triage",
                ),
            )
        )
        custom_cfg = SynthesisConfig(tier_config=custom_tier_config)

        # Single illicit signal at 0.85 -> score 85.0 -> would be CRITICAL under default,
        # but HIGH under custom (since custom CRITICAL requires 95.0)
        signals = SignalInput(
            entity_id="tx_tiered",
            entity_type="transaction",
            illicit_probability=0.85,
        )

        risk_score = synthesize_risk_score(signals, custom_cfg)
        assert risk_score.score == 85.0
        assert risk_score.priority_tier == PriorityTier.HIGH
        assert risk_score.tier_description == "High priority triage required"


class TestEntityAggregationTraceability:
    """Tests for traceable transaction-to-wallet/entity aggregation (AGENTS.md §3.7)."""

    def _sample_tx_scores(self) -> list[RiskScore]:
        return [
            RiskScore(
                entity_id="tx_001",
                entity_type="transaction",
                score=88.0,
                priority_tier=PriorityTier.CRITICAL,
                behavioral_signal=0.90,
                anomaly_signal=0.85,
                contains_synthetic_input=False,
                explanation="High-risk transaction with multi-signal corroboration.",
            ),
            RiskScore(
                entity_id="tx_002",
                entity_type="transaction",
                score=45.0,
                priority_tier=PriorityTier.MEDIUM,
                behavioral_signal=0.40,
                anomaly_signal=0.50,
                contains_synthetic_input=True,
                explanation="Moderate risk with synthetic network observation.",
            ),
            RiskScore(
                entity_id="tx_003",
                entity_type="transaction",
                score=15.0,
                priority_tier=PriorityTier.LOW,
                behavioral_signal=0.10,
                anomaly_signal=0.20,
                contains_synthetic_input=False,
                explanation="Routine low-risk transaction.",
            ),
        ]

    def test_aggregation_max_method(self):
        tx_scores = self._sample_tx_scores()
        agg = aggregate_transaction_scores(
            entity_id="wallet_alpha",
            transaction_scores=tx_scores,
            method=AggregationMethod.MAX,
        )

        assert agg.entity_id == "wallet_alpha"
        assert agg.aggregation_method == AggregationMethod.MAX
        assert agg.aggregated_score == 88.0
        assert agg.priority_tier == PriorityTier.CRITICAL
        assert agg.transaction_count == 3
        assert agg.flagged_transaction_count == 2  # tx_001 (88.0) and tx_002 (45.0) >= 40
        assert agg.aggregation_metadata["max_contributing_tx"] == "tx_001"
        assert len(agg.contributing_transactions) == 3
        assert agg.contributing_transactions[0].txid == "tx_001"
        assert agg.contributing_transactions[0].risk_score == 88.0

    def test_aggregation_mean_method(self):
        tx_scores = self._sample_tx_scores()
        agg = aggregate_transaction_scores(
            entity_id="wallet_alpha",
            transaction_scores=tx_scores,
            method=AggregationMethod.MEAN,
        )

        expected_mean = round((88.0 + 45.0 + 15.0) / 3.0, 2)  # 49.33
        assert agg.aggregated_score == expected_mean
        assert agg.priority_tier == PriorityTier.MEDIUM
        assert agg.aggregation_metadata["sum_scores"] == 148.0

    def test_aggregation_volume_weighted_method(self):
        tx_scores = self._sample_tx_scores()
        volumes = {
            "tx_001": 0.5,   # small illicit tx: score 88.0 * 0.5 = 44.0
            "tx_002": 1.0,   # moderate tx: score 45.0 * 1.0 = 45.0
            "tx_003": 8.5,   # large licit tx: score 15.0 * 8.5 = 127.5
        }
        # Total volume = 10.0, weighted sum = 216.5, weighted score = 21.65
        agg = aggregate_transaction_scores(
            entity_id="wallet_alpha",
            transaction_scores=tx_scores,
            method=AggregationMethod.VOLUME_WEIGHTED,
            volume_weights=volumes,
        )

        assert agg.aggregated_score == 21.65
        assert agg.priority_tier == PriorityTier.LOW
        assert agg.aggregation_metadata["total_volume"] == 10.0

    def test_aggregation_frequency_method(self):
        tx_scores = self._sample_tx_scores()
        # With threshold 50.0: only tx_001 (88.0) qualifies -> 1 / 3 = 33.33%
        agg = aggregate_transaction_scores(
            entity_id="wallet_alpha",
            transaction_scores=tx_scores,
            method=AggregationMethod.FREQUENCY,
            frequency_threshold=50.0,
        )

        assert agg.aggregated_score == 33.33
        assert agg.priority_tier == PriorityTier.LOW
        assert agg.aggregation_metadata["flagged_count"] == 1

    def test_non_accusation_principle_and_forensic_disclaimer(self):
        """A single illicit transaction must NOT become a blanket accusation against the wallet."""
        tx_scores = self._sample_tx_scores()
        agg = aggregate_transaction_scores(
            entity_id="wallet_alpha",
            transaction_scores=tx_scores,
            method=AggregationMethod.MEAN,
        )

        # Mandatory disclaimer per AGENTS.md §3.7
        assert "blanket accusations" in agg.forensic_disclaimer.lower()
        assert "investigative prioritization" in agg.forensic_disclaimer.lower()

        # Under MEAN aggregation, wallet priority is MEDIUM even though one tx is CRITICAL
        assert agg.priority_tier == PriorityTier.MEDIUM
        assert agg.aggregated_score < 50.0

    def test_synthetic_provenance_propagation(self):
        tx_scores = self._sample_tx_scores()
        agg = aggregate_transaction_scores(
            entity_id="wallet_alpha",
            transaction_scores=tx_scores,
        )
        # tx_002 has synthetic data -> entity aggregation must carry contains_synthetic_input=True
        assert agg.contains_synthetic_input is True

    def test_traceability_with_evidence_ledgers(self):
        tx_scores = self._sample_tx_scores()
        tx = Transaction(txid=1001, time_step=1, total_btc=5.0)
        ml_score = MLScore(
            entity_id="tx_001",
            entity_type="transaction",
            prediction=1,
            probability=0.90,
            model_version="xgb_v1",
        )
        ledger = build_evidence_ledger(
            entity_id="tx_001",
            entity_type="transaction",
            tx=tx,
            ml_score=ml_score,
        )

        agg = aggregate_transaction_scores(
            entity_id="wallet_alpha",
            transaction_scores=tx_scores,
            transaction_ledgers={"tx_001": ledger},
        )

        # First contribution (tx_001) should have summary from ledger
        c0 = agg.contributing_transactions[0]
        assert c0.txid == "tx_001"
        assert c0.evidence_summary is not None
        assert "Supervised Classification" in c0.evidence_summary

    def test_aggregation_validation_errors(self):
        with pytest.raises(ValueError, match="no transaction scores"):
            aggregate_transaction_scores(entity_id="w1", transaction_scores=[])

        tx_scores = self._sample_tx_scores()
        with pytest.raises(ValueError, match="volume_weights"):
            aggregate_transaction_scores(
                entity_id="w1",
                transaction_scores=tx_scores,
                method=AggregationMethod.VOLUME_WEIGHTED,
                volume_weights=None,
            )
