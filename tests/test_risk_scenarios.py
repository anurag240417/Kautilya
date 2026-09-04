"""Investigative forensic scenario validations (Phase 7).

Validates end-to-end risk scoring, explainability, entity aggregation,
and alert prioritization across realistic forensic investigative scenarios:

Scenario 1: Multi-signal corroboration boost (Ransomware / illicit cashout)
Scenario 2: Isolated high anomaly deviance (Novelty != Illicitness, safety cap)
Scenario 3: High licit confidence (Confidence in licit != Risk score)
Scenario 4: High-traffic commercial wallet with single rogue deposit (Non-accusation)
Scenario 5: Structured micro-payments / smurfing pattern (Frequency aggregation)
Scenario 6: Synthetic network & temporal correlation provenance tracking
Scenario 7: Open-ended custom investigative signals extensibility
Scenario 8: Dynamic normalization missing-signal invariance
"""

from datetime import UTC, datetime, timedelta

from backend.domain.alert import AlertFilter
from backend.domain.correlation import (
    CandidateIPRole,
    TemporalCorrelation,
    TransactionIPCorrelation,
)
from backend.domain.ml import MLScore
from backend.domain.risk import (
    EvidenceCategory,
    RiskScore,
    SignalInput,
)
from backend.domain.transaction import Transaction
from backend.domain.types import EntityClass, ExplanationType, PriorityTier
from backend.risk.aggregation import (
    AggregationMethod,
    aggregate_transaction_scores,
)
from backend.risk.evidence import (
    build_evidence_ledger,
    generate_narrative_explanation,
)
from backend.risk.ranking import (
    filter_and_prioritize_alerts,
    generate_alert,
    generate_entity_alert,
    rank_entities,
)
from backend.risk.scorer import (
    DEFAULT_SYNTHESIS_CONFIG,
    SynthesisConfig,
    SynthesisPolicy,
    synthesize_risk_score,
)


class TestScenario1MultiSignalCorroboration:
    """Scenario 1: High supervised probability corroborated by graph and anomaly signals."""

    def test_corroborated_illicit_cashout_triggers_critical_alert(self):
        tx = Transaction(
            txid=554433,
            time_step=38,
            total_btc=85.25,
            fees=0.012,
            num_input_addresses=4,
            num_output_addresses=2,
            in_txs_degree=6,
            out_txs_degree=12,
        )
        ml_score = MLScore(
            entity_id="554433",
            entity_type="transaction",
            prediction=1,
            probability=0.88,
            model_version="xgb_M2_v1",
        )
        signals = SignalInput(
            entity_id="554433",
            entity_type="transaction",
            illicit_probability=0.88,
            predicted_label=EntityClass.ILLICIT,
            anomaly_score=0.72,
            graph_signal=0.68,
        )

        # 1. Synthesize risk score
        risk = synthesize_risk_score(signals)
        # Corroborated tier policy should apply 1.15x boost
        assert risk.score >= 80.0
        assert risk.priority_tier == PriorityTier.CRITICAL

        # 2. Build structured Evidence Ledger
        ledger = build_evidence_ledger(
            entity_id="554433",
            entity_type="transaction",
            tx=tx,
            ml_score=ml_score,
            anomaly_score=0.72,
            graph_metrics={"pagerank": 0.008, "betweenness": 0.015},
        )
        assert len(ledger.records) == 3
        # Verify classifier explanation tagged
        ml_rec = ledger.get_by_category(EvidenceCategory.ML_BEHAVIORAL)[0]
        assert ml_rec.explanation_type == ExplanationType.CLASSIFIER_EXPLANATION
        assert "high illicit probability" in ml_rec.description

        # Verify no anonymized feature leaks
        narrative = generate_narrative_explanation(ledger)
        assert "Local_feature" not in narrative
        assert "Aggregate_feature" not in narrative
        assert "85.2500 BTC" in narrative

        # 3. Generate Alert and check triage ranking
        alert = generate_alert(risk, evidence_ledger=ledger)
        assert alert is not None
        assert alert.priority_tier == PriorityTier.CRITICAL
        assert "Immediate investigator review required" in alert.recommended_action
        assert "[CRITICAL]" in alert.headline

        # 4. Triage queue ranking
        ranked = rank_entities([risk])
        assert ranked[0].rank == 1
        assert ranked[0].priority_tier == PriorityTier.CRITICAL
        assert ranked[0].corroboration_count == 3


class TestScenario2IsolatedAnomalyDeviance:
    """Scenario 2: High statistical anomaly deviance without supervised corroboration.

    Demonstrates that statistical novelty != illicitness, enforcing the safety cap.
    """

    def test_isolated_anomaly_safety_cap_prevents_critical_alert(self):
        # Transaction with very high anomaly score (0.94) without supervised or graph corroboration
        signals = SignalInput(
            entity_id="tx_novel_whale",
            entity_type="transaction",
            anomaly_score=0.94,
        )

        risk = synthesize_risk_score(signals)
        # Must be capped at uncorroborated_anomaly_cap (60.0)
        assert risk.score == DEFAULT_SYNTHESIS_CONFIG.uncorroborated_anomaly_cap
        # Must NOT be CRITICAL tier
        assert risk.priority_tier != PriorityTier.CRITICAL
        assert risk.priority_tier == PriorityTier.HIGH

        # Build evidence ledger
        ledger = build_evidence_ledger(
            entity_id="tx_novel_whale",
            anomaly_score=0.94,
        )
        anom_rec = ledger.get_by_category(EvidenceCategory.ANOMALY)[0]
        assert anom_rec.explanation_type == ExplanationType.ANOMALY_EXPLANATION
        # Must explicitly disclaim illicitness per AGENTS.md §3.5
        assert "not confirmed illicit" in anom_rec.description.lower()
        assert "unusual behavior, not guilt" in anom_rec.description.lower()

        # Generate alert
        alert = generate_alert(risk, evidence_ledger=ledger)
        assert alert is not None
        assert alert.priority_tier != PriorityTier.CRITICAL
        # Triage guidance should be secondary or elevated review, NOT immediate critical action
        assert "Immediate investigator review required" not in alert.recommended_action


class TestScenario3HighLicitConfidence:
    """Scenario 3: Model 99% confident in Licit behavior must NOT be confused with 99% risk."""

    def test_high_licit_confidence_produces_low_risk(self):
        # Classifier is 99% confident in Licit -> illicit_probability is 0.01
        signals = SignalInput(
            entity_id="tx_routine_payment",
            entity_type="transaction",
            illicit_probability=0.01,
            predicted_label=EntityClass.LICIT,
            anomaly_score=0.05,
        )

        risk = synthesize_risk_score(signals)
        # Score must be very low (< 5.0) and in LOW tier
        assert risk.score < 5.0
        assert risk.priority_tier == PriorityTier.LOW

        # An alert queue filtering for MEDIUM or above must suppress this
        alert = generate_alert(risk, min_alert_tier=PriorityTier.MEDIUM)
        assert alert is None

        # Build evidence ledger
        ml_score = MLScore(
            entity_id="tx_routine_payment",
            entity_type="transaction",
            prediction=2,
            probability=0.01,
            model_version="xgb_v1",
        )
        ledger = build_evidence_ledger(
            entity_id="tx_routine_payment",
            ml_score=ml_score,
        )
        rec = ledger.get_by_category(EvidenceCategory.ML_BEHAVIORAL)[0]
        assert "low illicit probability" in rec.description


class TestScenario4CommercialWalletWithRogueDeposit:
    """Scenario 4: High-traffic commercial wallet with an isolated tainted deposit.

    Enforces the non-accusation principle (AGENTS.md §3.7).
    """

    def test_commercial_wallet_non_accusation_and_traceability(self):
        # 100 transactions: 99 legitimate commercial transactions (score 12.0)
        # plus 1 ransomware deposit (score 88.0)
        licit_txs = [
            RiskScore(
                entity_id=f"tx_cust_{i:03d}",
                entity_type="transaction",
                score=12.0,
                priority_tier=PriorityTier.LOW,
                explanation="Routine commercial customer deposit.",
            )
            for i in range(1, 100)
        ]
        rogue_tx = RiskScore(
            entity_id="tx_ransom_deposit",
            entity_type="transaction",
            score=88.0,
            priority_tier=PriorityTier.CRITICAL,
            explanation="High-risk deposit from suspected ransomware cluster.",
        )
        all_txs = licit_txs + [rogue_tx]

        # 1. Under MEAN aggregation: the wallet overall priority is LOW
        agg_mean = aggregate_transaction_scores(
            entity_id="wallet_exchange_hot",
            transaction_scores=all_txs,
            method=AggregationMethod.MEAN,
        )
        # (99 * 12.0 + 88.0) / 100 = (1188 + 88) / 100 = 12.76 -> LOW tier
        assert agg_mean.aggregated_score < 15.0
        assert agg_mean.priority_tier == PriorityTier.LOW

        # 2. Under MAX aggregation: the wallet score is 88.0, BUT:
        agg_max = aggregate_transaction_scores(
            entity_id="wallet_exchange_hot",
            transaction_scores=all_txs,
            method=AggregationMethod.MAX,
        )
        assert agg_max.aggregated_score == 88.0
        assert agg_max.priority_tier == PriorityTier.CRITICAL

        # Non-accusation principle verification:
        assert "blanket accusations" in agg_max.forensic_disclaimer.lower()
        assert agg_max.flagged_transaction_count == 1
        assert agg_max.transaction_count == 100

        # Full traceability: top contributing transaction is permanently recorded
        top_contrib = agg_max.contributing_transactions[0]
        assert top_contrib.txid == "tx_ransom_deposit"
        assert top_contrib.risk_score == 88.0

        # 3. Generate entity alert
        alert = generate_entity_alert(agg_max)
        assert alert is not None
        flag_summary = "1 transaction(s) individually exceeded the medium priority threshold"
        assert flag_summary in alert.summary
        assert "tx_ransom_deposit" in alert.summary
        assert "blanket accusations" in alert.forensic_disclaimer.lower()


class TestScenario5StructuringSmurfingPattern:
    """Scenario 5: Repeated small structured micro-payments evaluated via FREQUENCY aggregation."""

    def test_structuring_pattern_elevated_by_frequency(self):
        # 10 small transactions, each with score ~48.0 (individually MEDIUM)
        structured_txs = [
            RiskScore(
                entity_id=f"tx_smurf_{i:02d}",
                entity_type="transaction",
                score=48.0,
                priority_tier=PriorityTier.MEDIUM,
                explanation="Sub-threshold structured transfer.",
            )
            for i in range(10)
        ]

        # Under MEAN: average is 48.0 (MEDIUM tier)
        agg_mean = aggregate_transaction_scores(
            entity_id="wallet_smurf",
            transaction_scores=structured_txs,
            method=AggregationMethod.MEAN,
        )
        assert agg_mean.aggregated_score == 48.0
        assert agg_mean.priority_tier == PriorityTier.MEDIUM

        # Under FREQUENCY with threshold 40.0: 10 / 10 = 100% flagged -> score 100.0 (CRITICAL)
        agg_freq = aggregate_transaction_scores(
            entity_id="wallet_smurf",
            transaction_scores=structured_txs,
            method=AggregationMethod.FREQUENCY,
            frequency_threshold=40.0,
        )
        assert agg_freq.aggregated_score == 100.0
        assert agg_freq.priority_tier == PriorityTier.CRITICAL
        assert agg_freq.flagged_transaction_count == 10

        alert = generate_entity_alert(agg_freq)
        assert alert is not None
        assert alert.priority_tier == PriorityTier.CRITICAL


class TestScenario6SyntheticProvenanceTracking:
    """Scenario 6: End-to-end synthetic network & temporal correlation provenance tracking."""

    def test_synthetic_provenance_propagates_to_alert_and_queue(self):
        t_base = datetime(2019, 6, 17, 12, 0, 0, tzinfo=UTC)

        temp_corr = TemporalCorrelation(
            txid=990011,
            time_step=15,
            observation_timestamp=t_base + timedelta(minutes=2),
            window_start=t_base,
            window_end=t_base + timedelta(hours=2),
            delta_seconds=120.0,
            temporal_proximity=0.95,
            is_synthetic=True,
        )
        tx_ip_corr = TransactionIPCorrelation(
            txid=990011,
            ip="198.51.100.22",
            role=CandidateIPRole.ORIGIN,
            correlation_confidence=0.88,
            metrics={
                "temporal_proximity": 0.95,
                "propagation_consistency": 0.90,
                "topology_consistency": 0.85,
                "composite_confidence": 0.88,
            },
            observations_count=1,
            earliest_observation=t_base,
            latest_observation=t_base + timedelta(minutes=2),
            is_synthetic=True,
        )

        # Build ledger
        ledger = build_evidence_ledger(
            entity_id="990011",
            temporal_correlations=[temp_corr],
            tx_ip_correlations=[tx_ip_corr],
        )
        assert len(ledger.records) == 2
        assert all(r.is_synthetic for r in ledger.records)

        # Synthesize risk score
        signals = SignalInput(
            entity_id="990011",
            entity_type="transaction",
            correlation_confidence=0.88,
            contains_synthetic_input=True,
        )
        risk = synthesize_risk_score(signals)
        assert risk.contains_synthetic_input is True

        # Aggregate wallet
        agg = aggregate_transaction_scores(
            entity_id="wallet_syn_test",
            transaction_scores=[risk],
        )
        assert agg.contains_synthetic_input is True

        # Generate alert
        alert = generate_alert(risk, evidence_ledger=ledger)
        assert alert is not None
        assert alert.contains_synthetic_input is True

        # Rank queue
        ranked = rank_entities([risk])
        assert ranked[0].contains_synthetic_input is True

        # Filter queue: excluding synthetic removes this alert
        filtered = filter_and_prioritize_alerts([alert], AlertFilter(include_synthetic=False))
        assert len(filtered) == 0


class TestScenario7CustomSignalsExtensibility:
    """Scenario 7: Open-ended custom domain signals extensibility."""

    def test_custom_signals_influence_score_without_schema_mutation(self):
        signals = SignalInput(
            entity_id="tx_darknet_mixer",
            entity_type="transaction",
            illicit_probability=0.40,
            custom_signals={
                "darknet_vendor_interaction": 0.90,
                "mixer_hop_depth": 0.85,
            },
        )
        custom_cfg = SynthesisConfig(
            policy=SynthesisPolicy.DYNAMIC_NORMALIZED,
            custom_weights={
                "darknet_vendor_interaction": 0.40,
                "mixer_hop_depth": 0.30,
            },
        )

        risk = synthesize_risk_score(signals, config=custom_cfg)
        # Expected: (0.40 * 0.35 + 0.90 * 0.40 + 0.85 * 0.30) / (0.35 + 0.40 + 0.30)
        expected_num = 0.40 * 0.35 + 0.90 * 0.40 + 0.85 * 0.30
        expected_den = 0.35 + 0.40 + 0.30
        expected = round(expected_num / expected_den * 100.0, 2)
        assert risk.score == expected
        assert risk.priority_tier == PriorityTier.HIGH
        assert "darknet_vendor_interaction" in risk.active_signals
        assert "mixer_hop_depth" in risk.active_signals


class TestScenario8DynamicNormalizationInvariance:
    """Scenario 8: Dynamic normalization invariance.

    Ensures missing signals do not artificially depress priority.
    """

    def test_missing_signals_do_not_depress_priority(self):
        # Transaction with only a single strong supervised signal (0.80)
        signals_solo = SignalInput(
            entity_id="tx_solo",
            entity_type="transaction",
            illicit_probability=0.80,
        )
        cfg = SynthesisConfig(policy=SynthesisPolicy.DYNAMIC_NORMALIZED)
        risk_solo = synthesize_risk_score(signals_solo, config=cfg)
        # Should be exactly 80.0
        assert risk_solo.score == 80.0
        assert risk_solo.priority_tier == PriorityTier.CRITICAL

        # Transaction with supervised (0.80) and licit anomaly (0.20)
        signals_dual = SignalInput(
            entity_id="tx_dual",
            entity_type="transaction",
            illicit_probability=0.80,
            anomaly_score=0.20,
        )
        risk_dual = synthesize_risk_score(signals_dual, config=cfg)
        # (0.80 * 0.35 + 0.20 * 0.20) / (0.35 + 0.20) = (0.28 + 0.04) / 0.55 = 0.32 / 0.55 = 58.18
        assert risk_dual.score == round((0.80 * 0.35 + 0.20 * 0.20) / (0.35 + 0.20) * 100.0, 2)
