"""Tests for entity ranking and alert prioritization (Phase 7).

Verifies:
1. Investigative alert generation from RiskScore (transaction-level).
2. Investigative alert generation from EntityAggregation (wallet-level),
   enforcing the non-accusation principle and forensic disclaimers.
3. Alert filtering by priority tier, status, entity type, and synthetic provenance.
4. Entity queue ranking (primary sort by score descending, secondary sort
   by corroboration count descending).
5. Actionable investigator triage recommendations per priority tier.
"""

from backend.domain.alert import AlertFilter, RankingCriteria
from backend.domain.risk import EvidenceCategory, EvidenceLedger, EvidenceRecord, RiskScore
from backend.domain.types import AlertStatus, EvidenceType, PriorityTier
from backend.risk.aggregation import (
    AggregationMethod,
    EntityAggregation,
    TransactionContribution,
)
from backend.risk.ranking import (
    filter_and_prioritize_alerts,
    generate_alert,
    generate_entity_alert,
    rank_entities,
)


class TestAlertGeneration:
    """Tests for individual transaction and entity alert generation."""

    def test_generate_alert_for_transaction_critical(self):
        risk = RiskScore(
            entity_id="tx_crit_001",
            entity_type="transaction",
            score=88.5,
            priority_tier=PriorityTier.CRITICAL,
            tier_description="Requires immediate investigator attention",
            active_signals=["illicit_probability", "graph_signal", "anomaly_score"],
            explanation="High illicit probability corroborated by graph centrality.",
            contains_synthetic_input=False,
        )

        alert = generate_alert(risk)
        assert alert is not None
        assert alert.entity_id == "tx_crit_001"
        assert alert.entity_type == "transaction"
        assert alert.risk_score == 88.5
        assert alert.priority_tier == PriorityTier.CRITICAL
        assert alert.status == AlertStatus.NEW
        assert "[CRITICAL]" in alert.headline
        assert "Immediate investigator review required" in alert.recommended_action
        assert alert.contains_synthetic_input is False
        assert "investigative triage prioritization" in alert.forensic_disclaimer.lower()

    def test_generate_alert_below_min_tier_returns_none(self):
        risk = RiskScore(
            entity_id="tx_low_001",
            entity_type="transaction",
            score=25.0,
            priority_tier=PriorityTier.LOW,
            active_signals=["illicit_probability"],
            explanation="Routine transaction.",
        )

        # Default min_alert_tier is MEDIUM -> LOW should return None
        assert generate_alert(risk) is None

        # Explicitly allowing LOW returns the alert
        alert = generate_alert(risk, min_alert_tier=PriorityTier.LOW)
        assert alert is not None
        assert alert.priority_tier == PriorityTier.LOW
        assert "Routine monitoring" in alert.recommended_action

    def test_generate_alert_with_evidence_ledger(self):
        risk = RiskScore(
            entity_id="tx_ev_001",
            entity_type="transaction",
            score=72.0,
            priority_tier=PriorityTier.HIGH,
            active_signals=["illicit_probability"],
            explanation="High behavioral probability.",
        )
        ledger = EvidenceLedger(entity_id="tx_ev_001", entity_type="transaction")
        ledger.add_record(
            EvidenceRecord(
                evidence_id="ev_01",
                category=EvidenceCategory.ML_BEHAVIORAL,
                evidence_type=EvidenceType.MODEL_PREDICTION,
                source_entity_id="tx_ev_001",
                source_entity_type="transaction",
                headline="Supervised Classification: Illicit (72.0%)",
                description="High likelihood of illicit behavior.",
            )
        )
        ledger.add_record(
            EvidenceRecord(
                evidence_id="ev_02",
                category=EvidenceCategory.GRAPH_STRUCTURAL,
                evidence_type=EvidenceType.OBSERVATION,
                source_entity_id="tx_ev_001",
                source_entity_type="transaction",
                headline="In-degree of 6 transaction inputs",
                description="Dense input fan-in pattern.",
            )
        )

        alert = generate_alert(risk, evidence_ledger=ledger)
        assert alert is not None
        assert alert.evidence_count == 2
        assert "Supervised Classification" in alert.summary
        assert "Elevated priority review" in alert.recommended_action

    def test_generate_alert_synthetic_provenance_propagation(self):
        risk = RiskScore(
            entity_id="tx_syn_001",
            entity_type="transaction",
            score=65.0,
            priority_tier=PriorityTier.HIGH,
            active_signals=["correlation_confidence"],
            contains_synthetic_input=True,
        )

        alert = generate_alert(risk)
        assert alert is not None
        assert alert.contains_synthetic_input is True


class TestEntityAlertGeneration:
    """Tests for wallet/cluster aggregated alert generation."""

    def test_generate_entity_alert_for_wallet(self):
        contributions = [
            TransactionContribution(
                txid="tx_101",
                risk_score=92.0,
                priority_tier=PriorityTier.CRITICAL,
                illicit_probability=0.95,
            ),
            TransactionContribution(
                txid="tx_102",
                risk_score=50.0,
                priority_tier=PriorityTier.MEDIUM,
                illicit_probability=0.45,
            ),
        ]
        agg = EntityAggregation(
            entity_id="1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa",
            entity_type="wallet",
            aggregated_score=71.0,
            priority_tier=PriorityTier.HIGH,
            tier_description="Elevated priority review required",
            aggregation_method=AggregationMethod.MEAN,
            transaction_count=2,
            flagged_transaction_count=2,
            contributing_transactions=contributions,
            contains_synthetic_input=False,
        )

        alert = generate_entity_alert(agg)
        assert alert is not None
        assert alert.entity_id == "1A1zP1eP5QGefi2DMPTfTL5SLmv7DivfNa"
        assert alert.entity_type == "wallet"
        assert alert.risk_score == 71.0
        assert alert.priority_tier == PriorityTier.HIGH
        assert "[HIGH] Wallet" in alert.headline
        assert "mean" in alert.headline
        assert "tx_101" in alert.summary
        assert alert.evidence_count == 2
        assert "blanket accusations" in alert.forensic_disclaimer.lower()

    def test_generate_entity_alert_below_min_tier_returns_none(self):
        agg = EntityAggregation(
            entity_id="wallet_routine",
            entity_type="wallet",
            aggregated_score=20.0,
            priority_tier=PriorityTier.LOW,
            aggregation_method=AggregationMethod.MEAN,
            transaction_count=5,
        )
        assert generate_entity_alert(agg, min_alert_tier=PriorityTier.MEDIUM) is None


class TestEntityRanking:
    """Tests for entity triage queue ranking."""

    def test_rank_entities_score_descending(self):
        scores = [
            RiskScore(
                entity_id="tx_mid",
                entity_type="transaction",
                score=55.0,
                priority_tier=PriorityTier.MEDIUM,
                active_signals=["anomaly_score"],
            ),
            RiskScore(
                entity_id="tx_high",
                entity_type="transaction",
                score=85.0,
                priority_tier=PriorityTier.CRITICAL,
                active_signals=["illicit_probability", "graph_signal"],
            ),
            RiskScore(
                entity_id="tx_low",
                entity_type="transaction",
                score=15.0,
                priority_tier=PriorityTier.LOW,
                active_signals=[],
            ),
        ]

        ranked = rank_entities(scores)
        assert len(ranked) == 3
        assert ranked[0].rank == 1
        assert ranked[0].entity_id == "tx_high"
        assert ranked[0].risk_score == 85.0

        assert ranked[1].rank == 2
        assert ranked[1].entity_id == "tx_mid"
        assert ranked[1].risk_score == 55.0

        assert ranked[2].rank == 3
        assert ranked[2].entity_id == "tx_low"
        assert ranked[2].risk_score == 15.0

    def test_rank_entities_secondary_sort_by_corroboration(self):
        # Two entities with identical score (70.0), but different corroboration counts
        entity_a = RiskScore(
            entity_id="tx_single_signal",
            entity_type="transaction",
            score=70.0,
            priority_tier=PriorityTier.HIGH,
            active_signals=["illicit_probability"],
        )
        entity_b = RiskScore(
            entity_id="tx_multi_signal",
            entity_type="transaction",
            score=70.0,
            priority_tier=PriorityTier.HIGH,
            active_signals=["illicit_probability", "graph_signal", "anomaly_score"],
        )

        ranked = rank_entities([entity_a, entity_b])
        # Multi-signal entity must take precedence at same score
        assert ranked[0].entity_id == "tx_multi_signal"
        assert ranked[0].corroboration_count == 3
        assert ranked[1].entity_id == "tx_single_signal"
        assert ranked[1].corroboration_count == 1

    def test_rank_entities_with_aggregations_and_criteria(self):
        tx = RiskScore(
            entity_id="tx_01",
            entity_type="transaction",
            score=90.0,
            priority_tier=PriorityTier.CRITICAL,
            active_signals=["illicit_probability"],
        )
        agg_wallet = EntityAggregation(
            entity_id="wallet_01",
            entity_type="wallet",
            aggregated_score=75.0,
            priority_tier=PriorityTier.HIGH,
            aggregation_method=AggregationMethod.MAX,
            transaction_count=4,
            flagged_transaction_count=2,
        )
        low_tx = RiskScore(
            entity_id="tx_02",
            entity_type="transaction",
            score=30.0,
            priority_tier=PriorityTier.LOW,
        )

        criteria = RankingCriteria(min_tier=PriorityTier.HIGH, limit=2)
        ranked = rank_entities([low_tx, agg_wallet, tx], criteria=criteria)

        # low_tx filtered out by min_tier=HIGH
        assert len(ranked) == 2
        assert ranked[0].entity_id == "tx_01"
        assert ranked[0].rank == 1
        assert ranked[1].entity_id == "wallet_01"
        assert ranked[1].rank == 2


class TestAlertPrioritizationAndFiltering:
    """Tests for filtering and prioritizing investigator alert queues."""

    def _sample_alerts(self) -> list:
        return [
            generate_alert(
                RiskScore(
                    entity_id="tx_med",
                    entity_type="transaction",
                    score=48.0,
                    priority_tier=PriorityTier.MEDIUM,
                )
            ),
            generate_alert(
                RiskScore(
                    entity_id="tx_crit",
                    entity_type="transaction",
                    score=92.0,
                    priority_tier=PriorityTier.CRITICAL,
                )
            ),
            generate_alert(
                RiskScore(
                    entity_id="tx_high",
                    entity_type="transaction",
                    score=75.0,
                    priority_tier=PriorityTier.HIGH,
                )
            ),
            generate_alert(
                RiskScore(
                    entity_id="tx_syn",
                    entity_type="transaction",
                    score=65.0,
                    priority_tier=PriorityTier.HIGH,
                    contains_synthetic_input=True,
                )
            ),
        ]

    def test_prioritize_alerts_tier_and_score_order(self):
        alerts = [a for a in self._sample_alerts() if a is not None]
        prioritized = filter_and_prioritize_alerts(alerts)

        assert len(prioritized) == 4
        # First must be CRITICAL
        assert prioritized[0].priority_tier == PriorityTier.CRITICAL
        assert prioritized[0].entity_id == "tx_crit"

        # Second must be highest HIGH (75.0 > 65.0)
        assert prioritized[1].priority_tier == PriorityTier.HIGH
        assert prioritized[1].entity_id == "tx_high"

        # Third must be second HIGH (65.0)
        assert prioritized[2].priority_tier == PriorityTier.HIGH
        assert prioritized[2].entity_id == "tx_syn"

        # Fourth must be MEDIUM (48.0)
        assert prioritized[3].priority_tier == PriorityTier.MEDIUM
        assert prioritized[3].entity_id == "tx_med"

    def test_filter_alerts_by_min_tier(self):
        alerts = [a for a in self._sample_alerts() if a is not None]
        filtered = filter_and_prioritize_alerts(
            alerts, AlertFilter(min_tier=PriorityTier.HIGH)
        )
        assert len(filtered) == 3
        assert all(a.priority_tier in (PriorityTier.CRITICAL, PriorityTier.HIGH) for a in filtered)

    def test_filter_alerts_exclude_synthetic(self):
        alerts = [a for a in self._sample_alerts() if a is not None]
        filtered = filter_and_prioritize_alerts(
            alerts, AlertFilter(include_synthetic=False)
        )
        assert len(filtered) == 3
        assert all(not a.contains_synthetic_input for a in filtered)

    def test_filter_alerts_with_limit(self):
        alerts = [a for a in self._sample_alerts() if a is not None]
        filtered = filter_and_prioritize_alerts(
            alerts, AlertFilter(limit=2)
        )
        assert len(filtered) == 2
        assert filtered[0].entity_id == "tx_crit"
        assert filtered[1].entity_id == "tx_high"
