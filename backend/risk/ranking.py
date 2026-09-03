"""Investigative alert generation, entity ranking, and queue prioritization.

Implements entity prioritization and alert triage queues per ARCHITECTURE.md §risk/
and AGENTS.md §3.6.

Forensic Rules:
    1. Alert Prioritization represents investigative triage ranking to optimize
       investigator caseloads, NOT a legal accusation, proof of guilt, or
       calibrated probability of criminality.
    2. Alerts support configurable priority tiers (Critical, High, Medium, Low).
    3. Wallet/entity alerts preserve the non-accusation principle: individual
       transaction predictions do NOT automatically brand the entire wallet.
    4. Synthetic provenance flags (`contains_synthetic_input`) are strictly
       propagated to alerts and ranked queues.
"""

import uuid
from typing import Any

from backend.domain.alert import (
    AlertFilter,
    InvestigativeAlert,
    RankedEntity,
    RankingCriteria,
)
from backend.domain.risk import EvidenceLedger, RiskScore
from backend.domain.types import AlertStatus, PriorityTier
from backend.risk.aggregation import EntityAggregation

# Numeric order for tier severity comparison
TIER_SEVERITY: dict[PriorityTier, int] = {
    PriorityTier.CRITICAL: 4,
    PriorityTier.HIGH: 3,
    PriorityTier.MEDIUM: 2,
    PriorityTier.LOW: 1,
}

_TIER_RECOMMENDATIONS: dict[PriorityTier, str] = {
    PriorityTier.CRITICAL: (
        "Immediate investigator review required. Multiple corroborating signals present. "
        "Recommend initiating counterparty cluster tracing and preserving transaction records."
    ),
    PriorityTier.HIGH: (
        "Elevated priority review. Significant risk indicators present. "
        "Recommend verifying transaction counterparties and inspecting related wallet history."
    ),
    PriorityTier.MEDIUM: (
        "Secondary queue review. Moderate indicators or isolated statistical deviance. "
        "Recommend routine counterparty verification and activity monitoring."
    ),
    PriorityTier.LOW: (
        "Routine monitoring. Baseline activity with no immediate indicators of concern."
    ),
}


def generate_alert(
    risk_score: RiskScore,
    evidence_ledger: EvidenceLedger | None = None,
    min_alert_tier: PriorityTier = PriorityTier.MEDIUM,
) -> InvestigativeAlert | None:
    """Generate an investigative alert from a transaction or wallet RiskScore.

    If the risk score's priority tier is below `min_alert_tier`, returns None.

    Args:
        risk_score: RiskScore domain object.
        evidence_ledger: Optional structured EvidenceLedger supporting the score.
        min_alert_tier: Minimum tier threshold required to generate an alert
            (default: PriorityTier.MEDIUM).

    Returns:
        InvestigativeAlert if the tier threshold is met, else None.
    """
    if TIER_SEVERITY[risk_score.priority_tier] < TIER_SEVERITY[min_alert_tier]:
        return None

    tier_label = risk_score.priority_tier.value.upper()
    entity_label = risk_score.entity_type.capitalize()
    headline = (
        f"[{tier_label}] {entity_label} {risk_score.entity_id} — "
        f"Priority {risk_score.score:.1f}/100"
    )

    # Build summary narrative
    if evidence_ledger and evidence_ledger.records:
        top_headlines = [r.headline for r in evidence_ledger.records[:2]]
        summary = f"{risk_score.explanation or ''} Key evidence: {'; '.join(top_headlines)}."
    else:
        summary = (
            risk_score.explanation
            or f"Investigative priority score {risk_score.score:.1f}/100 ({tier_label} tier)."
        )

    action = _TIER_RECOMMENDATIONS.get(
        risk_score.priority_tier,
        "Review entity activity according to standard operating procedures.",
    )

    evidence_count = len(evidence_ledger.records) if evidence_ledger else 0

    return InvestigativeAlert(
        alert_id=f"alt-{uuid.uuid4().hex[:10]}",
        entity_id=risk_score.entity_id,
        entity_type=risk_score.entity_type,
        risk_score=risk_score.score,
        priority_tier=risk_score.priority_tier,
        tier_description=risk_score.tier_description,
        status=AlertStatus.NEW,
        headline=headline,
        summary=summary.strip(),
        active_signals=list(risk_score.active_signals),
        recommended_action=action,
        evidence_count=evidence_count,
        contains_synthetic_input=risk_score.contains_synthetic_input,
    )


def generate_entity_alert(
    aggregation: EntityAggregation,
    min_alert_tier: PriorityTier = PriorityTier.MEDIUM,
) -> InvestigativeAlert | None:
    """Generate an investigative alert from an EntityAggregation.

    Preserves the non-accusation principle (AGENTS.md §3.7): a single
    suspicious transaction does NOT automatically brand the entire wallet.
    The alert explicitly records contributing transaction statistics and
    the mandatory forensic disclaimer.

    Args:
        aggregation: EntityAggregation domain object for a wallet or cluster.
        min_alert_tier: Minimum tier threshold required to generate an alert.

    Returns:
        InvestigativeAlert if the tier threshold is met, else None.
    """
    if TIER_SEVERITY[aggregation.priority_tier] < TIER_SEVERITY[min_alert_tier]:
        return None

    tier_label = aggregation.priority_tier.value.upper()
    entity_label = aggregation.entity_type.capitalize()
    headline = (
        f"[{tier_label}] {entity_label} {aggregation.entity_id} — "
        f"Aggregated Priority {aggregation.aggregated_score:.1f}/100 "
        f"({aggregation.aggregation_method.value})"
    )

    summary_parts = [
        f"{entity_label} {aggregation.entity_id} aggregated across "
        f"{aggregation.transaction_count} transaction(s) using "
        f"'{aggregation.aggregation_method.value}' pooling.",
        f"{aggregation.flagged_transaction_count} transaction(s) individually "
        "exceeded the medium priority threshold.",
    ]
    if aggregation.contributing_transactions:
        top_tx = aggregation.contributing_transactions[0]
        summary_parts.append(
            f"Highest-scoring transaction: {top_tx.txid} (score {top_tx.risk_score:.1f})."
        )

    summary = " ".join(summary_parts)

    action = _TIER_RECOMMENDATIONS.get(
        aggregation.priority_tier,
        "Review wallet transaction history and counterparty connections.",
    )

    metadata: dict[str, Any] = {
        "aggregation_method": aggregation.aggregation_method.value,
        "transaction_count": aggregation.transaction_count,
        "flagged_transaction_count": aggregation.flagged_transaction_count,
        **aggregation.aggregation_metadata,
    }

    return InvestigativeAlert(
        alert_id=f"alt-entity-{uuid.uuid4().hex[:10]}",
        entity_id=aggregation.entity_id,
        entity_type=aggregation.entity_type,
        risk_score=aggregation.aggregated_score,
        priority_tier=aggregation.priority_tier,
        tier_description=aggregation.tier_description,
        status=AlertStatus.NEW,
        headline=headline,
        summary=summary,
        active_signals=[f"aggregation:{aggregation.aggregation_method.value}"],
        recommended_action=action,
        evidence_count=len(aggregation.contributing_transactions),
        metadata=metadata,
        contains_synthetic_input=aggregation.contains_synthetic_input,
        forensic_disclaimer=aggregation.forensic_disclaimer,
    )


def rank_entities(
    items: list[RiskScore | EntityAggregation],
    criteria: RankingCriteria | None = None,
) -> list[RankedEntity]:
    """Rank entities into an ordered investigator triage queue.

    Entities are sorted primarily by risk score (descending). When
    `secondary_sort_by_corroboration` is enabled (default), ties in score
    are broken by the count of active corroborating signals (or flagged
    transactions for aggregated entities).

    Args:
        items: List of RiskScore or EntityAggregation objects.
        criteria: Optional RankingCriteria specifying filters and limit.

    Returns:
        List of RankedEntity objects with 1-indexed rank assignments.
    """
    cfg = criteria or RankingCriteria()

    filtered: list[tuple[float, int, RiskScore | EntityAggregation]] = []

    for item in items:
        score = item.score if isinstance(item, RiskScore) else item.aggregated_score
        tier = item.priority_tier

        # Filter by min_score
        if cfg.min_score is not None and score < cfg.min_score:
            continue

        # Filter by min_tier
        if cfg.min_tier is not None and TIER_SEVERITY[tier] < TIER_SEVERITY[cfg.min_tier]:
            continue

        # Compute corroboration count for secondary sorting
        if isinstance(item, RiskScore):
            corrob_count = len(item.active_signals)
        else:
            corrob_count = item.flagged_transaction_count

        filtered.append((score, corrob_count, item))

    # Sort: Primary = score descending, Secondary = corroboration descending
    if cfg.secondary_sort_by_corroboration:
        filtered.sort(key=lambda x: (x[0], x[1]), reverse=True)
    else:
        filtered.sort(key=lambda x: x[0], reverse=True)

    # Apply limit
    if cfg.limit is not None:
        filtered = filtered[: cfg.limit]

    ranked: list[RankedEntity] = []
    for rank_idx, (_, corrob, item) in enumerate(filtered, start=1):
        if isinstance(item, RiskScore):
            ranked.append(
                RankedEntity(
                    rank=rank_idx,
                    entity_id=item.entity_id,
                    entity_type=item.entity_type,
                    risk_score=item.score,
                    priority_tier=item.priority_tier,
                    tier_description=item.tier_description,
                    active_signals=list(item.active_signals),
                    corroboration_count=corrob,
                    contains_synthetic_input=item.contains_synthetic_input,
                    summary=item.explanation,
                )
            )
        else:
            ranked.append(
                RankedEntity(
                    rank=rank_idx,
                    entity_id=item.entity_id,
                    entity_type=item.entity_type,
                    risk_score=item.aggregated_score,
                    priority_tier=item.priority_tier,
                    tier_description=item.tier_description,
                    active_signals=[f"aggregation:{item.aggregation_method.value}"],
                    corroboration_count=corrob,
                    contains_synthetic_input=item.contains_synthetic_input,
                    summary=(
                        f"Aggregated {item.aggregation_method.value} score across "
                        f"{item.transaction_count} transaction(s)."
                    ),
                )
            )

    return ranked


def filter_and_prioritize_alerts(
    alerts: list[InvestigativeAlert],
    filter_criteria: AlertFilter | None = None,
) -> list[InvestigativeAlert]:
    """Filter and prioritize alerts for investigator queue display.

    Sorts alerts by:
        1. Priority tier severity descending (CRITICAL > HIGH > MEDIUM > LOW)
        2. Risk score descending (100.0 -> 0.0)
        3. Creation timestamp ascending (FIFO within same tier and score)

    Args:
        alerts: List of InvestigativeAlert objects.
        filter_criteria: Optional AlertFilter with filter thresholds.

    Returns:
        Sorted, filtered list of InvestigativeAlert objects.
    """
    crit = filter_criteria or AlertFilter()

    filtered: list[InvestigativeAlert] = []
    for a in alerts:
        if (
            crit.min_tier is not None
            and TIER_SEVERITY[a.priority_tier] < TIER_SEVERITY[crit.min_tier]
        ):
            continue
        if crit.entity_type is not None and a.entity_type != crit.entity_type:
            continue
        if crit.status is not None and a.status != crit.status:
            continue
        if not crit.include_synthetic and a.contains_synthetic_input:
            continue
        if crit.min_score is not None and a.risk_score < crit.min_score:
            continue
        filtered.append(a)

    # Sort: Tier severity desc, Risk score desc, Created at asc
    filtered.sort(
        key=lambda a: (
            -TIER_SEVERITY[a.priority_tier],
            -a.risk_score,
            a.created_at,
        )
    )

    if crit.limit is not None:
        filtered = filtered[: crit.limit]

    return filtered
