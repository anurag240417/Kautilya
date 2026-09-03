"""Transaction-to-wallet/entity aggregation with reproducible traceability.

Aggregates transaction-level risk scores and evidence into wallet/entity-level
prioritization while maintaining full traceability to contributing transactions.

Forensic Rules (per AGENTS.md §3.7 — Entity-Level Rollup Traceability):
    1. Transaction-level predictions and scores must remain permanently
       traceable to their source transactions.
    2. Wallet/entity-level prioritization must be computed separately through
       an explicit, documented aggregation method.
    3. A transaction prediction must NEVER automatically become a blanket
       accusation against the entire wallet or entity.
    4. The aggregation method and contributing transaction evidence must be
       stored or reproducible.
"""

from enum import StrEnum
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, Field

from backend.domain.risk import EvidenceLedger, RiskScore
from backend.domain.types import PriorityTier

if TYPE_CHECKING:
    from backend.risk.scorer import PriorityTierConfig


class AggregationMethod(StrEnum):
    """Documented aggregation methods for transaction-to-entity rollup.

    Each method defines how individual transaction risk scores combine
    into an entity-level prioritization score. The method used is always
    stored with the result for reproducibility.
    """

    # Use the maximum transaction score as the entity score
    MAX = "max"
    # Use the arithmetic mean of all transaction scores
    MEAN = "mean"
    # Weight each transaction score by its BTC volume
    VOLUME_WEIGHTED = "volume_weighted"
    # Count of transactions exceeding a configurable threshold
    FREQUENCY = "frequency"


class TransactionContribution(BaseModel):
    """A single transaction's contribution to an entity-level aggregation.

    Preserves full traceability: each contributing transaction retains
    its own risk score, priority tier, and evidence summary so that
    downstream investigators can trace exactly which transactions
    influenced the entity-level assessment.
    """

    txid: str = Field(description="Transaction ID that contributed")
    risk_score: float = Field(
        description="This transaction's individual risk score (0–100)"
    )
    priority_tier: PriorityTier = Field(
        description="This transaction's individual priority tier"
    )
    illicit_probability: float | None = Field(
        default=None,
        description="Transaction-level illicit probability, if available",
    )
    anomaly_score: float | None = Field(
        default=None,
        description="Transaction-level anomaly score, if available",
    )
    evidence_summary: str | None = Field(
        default=None,
        description="Brief summary of evidence for this transaction",
    )
    contains_synthetic: bool = Field(
        default=False,
        description="Whether this transaction used synthetic data",
    )


class EntityAggregation(BaseModel):
    """Entity-level aggregated risk assessment with full transaction traceability.

    Per AGENTS.md §3.7: Wallet and entity-level prioritization is computed
    separately from transaction-level predictions. A transaction prediction
    must NEVER automatically become a blanket accusation against the entire
    wallet or entity.
    """

    entity_id: str = Field(
        description="Wallet address or entity cluster identifier"
    )
    entity_type: str = Field(
        default="wallet",
        description="Type of entity being aggregated",
    )
    aggregated_score: float = Field(
        description=(
            "Entity-level investigative priority score (0–100); "
            "NOT a probability of criminality"
        ),
    )
    priority_tier: PriorityTier = Field(
        description="Entity-level priority tier based on aggregated score"
    )
    tier_description: str | None = Field(
        default=None,
        description="Human-readable description of the assigned tier",
    )
    aggregation_method: AggregationMethod = Field(
        description="Documented method used to aggregate transaction scores"
    )
    transaction_count: int = Field(
        description="Total number of transactions evaluated"
    )
    flagged_transaction_count: int = Field(
        default=0,
        description=(
            "Number of transactions that individually exceeded "
            "the medium tier threshold"
        ),
    )
    contributing_transactions: list[TransactionContribution] = Field(
        default_factory=list,
        description=(
            "Ordered list of contributing transactions with individual "
            "scores and evidence, preserving full traceability"
        ),
    )
    aggregation_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Additional metadata about the aggregation "
            "(e.g., volume weights, frequency threshold)"
        ),
    )
    contains_synthetic_input: bool = Field(
        default=False,
        description="True if any contributing transaction used synthetic data",
    )
    forensic_disclaimer: str = Field(
        default=(
            "This entity-level assessment is an investigative prioritization "
            "derived from aggregated transaction-level evidence. Individual "
            "transaction predictions do not constitute blanket accusations "
            "against the entire wallet or entity."
        ),
        description="Mandatory forensic disclaimer per AGENTS.md §3.7",
    )


def _build_contribution(
    risk_score: RiskScore,
    evidence_ledger: EvidenceLedger | None = None,
) -> TransactionContribution:
    """Build a TransactionContribution from a RiskScore."""
    summary = None
    if evidence_ledger and evidence_ledger.records:
        summary = "; ".join(
            r.headline for r in evidence_ledger.records[:3]
        )
        if len(evidence_ledger.records) > 3:
            remaining = len(evidence_ledger.records) - 3
            summary += f" (+{remaining} more)"

    return TransactionContribution(
        txid=risk_score.entity_id,
        risk_score=risk_score.score,
        priority_tier=risk_score.priority_tier,
        illicit_probability=risk_score.behavioral_signal,
        anomaly_score=risk_score.anomaly_signal,
        evidence_summary=summary or risk_score.explanation,
        contains_synthetic=risk_score.contains_synthetic_input,
    )


def aggregate_transaction_scores(
    entity_id: str,
    transaction_scores: list[RiskScore],
    transaction_ledgers: dict[str, EvidenceLedger] | None = None,
    method: AggregationMethod = AggregationMethod.MAX,
    volume_weights: dict[str, float] | None = None,
    frequency_threshold: float = 40.0,
    tier_config: "PriorityTierConfig | None" = None,
) -> EntityAggregation:
    """Aggregate transaction-level risk scores into an entity-level assessment.

    Implements documented, reproducible aggregation methods per AGENTS.md §3.7.
    The aggregation method and all contributing transaction evidence are stored
    in the result for full traceability.

    Args:
        entity_id: Wallet address or entity cluster identifier.
        transaction_scores: List of transaction-level RiskScore objects.
        transaction_ledgers: Optional mapping of txid -> EvidenceLedger
            for richer evidence summaries.
        method: Aggregation method to use (MAX, MEAN, VOLUME_WEIGHTED,
            FREQUENCY).
        volume_weights: Required for VOLUME_WEIGHTED method. Maps txid
            to its BTC volume. Transactions without volume entries are
            assigned equal weight.
        frequency_threshold: Score threshold for FREQUENCY method;
            transactions scoring above this are counted as flagged.
        tier_config: Optional PriorityTierConfig for tier assignment.
            If None, uses default tier boundaries.

    Returns:
        EntityAggregation with full traceability to contributing
        transactions.

    Raises:
        ValueError: If transaction_scores is empty or method is
            VOLUME_WEIGHTED without volume_weights.
    """
    if not transaction_scores:
        msg = "Cannot aggregate: no transaction scores provided"
        raise ValueError(msg)

    ledgers = transaction_ledgers or {}

    # Build traceable contribution records
    contributions = [
        _build_contribution(
            rs, ledgers.get(rs.entity_id)
        )
        for rs in transaction_scores
    ]

    # Sort contributions by risk score descending for investigator convenience
    contributions.sort(key=lambda c: c.risk_score, reverse=True)

    # Compute aggregated score based on method
    metadata: dict[str, Any] = {"method": method.value}

    if method == AggregationMethod.MAX:
        agg_score = max(rs.score for rs in transaction_scores)
        metadata["max_contributing_tx"] = max(
            transaction_scores, key=lambda rs: rs.score
        ).entity_id

    elif method == AggregationMethod.MEAN:
        total = sum(rs.score for rs in transaction_scores)
        agg_score = total / len(transaction_scores)
        metadata["sum_scores"] = round(total, 2)

    elif method == AggregationMethod.VOLUME_WEIGHTED:
        if not volume_weights:
            msg = (
                "VOLUME_WEIGHTED aggregation requires volume_weights "
                "mapping (txid -> BTC volume)"
            )
            raise ValueError(msg)
        total_volume = 0.0
        weighted_sum = 0.0
        for rs in transaction_scores:
            vol = volume_weights.get(rs.entity_id, 1.0)
            weighted_sum += rs.score * vol
            total_volume += vol
        agg_score = (
            weighted_sum / total_volume if total_volume > 0 else 0.0
        )
        metadata["total_volume"] = round(total_volume, 4)

    elif method == AggregationMethod.FREQUENCY:
        flagged = [
            rs for rs in transaction_scores
            if rs.score >= frequency_threshold
        ]
        # Frequency score: proportion of flagged transactions scaled to 100
        freq_ratio = (
            len(flagged) / len(transaction_scores)
            if len(transaction_scores) > 0
            else 0.0
        )
        agg_score = min(100.0, freq_ratio * 100.0)
        metadata["frequency_threshold"] = frequency_threshold
        metadata["flagged_count"] = len(flagged)

    else:
        msg = f"Unknown aggregation method: {method}"
        raise ValueError(msg)

    agg_score = round(max(0.0, min(100.0, float(agg_score))), 2)

    # Determine entity-level tier
    # Import here to avoid circular dependency
    from backend.risk.scorer import DEFAULT_TIER_CONFIG

    tc = tier_config or DEFAULT_TIER_CONFIG
    tier_def = tc.determine_tier(agg_score)

    # Count flagged transactions (those at MEDIUM or above)
    flagged_count = sum(
        1 for c in contributions if c.risk_score >= 40.0
    )

    # Track synthetic provenance
    has_synthetic = any(
        c.contains_synthetic for c in contributions
    )

    return EntityAggregation(
        entity_id=entity_id,
        entity_type="wallet",
        aggregated_score=agg_score,
        priority_tier=tier_def.tier,
        tier_description=tier_def.description,
        aggregation_method=method,
        transaction_count=len(transaction_scores),
        flagged_transaction_count=flagged_count,
        contributing_transactions=contributions,
        aggregation_metadata=metadata,
        contains_synthetic_input=has_synthetic,
    )
