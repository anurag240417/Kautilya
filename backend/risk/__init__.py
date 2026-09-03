"""Risk assessment module.

Combines model outputs and investigative signals into an explainable
risk assessment. Risk scores represent investigative priority, NOT
probability of guilt.
"""

from .aggregation import (
    AggregationMethod,
    EntityAggregation,
    TransactionContribution,
    aggregate_transaction_scores,
)
from .evidence import (
    build_evidence_ledger,
    generate_narrative_explanation,
)
from .scorer import (
    DEFAULT_SYNTHESIS_CONFIG,
    DEFAULT_TIER_CONFIG,
    PriorityTierConfig,
    PriorityTierDefinition,
    SynthesisConfig,
    SynthesisPolicy,
    determine_priority_tier,
    synthesize_risk_score,
)

__all__ = [
    # Scorer & Synthesis
    "SynthesisPolicy",
    "SynthesisConfig",
    "DEFAULT_SYNTHESIS_CONFIG",
    "determine_priority_tier",
    "synthesize_risk_score",
    # Priority Tier Configuration
    "PriorityTierConfig",
    "PriorityTierDefinition",
    "DEFAULT_TIER_CONFIG",
    # Evidence Ledger
    "build_evidence_ledger",
    "generate_narrative_explanation",
    # Entity Aggregation
    "AggregationMethod",
    "EntityAggregation",
    "TransactionContribution",
    "aggregate_transaction_scores",
]
