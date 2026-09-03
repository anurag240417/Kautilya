"""Risk assessment module.

Combines model outputs and investigative signals into an explainable
risk assessment. Risk scores represent investigative priority, NOT
probability of guilt.
"""

from .evidence import (
    build_evidence_ledger,
    generate_narrative_explanation,
)
from .scorer import (
    DEFAULT_SYNTHESIS_CONFIG,
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
    # Evidence Ledger
    "build_evidence_ledger",
    "generate_narrative_explanation",
]
