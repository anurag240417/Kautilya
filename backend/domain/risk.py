"""Risk scoring and evidence domain models.

Risk scores represent investigative priority, NOT probability of
guilt or proof of criminal activity. See CONTEXT.md — Forensic
Principle.

Evidence must distinguish between observation, correlation,
model prediction, and risk assessment (never present one as another).
"""


from pydantic import BaseModel, Field

from backend.domain.types import EvidenceType


class RiskScore(BaseModel):
    """Composite risk assessment for an entity.

    Combines multiple signals into an overall investigative priority
    score. Each component signal is preserved for explainability.

    See ARCHITECTURE.md — Risk Architecture.
    """

    entity_id: str = Field(description="Transaction ID or wallet address")
    entity_type: str = Field(description="'transaction' or 'wallet'")
    score: float = Field(description="Overall risk score (investigative priority)")
    behavioral_signal: float | None = None
    graph_signal: float | None = None
    anomaly_signal: float | None = None
    correlation_signal: float | None = None
    known_indicator_signal: float | None = None
    explanation: str | None = Field(
        default=None, description="Human-readable explanation of why this score was assigned"
    )
    contains_synthetic_input: bool = Field(
        default=False,
        description="True if any input to this score included synthetic data",
    )


class Evidence(BaseModel):
    """An individual piece of evidence or inference.

    ChainTrace must separate raw observations from derived conclusions.
    The ``evidence_type`` field classifies the provenance of this record.
    """

    evidence_type: EvidenceType = Field(
        description="Whether this is an observation, correlation, prediction, or assessment"
    )
    source_entity_id: str = Field(description="Entity this evidence relates to")
    source_entity_type: str = Field(description="'transaction', 'wallet', 'network_event', etc.")
    description: str = Field(description="Human-readable description of the evidence")
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence level (0–1) for correlations and inferences",
    )
    is_synthetic: bool = Field(
        default=False,
        description="True if this evidence derives from synthetic data",
    )
    provenance: str | None = Field(
        default=None, description="Source system or data origin"
    )
