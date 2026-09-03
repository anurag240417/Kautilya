"""Risk scoring and evidence domain models.

Risk scores represent investigative priority and triage ranking, NOT
probability of guilt or legal proof of criminal activity. See CONTEXT.md
— Forensic Principle.

Evidence must distinguish between observation, correlation, model
prediction, and risk assessment (never present one as another).
"""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from backend.domain.types import EntityClass, EvidenceType, PriorityTier


class EvidenceCategory(StrEnum):
    """Categorization of evidence in the Evidence Ledger."""

    ML_BEHAVIORAL = "ml_behavioral"  # Supervised classifier predictions & interpretable features
    ANOMALY = "anomaly"  # Statistical deviance from learned baseline
    GRAPH_STRUCTURAL = "graph_structural"  # Graph topology, paths, degree, community context
    TEMPORAL = "temporal"  # Window proximity, timestamp alignment
    SYNTHETIC_NETWORK = "synthetic_network"  # Propagation hops, IP correlation, GeoIP/ASN
    KNOWN_INDICATOR = "known_indicator"  # Match against known watchlists or domain heuristics


class SignalInput(BaseModel):
    """Extensible container for raw investigative signals entering the risk engine.

    Maintains explicit separation across core signal categories and provides
    an open-ended `custom_signals` slot for future signals without schema breaks.
    """

    entity_id: str = Field(description="Transaction ID or wallet address")
    entity_type: str = Field(description="'transaction' or 'wallet'")

    # Core decoupled signals [0.0, 1.0]
    illicit_probability: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Continuous likelihood of illicit class from supervised ML",
    )
    predicted_label: EntityClass | None = Field(
        default=None,
        description="Categorical classification label (1=Illicit, 2=Licit, 3=Unknown)",
    )
    anomaly_score: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Statistical deviance from baseline; NOT proof of illicitness",
    )
    graph_signal: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Structural or relational pattern score; proximity alone is never proof",
    )
    correlation_confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Plausibility and consistency of correlation; distinct from guilt",
    )
    known_indicator_signal: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Match against known high-confidence watchlists or heuristics",
    )

    # Open-ended extensible signals slot
    custom_signals: dict[str, float] = Field(
        default_factory=dict,
        description="Extensible dictionary for future signal categories without schema changes",
    )

    # Provenance tracking
    contains_synthetic_input: bool = Field(
        default=False,
        description="True if any input signal relies on synthetic network or temporal data",
    )
    time_step: int | None = Field(
        default=None, ge=1, le=49, description="Elliptic++ time_step index if applicable"
    )


class EvidenceRecord(BaseModel):
    """An individual structured evidence entry for the Evidence Ledger.

    Forensic Rule:
        Human-facing explanations must use natural language derived exclusively
        from known interpretable dimensions (volume, fees, degree, hops).
        Anonymized dataset feature names (Local_feature_*, Aggregate_feature_*)
        must NEVER appear in headline or description.
    """

    evidence_id: str = Field(description="Unique identifier for this piece of evidence")
    category: EvidenceCategory = Field(description="Evidence domain category")
    evidence_type: EvidenceType = Field(
        description="Whether this is an observation, correlation, prediction, or assessment"
    )
    source_entity_id: str = Field(description="Entity this evidence relates to")
    source_entity_type: str = Field(description="'transaction', 'wallet', 'ip', etc.")
    headline: str = Field(description="Short human-understandable summary of this finding")
    description: str = Field(
        description="Human narrative explaining why this was flagged and investigative relevance"
    )
    supporting_metrics: dict[str, Any] = Field(
        default_factory=dict,
        description="Interpretable feature metrics (e.g. total_btc, fee_ratio, hop_count)",
    )
    confidence: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Confidence level (0–1) for correlations and model inferences",
    )
    is_synthetic: bool = Field(
        default=False,
        description="True if this evidence derives from synthetic data",
    )
    provenance: str | None = Field(
        default=None, description="Source system, model version, or generator run ID"
    )


class EvidenceLedger(BaseModel):
    """Structured Evidence Ledger compiling all evidence for an entity.

    Answers the investigative question: 'Why was this entity flagged?'
    by categorizing ML behavioral findings, anomaly deviance, graph patterns,
    temporal proximity, and synthetic network observations.
    """

    entity_id: str = Field(description="Transaction ID or wallet address")
    entity_type: str = Field(description="'transaction' or 'wallet'")
    records: list[EvidenceRecord] = Field(
        default_factory=list,
        description="Ordered collection of structured evidence records",
    )

    def add_record(self, record: EvidenceRecord) -> None:
        """Add an evidence record to the ledger."""
        self.records.append(record)

    def get_by_category(self, category: EvidenceCategory) -> list[EvidenceRecord]:
        """Filter evidence records by category."""
        return [r for r in self.records if r.category == category]

    def has_synthetic_evidence(self) -> bool:
        """Return True if any record in this ledger is synthetic."""
        return any(r.is_synthetic for r in self.records)

    def to_human_summary(self) -> list[str]:
        """Generate human-readable summary bullet points for investigator reports."""
        return [
            f"[{r.category.value.upper()}] {r.headline}: {r.description}"
            for r in self.records
        ]


class RiskScore(BaseModel):
    """Composite risk assessment and triage prioritization for an entity.

    Combines multiple signals into an overall investigative priority score.
    Each component signal and contributing evidence is preserved for full
    explainability.

    See ARCHITECTURE.md — Risk Architecture.
    """

    entity_id: str = Field(description="Transaction ID or wallet address")
    entity_type: str = Field(description="'transaction' or 'wallet'")
    score: float = Field(
        description="Investigative priority ranking score (0–100); NOT probability of criminality"
    )
    priority_tier: PriorityTier = Field(
        default=PriorityTier.LOW,
        description="Configurable priority tier (Critical, High, Medium, Low)",
    )

    # Decomposed component signals [0.0, 1.0]
    behavioral_signal: float | None = None
    graph_signal: float | None = None
    anomaly_signal: float | None = None
    correlation_signal: float | None = None
    known_indicator_signal: float | None = None

    active_signals: list[str] = Field(
        default_factory=list,
        description="List of signal names that contributed to this score",
    )
    explanation: str | None = Field(
        default=None, description="Human-readable explanation of why this score was assigned"
    )
    evidence_ledger: EvidenceLedger | None = Field(
        default=None, description="Structured Evidence Ledger supporting this assessment"
    )
    contains_synthetic_input: bool = Field(
        default=False,
        description="True if any input to this score included synthetic data",
    )


class Evidence(BaseModel):
    """An individual piece of evidence or inference (legacy support).

    Preserved for backwards compatibility with earlier pipeline stages.
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
