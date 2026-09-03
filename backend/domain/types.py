"""Shared enums and type aliases for ChainTrace domain models."""

from enum import IntEnum, StrEnum


class EntityClass(IntEnum):
    """Transaction/wallet classification labels from Elliptic++ dataset.

    Values correspond directly to the dataset encoding:
        1 = Illicit, 2 = Licit, 3 = Unknown.

    Unknown labels must NOT be automatically treated as illicit.
    """

    ILLICIT = 1
    LICIT = 2
    UNKNOWN = 3


class ScriptType(StrEnum):
    """Bitcoin script types required by the SIH problem statement.

    Used in synthetic network observations to classify the transaction
    script type associated with a network propagation event.
    """

    P2PKH = "P2PKH"
    P2SH = "P2SH"
    P2WPKH = "P2WPKH"
    TAPROOT = "TAPROOT"


class EvidenceType(StrEnum):
    """Classification of evidence provenance.

    ChainTrace must distinguish between these categories and never
    present one as another (see CONTEXT.md — Forensic Principle).
    """

    OBSERVATION = "observation"
    CORRELATION = "correlation"
    MODEL_PREDICTION = "model_prediction"
    RISK_ASSESSMENT = "risk_assessment"


class PriorityTier(StrEnum):
    """Investigative alert priority tiers.

    Used to organize investigator triage queues according to urgency.
    """

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ExplanationType(StrEnum):
    """Distinguishes classifier explanations from anomaly explanations.

    Per AGENTS.md §3.5:
    - CLASSIFIER_EXPLANATION: Explains why the model assigned a specific
      predicted_label or illicit_probability.
    - ANOMALY_EXPLANATION: Explains why an observation is statistically
      unusual relative to the learned baseline. Must NEVER be presented
      as an explanation of illicitness unless independently supported.
    """

    CLASSIFIER_EXPLANATION = "classifier_explanation"
    ANOMALY_EXPLANATION = "anomaly_explanation"


class AlertStatus(StrEnum):
    """Lifecycle status for investigative alerts."""

    NEW = "new"
    TRIAGED = "triaged"
    IN_REVIEW = "in_review"
    ESCALATED = "escalated"
    DISMISSED = "dismissed"
