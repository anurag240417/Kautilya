"""ChainTrace domain models.

Canonical data schemas used across all pipeline modules.
All models use Pydantic v2 for validation and serialization.
"""

from backend.domain.correlation import (
    CandidateIPRole,
    CorrelationConfidenceMetrics,
    TemporalCorrelation,
    TransactionIPCorrelation,
)
from backend.domain.experiment import ExperimentConfig, GeneratorConfig
from backend.domain.graph import AddrAddrEdge, AddrTxEdge, TxAddrEdge, TxTxEdge
from backend.domain.ml import MLScore
from backend.domain.network import NetworkObservation
from backend.domain.risk import (
    Evidence,
    EvidenceCategory,
    EvidenceLedger,
    EvidenceRecord,
    RiskScore,
    SignalInput,
)
from backend.domain.transaction import Transaction, TransactionFeatures
from backend.domain.types import (
    EntityClass,
    EvidenceType,
    ExplanationType,
    PriorityTier,
    ScriptType,
)
from backend.domain.wallet import StatsSummary, Wallet

__all__ = [
    # Enums
    "EntityClass",
    "EvidenceType",
    "ExplanationType",
    "ScriptType",
    "CandidateIPRole",
    "PriorityTier",
    "EvidenceCategory",
    # Blockchain layer
    "Transaction",
    "TransactionFeatures",
    "Wallet",
    "StatsSummary",
    # Network layer
    "NetworkObservation",
    # Correlation layer
    "CorrelationConfidenceMetrics",
    "TemporalCorrelation",
    "TransactionIPCorrelation",
    # Graph edges
    "TxTxEdge",
    "AddrTxEdge",
    "TxAddrEdge",
    "AddrAddrEdge",
    # ML / Risk
    "MLScore",
    "RiskScore",
    "SignalInput",
    "Evidence",
    "EvidenceRecord",
    "EvidenceLedger",
    # Configuration
    "ExperimentConfig",
    "GeneratorConfig",
]
