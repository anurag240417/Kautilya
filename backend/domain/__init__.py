"""ChainTrace domain models.

Canonical data schemas used across all pipeline modules.
All models use Pydantic v2 for validation and serialization.
"""

from backend.domain.experiment import ExperimentConfig, GeneratorConfig
from backend.domain.graph import AddrAddrEdge, AddrTxEdge, TxAddrEdge, TxTxEdge
from backend.domain.ml import MLScore
from backend.domain.network import NetworkObservation
from backend.domain.risk import Evidence, RiskScore
from backend.domain.transaction import Transaction, TransactionFeatures
from backend.domain.types import EntityClass, EvidenceType, ScriptType
from backend.domain.wallet import StatsSummary, Wallet

__all__ = [
    # Enums
    "EntityClass",
    "EvidenceType",
    "ScriptType",
    # Blockchain layer
    "Transaction",
    "TransactionFeatures",
    "Wallet",
    "StatsSummary",
    # Network layer
    "NetworkObservation",
    # Graph edges
    "TxTxEdge",
    "AddrTxEdge",
    "TxAddrEdge",
    "AddrAddrEdge",
    # ML / Risk
    "MLScore",
    "RiskScore",
    "Evidence",
    # Configuration
    "ExperimentConfig",
    "GeneratorConfig",
]
