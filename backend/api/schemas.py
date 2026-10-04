"""Pydantic request and response schemas for Kautilya Investigation API.

Defines the contract between the backend investigation services and
API consumers (frontend or automated clients).
"""

from typing import Any

from pydantic import BaseModel, Field, field_validator

from backend.domain.alert import InvestigativeAlert
from backend.domain.risk import RiskScore
from backend.domain.types import AlertStatus, EntityClass
from backend.risk.aggregation import EntityAggregation


class ErrorDetail(BaseModel):
    """Structured error detail payload."""

    code: str = Field(description="Machine-readable error code (e.g. NOT_FOUND)")
    message: str = Field(description="Human-readable error description")
    details: Any = Field(
        default=None,
        description="Optional diagnostic details or validation errors",
    )


class ErrorResponse(BaseModel):
    """Standardized error envelope per CODING_CONVENTIONS.md."""

    error: ErrorDetail


class TransactionResponse(BaseModel):
    """Investigation payload for a transaction query."""

    txid: int = Field(description="Elliptic++ anonymized transaction ID")
    time_step: int = Field(ge=1, le=49, description="Temporal step index (1–49)")
    label: EntityClass | None = Field(
        default=None,
        description="1=Illicit, 2=Licit, 3=Unknown",
    )
    interpretable_features: dict[str, float | None] = Field(
        default_factory=dict,
        description="Behavioral features (volume, fees, size, degrees; no anonymized features)",
    )
    risk_score: RiskScore | None = Field(
        default=None,
        description="Synthesized risk score and Evidence Ledger if evaluated",
    )
    correlations: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Correlated network events and candidate IPs with confidence",
    )
    connected_wallets: dict[str, list[str]] = Field(
        default_factory=lambda: {"inputs": [], "outputs": []},
        description="Connected input and output wallet addresses",
    )
    is_synthetic: bool = Field(
        default=False,
        description="True if transaction or its correlations rely on synthetic data",
    )


class WalletResponse(BaseModel):
    """Investigation payload for a wallet query."""

    address: str = Field(description="Elliptic++ wallet address identifier")
    time_step: int = Field(ge=1, le=49, description="Temporal step index (1–49)")
    label: EntityClass | None = Field(
        default=None,
        description="1=Illicit, 2=Licit, 3=Unknown",
    )
    stats: dict[str, Any] = Field(
        default_factory=dict,
        description="Interpretable wallet statistics (BTC sent/received, block lifetime, etc.)",
    )
    aggregation: EntityAggregation | None = Field(
        default=None,
        description="Aggregated risk score across transactions with full traceability",
    )
    counterparties: list[str] = Field(
        default_factory=list,
        description="Known counterparty wallet addresses or entities",
    )
    is_synthetic: bool = Field(
        default=False,
        description="True if any contributing data is synthetic",
    )
    forensic_disclaimer: str = Field(
        default=(
            "Individual transaction predictions do not constitute a blanket accusation "
            "against the entire wallet or entity. All risk scores represent investigative "
            "prioritization, NOT proof of criminality."
        ),
        description="Mandatory forensic disclaimer per AGENTS.md §3.7",
    )


class AlertListResponse(BaseModel):
    """Triage queue response containing ranked alerts."""

    total: int = Field(description="Total number of alerts in system matching type")
    filtered_count: int = Field(description="Count of alerts returned after applying filters")
    alerts: list[InvestigativeAlert] = Field(
        default_factory=list,
        description="Prioritized list of alerts",
    )
    forensic_disclaimer: str = Field(
        default=(
            "Alerts represent investigative triage prioritization to manage investigator "
            "caseloads, NOT a legal accusation, proof of guilt, or confirmed criminality."
        ),
        description="Mandatory triage disclaimer",
    )


class AlertUpdatePayload(BaseModel):
    """Payload for updating an alert's lifecycle status."""

    status: AlertStatus = Field(description="New lifecycle status")
    reviewer_notes: str | None = Field(
        default=None,
        description="Optional investigator notes or triage rationale",
    )

    @field_validator("status", mode="before")
    @classmethod
    def normalize_status(cls, v: Any) -> Any:
        if isinstance(v, str):
            v_norm = v.strip().lower()
            if v_norm == "closed":
                return AlertStatus.DISMISSED
            try:
                return AlertStatus(v_norm)
            except ValueError:
                return v
        return v


class GraphNode(BaseModel):
    """Node in an investigation graph response."""

    id: str = Field(description="Node identifier (transaction ID or wallet address)")
    type: str = Field(description="'transaction' or 'wallet'")
    label: str | None = Field(default=None, description="Classification label if available")
    priority_tier: str | None = Field(default=None, description="Assigned priority tier")
    risk_score: float | None = Field(default=None, description="Risk priority score")


class GraphEdge(BaseModel):
    """Edge in an investigation graph response."""

    source: str = Field(description="Source node ID")
    target: str = Field(description="Target node ID")
    relationship: str = Field(description="Edge type: tx_tx, addr_tx, tx_addr, addr_addr")
    is_synthetic: bool = Field(default=False, description="True if edge is synthetic")
    confidence: float | None = Field(default=None, description="Edge confidence score")
    temporal_context: int | None = Field(default=None, description="Time step context")
    provenance: str | None = Field(default=None, description="Origin or dataset source")


class GraphResponse(BaseModel):
    """Subgraph response for an entity investigation."""

    entity_id: str = Field(description="Query root entity ID")
    entity_type: str = Field(description="'transaction' or 'wallet'")
    depth: int = Field(description="Hop traversal depth")
    node_count: int = Field(description="Total nodes in subgraph")
    edge_count: int = Field(description="Total edges in subgraph")
    nodes: list[GraphNode] = Field(default_factory=list, description="Nodes in subgraph")
    edges: list[GraphEdge] = Field(default_factory=list, description="Edges in subgraph")


class GraphPathResponse(BaseModel):
    """Path query response between two entities."""

    source: str = Field(description="Starting entity ID")
    target: str = Field(description="Target entity ID")
    found: bool = Field(description="True if a path exists between entities")
    path_length: int | None = Field(default=None, description="Number of hops in path")
    path_nodes: list[str] = Field(default_factory=list, description="Ordered node IDs along path")
    path_edges: list[GraphEdge] = Field(default_factory=list, description="Edges along path")


class HealthResponse(BaseModel):
    """API health status response."""

    status: str = Field(default="healthy", description="System operational status")
    version: str = Field(default="0.1.0", description="Kautilya version")
    dataset_loaded: bool = Field(default=False, description="True if dataset is loaded")


class StatisticsResponse(BaseModel):
    """Investigation and triage queue aggregate statistics."""

    total_transactions: int = Field(description="Total transactions in system")
    total_wallets: int = Field(description="Total wallets in system")
    total_alerts: int = Field(description="Total alerts generated")
    tier_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Alert count by priority tier (critical, high, medium, low)",
    )
    status_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Alert count by status (new, in_review, escalated, closed)",
    )
    synthetic_alerts_count: int = Field(
        default=0,
        description="Count of alerts with synthetic input",
    )
    synthetic_alerts_percentage: float = Field(
        default=0.0,
        description="Percentage of alerts with synthetic input",
    )
    is_offline_mode: bool = Field(
        default=True,
        description="True if running in offline batch mode",
    )

