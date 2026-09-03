"""Investigative alert and entity ranking domain models.

Defines schemas for investigative alerts, ranked triage queues, and
filtering criteria.

Forensic Principles:
    1. An alert represents an investigative triage priority, NOT a legal
       accusation or proof of criminal guilt.
    2. Alerts support configurable priority tiers (Critical, High, Medium, Low).
    3. Entities utilizing synthetic network or temporal signals are tagged
       with `contains_synthetic_input = True`.
    4. Wallet alerts preserve entity aggregation disclaimers and non-accusation
       rules.
"""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, Field

from backend.domain.types import AlertStatus, PriorityTier


class InvestigativeAlert(BaseModel):
    """An alert generated for an entity requiring forensic review.

    Used by investigators to triage and track suspicious transactions
    and wallets across priority queues.
    """

    alert_id: str = Field(description="Unique alert identifier (e.g. 'alert-...')")
    entity_id: str = Field(description="Target entity ID (transaction ID or wallet address)")
    entity_type: str = Field(
        default="transaction",
        description="Type of entity ('transaction' or 'wallet')",
    )
    risk_score: float = Field(
        description="Investigative priority ranking score (0.0–100.0); NOT probability of guilt"
    )
    priority_tier: PriorityTier = Field(
        description="Priority tier (Critical, High, Medium, Low)"
    )
    tier_description: str | None = Field(
        default=None,
        description="Human-readable description of the assigned tier",
    )
    status: AlertStatus = Field(
        default=AlertStatus.NEW,
        description="Current lifecycle status of the alert",
    )
    headline: str = Field(description="Concise alert title for investigator dashboards")
    summary: str = Field(
        description="Natural-language summary of contributing evidence and risk rationale"
    )
    active_signals: list[str] = Field(
        default_factory=list,
        description="List of active analytical signals that contributed to this alert",
    )
    recommended_action: str = Field(
        description="Actionable triage recommendation for the investigator"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when the alert was generated (UTC)",
    )
    evidence_count: int = Field(
        default=0,
        description="Number of structured evidence records supporting this alert",
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Optional additional context (e.g. contributing transactions count)",
    )
    contains_synthetic_input: bool = Field(
        default=False,
        description="True if any input to this alert derived from synthetic network/temporal data",
    )
    forensic_disclaimer: str = Field(
        default=(
            "This alert represents an investigative triage prioritization, "
            "NOT a legal accusation, proof of guilt, or confirmed criminality."
        ),
        description="Mandatory forensic disclaimer",
    )


class RankedEntity(BaseModel):
    """An entity positioned in an investigator triage queue.

    Preserves 1-indexed queue ranking, priority tier, component signals,
    and synthetic provenance.
    """

    rank: int = Field(description="1-indexed priority ranking position in triage queue")
    entity_id: str = Field(description="Transaction ID or wallet address")
    entity_type: str = Field(
        default="transaction",
        description="'transaction' or 'wallet'",
    )
    risk_score: float = Field(
        description="Investigative priority score (0.0–100.0)"
    )
    priority_tier: PriorityTier = Field(
        description="Assigned investigative priority tier"
    )
    tier_description: str | None = Field(
        default=None,
        description="Description of why this tier applies",
    )
    active_signals: list[str] = Field(
        default_factory=list,
        description="Active signal names contributing to the score",
    )
    corroboration_count: int = Field(
        default=0,
        description="Number of independent signals contributing to the score",
    )
    contains_synthetic_input: bool = Field(
        default=False,
        description="True if any signal derived from synthetic data",
    )
    summary: str | None = Field(
        default=None,
        description="Brief summary rationale",
    )
    alert_id: str | None = Field(
        default=None,
        description="Associated alert ID, if generated",
    )


class AlertFilter(BaseModel):
    """Filter criteria for investigator alert queues."""

    min_tier: PriorityTier | None = Field(
        default=None,
        description="Minimum priority tier to include",
    )
    entity_type: str | None = Field(
        default=None,
        description="Filter by entity type ('transaction' or 'wallet')",
    )
    status: AlertStatus | None = Field(
        default=None,
        description="Filter by alert status (e.g. NEW, IN_REVIEW)",
    )
    include_synthetic: bool = Field(
        default=True,
        description="Whether to include alerts derived from synthetic data",
    )
    min_score: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Minimum risk score threshold",
    )
    limit: int | None = Field(
        default=None,
        gt=0,
        description="Maximum number of alerts to return",
    )


class RankingCriteria(BaseModel):
    """Configuration options for entity triage ranking."""

    secondary_sort_by_corroboration: bool = Field(
        default=True,
        description="If True, ties in score are broken by corroborating signal count descending",
    )
    min_tier: PriorityTier | None = Field(
        default=None,
        description="Filter out entities below this priority tier",
    )
    min_score: float | None = Field(
        default=None,
        ge=0.0,
        le=100.0,
        description="Filter out entities below this numerical score",
    )
    limit: int | None = Field(
        default=None,
        gt=0,
        description="Maximum number of ranked entities to return",
    )
