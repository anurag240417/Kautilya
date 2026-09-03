"""Correlation domain models.

Defines canonical schemas for temporal correlations, transaction-to-IP
correlations, and correlation confidence metrics.

Forensic Rule:
    Correlation does NOT imply attribution. Temporal correlation alone
    must NOT be treated as proof of wallet/IP ownership. All records
    derived from synthetic network data preserve ``is_synthetic=True``
    and carry explicit evidential caveats.
"""

from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class CandidateIPRole(StrEnum):
    """Role of an IP in observed transaction propagation."""

    ORIGIN = "origin"  # Suspected broadcast origin (earliest hop source)
    RELAY = "relay"  # Intermediate forwarding relay node
    DESTINATION = "destination"  # Receiver/destination peer
    OBSERVER = "observer"  # Generic peer observation


class CorrelationConfidenceMetrics(BaseModel):
    """Granular metric breakdown supporting correlation confidence.

    Separates temporal proximity, propagation timing consistency, and
    topology consistency into interpretable components in range [0.0, 1.0].
    """

    temporal_proximity: float = Field(
        ge=0.0,
        le=1.0,
        description="Temporal alignment between observation and synthetic transaction window",
    )
    propagation_consistency: float = Field(
        ge=0.0,
        le=1.0,
        description="Plausibility and monotonicity of propagation delays across hops",
    )
    topology_consistency: float = Field(
        ge=0.0,
        le=1.0,
        description="Consistency with known P2P network topology routing or valid port heuristics",
    )
    composite_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Weighted synthesis of correlation signals; distinct from risk or illicit probability"
        ),
    )


class TemporalCorrelation(BaseModel):
    """Temporal correlation between a transaction and a network observation."""

    txid: int = Field(description="Elliptic++ transaction ID")
    time_step: int = Field(ge=1, le=49, description="Transaction time_step index")
    observation_timestamp: datetime = Field(description="Normalized UTC observation timestamp")
    window_start: datetime = Field(
        description="Start of synthetic transaction datetime window (UTC)"
    )
    window_end: datetime = Field(description="End of synthetic transaction datetime window (UTC)")
    delta_seconds: float = Field(
        description="Time delta from the window boundary in seconds (0.0 if inside window)"
    )
    temporal_proximity: float = Field(
        ge=0.0,
        le=1.0,
        description="Temporal proximity score [0.0, 1.0]",
    )
    is_synthetic: bool = Field(
        default=True,
        description="Always True. Network layer and temporal mapping are synthetic.",
    )
    disclaimer: str = Field(
        default="Temporal correlation alone must NOT be treated as proof of wallet/IP ownership.",
        description="Forensic evidential constraint disclaimer",
    )


class TransactionIPCorrelation(BaseModel):
    """Inferred relationship between a transaction and an IP address.

    Captures candidate origin or relay status with explicit confidence
    metrics and forensic disclaimers.
    """

    txid: int = Field(description="Elliptic++ transaction ID")
    ip: str = Field(description="Candidate IP address (synthetic)")
    role: CandidateIPRole = Field(description="Role of this IP in observed propagation")
    correlation_confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence in the transaction-to-IP correlation [0.0, 1.0]",
    )
    metrics: CorrelationConfidenceMetrics = Field(
        description="Decomposed confidence metrics"
    )
    observations_count: int = Field(
        ge=1, description="Number of network observations linking this IP and transaction"
    )
    earliest_observation: datetime = Field(
        description="Earliest timestamp linking this IP to the transaction (UTC)"
    )
    latest_observation: datetime = Field(
        description="Latest timestamp linking this IP to the transaction (UTC)"
    )
    asn: int | None = Field(default=None, description="Autonomous System Number (local GeoIP)")
    country: str | None = Field(default=None, description="Country code (local GeoIP)")
    is_synthetic: bool = Field(
        default=True,
        description="Always True. Derived from synthetic network observations.",
    )
    disclaimer: str = Field(
        default="Temporal correlation alone must NOT be treated as proof of wallet/IP ownership.",
        description="Mandatory forensic disclaimer",
    )
