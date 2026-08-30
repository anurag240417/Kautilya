"""Scenario definitions for synthetic network generation.

Defines different network behavior scenarios (normal activity,
IP clustering, Tor usage, VPN usage, etc.) that shape how
transactions propagate through the synthetic network.

Scenarios are conditioned on the transaction's label:
    - Illicit transactions → anomalous network patterns
    - Licit transactions → normal network patterns
    - Unknown transactions → mostly normal, small anomaly chance

This conditioning creates a learnable signal for ML models
in the network layer while maintaining overlap/noise so the
signal is non-trivial.

See CONTEXT.md §3 and DATASET.md §9 for generation rules.
"""

import random
from dataclasses import dataclass
from enum import StrEnum

from backend.domain.types import EntityClass
from backend.generator.node_pool import NetworkNode, NodeType


class ScenarioType(StrEnum):
    """Types of network behavior scenarios."""

    NORMAL = "normal"
    IP_CLUSTERING = "ip_clustering"
    TOR_ORIGIN = "tor_origin"
    VPN_ORIGIN = "vpn_origin"
    LOW_FAN_OUT = "low_fan_out"
    GEO_CONCENTRATION = "geo_concentration"
    RAPID_BROADCAST = "rapid_broadcast"


@dataclass(frozen=True)
class ScenarioParams:
    """Parameters controlling a scenario's propagation behavior.

    These override the defaults in the propagation module to create
    distinguishable network patterns.
    """

    scenario_type: ScenarioType
    prefer_node_type: NodeType | None = None
    max_hops: int = 8
    max_fan_out: int = 4
    delay_min_ms: float = 50.0
    delay_max_ms: float = 400.0
    restrict_countries: list[str] | None = None


# --- Scenario definitions ---

_NORMAL_SCENARIO = ScenarioParams(
    scenario_type=ScenarioType.NORMAL,
    prefer_node_type=None,
    max_hops=8,
    max_fan_out=4,
    delay_min_ms=50.0,
    delay_max_ms=400.0,
)

_IP_CLUSTERING_SCENARIO = ScenarioParams(
    scenario_type=ScenarioType.IP_CLUSTERING,
    prefer_node_type=None,
    max_hops=3,
    max_fan_out=2,
    delay_min_ms=20.0,
    delay_max_ms=150.0,
)

_TOR_ORIGIN_SCENARIO = ScenarioParams(
    scenario_type=ScenarioType.TOR_ORIGIN,
    prefer_node_type=NodeType.TOR_EXIT,
    max_hops=4,
    max_fan_out=2,
    delay_min_ms=200.0,
    delay_max_ms=1500.0,
)

_VPN_ORIGIN_SCENARIO = ScenarioParams(
    scenario_type=ScenarioType.VPN_ORIGIN,
    prefer_node_type=NodeType.VPN,
    max_hops=5,
    max_fan_out=3,
    delay_min_ms=80.0,
    delay_max_ms=600.0,
)

_LOW_FAN_OUT_SCENARIO = ScenarioParams(
    scenario_type=ScenarioType.LOW_FAN_OUT,
    prefer_node_type=None,
    max_hops=6,
    max_fan_out=1,
    delay_min_ms=30.0,
    delay_max_ms=200.0,
)

_GEO_CONCENTRATION_SCENARIO = ScenarioParams(
    scenario_type=ScenarioType.GEO_CONCENTRATION,
    prefer_node_type=None,
    max_hops=5,
    max_fan_out=3,
    delay_min_ms=10.0,
    delay_max_ms=80.0,
    restrict_countries=["RU", "UA", "RO", "CZ"],
)

_RAPID_BROADCAST_SCENARIO = ScenarioParams(
    scenario_type=ScenarioType.RAPID_BROADCAST,
    prefer_node_type=None,
    max_hops=3,
    max_fan_out=8,
    delay_min_ms=5.0,
    delay_max_ms=30.0,
)


# Anomalous scenarios and their relative weights for illicit transactions.
_ILLICIT_SCENARIOS: list[tuple[ScenarioParams, float]] = [
    (_TOR_ORIGIN_SCENARIO, 0.30),
    (_VPN_ORIGIN_SCENARIO, 0.20),
    (_IP_CLUSTERING_SCENARIO, 0.20),
    (_LOW_FAN_OUT_SCENARIO, 0.15),
    (_GEO_CONCENTRATION_SCENARIO, 0.10),
    (_RAPID_BROADCAST_SCENARIO, 0.05),
]

# Probability that a licit transaction gets a mildly anomalous pattern
# (noise to prevent trivial separation).
_LICIT_ANOMALY_CHANCE = 0.05

# Probability that an unknown transaction gets an anomalous pattern.
_UNKNOWN_ANOMALY_CHANCE = 0.10


def select_scenario(
    label: EntityClass | None,
    rng: random.Random,
) -> ScenarioParams:
    """Select a network behavior scenario based on the transaction label.

    Illicit transactions are assigned anomalous scenarios to create
    learnable network-layer signals for ML. Licit and unknown
    transactions are mostly normal with a small anomaly chance
    (noise) to prevent trivial separation.

    Args:
        label: Transaction class label (1=Illicit, 2=Licit, 3=Unknown, None).
        rng: Seeded Random instance.

    Returns:
        A ``ScenarioParams`` instance defining the propagation behavior.
    """
    if label == EntityClass.ILLICIT:
        # Illicit → always an anomalous scenario
        scenarios = [s for s, _ in _ILLICIT_SCENARIOS]
        weights = [w for _, w in _ILLICIT_SCENARIOS]
        return rng.choices(scenarios, weights=weights, k=1)[0]

    if label == EntityClass.LICIT:
        # Licit → mostly normal, small anomaly chance
        if rng.random() < _LICIT_ANOMALY_CHANCE:
            scenarios = [s for s, _ in _ILLICIT_SCENARIOS]
            weights = [w for _, w in _ILLICIT_SCENARIOS]
            return rng.choices(scenarios, weights=weights, k=1)[0]
        return _NORMAL_SCENARIO

    # Unknown or None → slightly higher anomaly chance than licit
    if rng.random() < _UNKNOWN_ANOMALY_CHANCE:
        scenarios = [s for s, _ in _ILLICIT_SCENARIOS]
        weights = [w for _, w in _ILLICIT_SCENARIOS]
        return rng.choices(scenarios, weights=weights, k=1)[0]
    return _NORMAL_SCENARIO


def get_all_scenario_types() -> list[ScenarioType]:
    """Return all defined scenario types."""
    return list(ScenarioType)
