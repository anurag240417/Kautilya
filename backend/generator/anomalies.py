"""Controlled anomaly injection for synthetic network data.

Applies scenario-specific modifications to the propagation process.
This module bridges scenarios (what behavior to produce) with the
propagation engine (how to produce it).

The key function ``generate_anomalous_observations`` takes a scenario
and produces NetworkObservation records with the scenario's
characteristic propagation pattern.

See DATASET.md §9–10 and CONTEXT.md §3 for generation rules.
"""

import math
import random
from datetime import timedelta

from backend.domain.network import NetworkObservation
from backend.generator.node_pool import NetworkNode
from backend.generator.propagation import (
    generate_observations_for_transaction,
    pick_origin_node,
    simulate_propagation,
    _pick_origin_timestamp,
    _pick_script_type,
    PropagationHop,
)
from backend.generator.scenarios import ScenarioParams, ScenarioType
from backend.generator.temporal import time_step_to_datetime_range

import networkx as nx


def _pick_delay_for_scenario(
    rng: random.Random,
    params: ScenarioParams,
) -> float:
    """Generate a relay delay using the scenario's delay range."""
    log_min = math.log(max(params.delay_min_ms, 0.1))
    log_max = math.log(max(params.delay_max_ms, 0.2))
    return math.exp(rng.uniform(log_min, log_max))


def _filter_nodes_by_country(
    nodes: list[NetworkNode],
    countries: list[str],
) -> list[NetworkNode]:
    """Filter nodes to only those in the specified countries."""
    filtered = [n for n in nodes if n.country in countries]
    return filtered if filtered else nodes  # Fall back to full pool if empty


def generate_scenario_observations(
    txid: int,
    time_step: int,
    label: int | None,
    scenario: ScenarioParams,
    topology: nx.Graph,
    nodes: list[NetworkNode],
    node_lookup: dict[int, NetworkNode],
    rng: random.Random,
    generation_run_id: str | None = None,
    generator_version: str | None = None,
) -> tuple[list[NetworkObservation], int]:
    """Generate network observations for a transaction using a specific scenario.

    Applies the scenario's parameters (preferred origin, fan-out, delays,
    geographic restrictions) to produce characteristic network patterns.

    Args:
        txid: Elliptic++ transaction ID.
        time_step: Transaction's time step (1–49).
        label: Transaction class label (for provenance, not used for behavior).
        scenario: The scenario parameters to apply.
        topology: The P2P network topology graph.
        nodes: Full node pool.
        node_lookup: Mapping of node_id → NetworkNode.
        rng: Seeded Random instance.
        generation_run_id: Generation run ID (provenance).
        generator_version: Generator version (provenance).

    Returns:
        Tuple of (list of NetworkObservation records, origin_node_id).
    """
    # Select origin node based on scenario preference
    origin_id = pick_origin_node(
        nodes, rng, prefer_type=scenario.prefer_node_type,
    )

    # For geo_concentration, try to pick an origin from restricted countries
    if scenario.restrict_countries:
        geo_nodes = _filter_nodes_by_country(nodes, scenario.restrict_countries)
        if geo_nodes:
            origin_id = pick_origin_node(geo_nodes, rng)

    # Generate observations using scenario-specific parameters
    observations = generate_observations_for_transaction(
        txid=txid,
        time_step=time_step,
        origin_node_id=origin_id,
        topology=topology,
        node_lookup=node_lookup,
        rng=rng,
        max_hops=scenario.max_hops,
        max_fan_out=scenario.max_fan_out,
        generation_run_id=generation_run_id,
        scenario_id=scenario.scenario_type.value,
        generator_version=generator_version,
    )

    return observations, origin_id
