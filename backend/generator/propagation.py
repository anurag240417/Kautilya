"""Transaction propagation simulation.

Simulates how Bitcoin transactions propagate through the synthetic
network topology, generating observation timestamps at each relay hop.

The simulation uses a BFS-like flood from an origin node through
the topology graph. Each hop produces a ``PropagationHop`` record
with a realistic relay delay. These hops are then converted into
``NetworkObservation`` domain objects with full provenance.

See DATASET.md §9 and CONTEXT.md §3 for generation rules.
"""

import random
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta

import networkx as nx

from backend.domain.network import NetworkObservation
from backend.domain.types import ScriptType
from backend.generator.node_pool import NetworkNode, NodeType
from backend.generator.temporal import time_step_to_datetime_range


@dataclass(frozen=True)
class PropagationHop:
    """A single relay hop in a transaction propagation.

    Records the source and destination node IDs, the hop index
    (0 = first relay from origin), and the cumulative delay in
    milliseconds from the origin timestamp.
    """

    from_node_id: int
    to_node_id: int
    hop_index: int
    cumulative_delay_ms: float


# --- Delay configuration ---

# Normal relay delay range (milliseconds).
_NORMAL_DELAY_MIN_MS = 50.0
_NORMAL_DELAY_MAX_MS = 400.0

# Bitcoin script type distribution (realistic mainnet proportions).
_SCRIPT_TYPE_WEIGHTS: list[tuple[ScriptType, float]] = [
    (ScriptType.P2PKH, 0.35),
    (ScriptType.P2SH, 0.25),
    (ScriptType.P2WPKH, 0.30),
    (ScriptType.TAPROOT, 0.10),
]


def _pick_relay_delay(rng: random.Random) -> float:
    """Generate a realistic relay delay in milliseconds.

    Uses a log-uniform distribution to model the fat-tailed nature
    of network latencies — most hops are fast, some are slow.
    """
    import math

    log_min = math.log(_NORMAL_DELAY_MIN_MS)
    log_max = math.log(_NORMAL_DELAY_MAX_MS)
    return math.exp(rng.uniform(log_min, log_max))


def _pick_script_type(rng: random.Random) -> ScriptType:
    """Pick a Bitcoin script type from realistic distribution."""
    types = [st for st, _ in _SCRIPT_TYPE_WEIGHTS]
    weights = [w for _, w in _SCRIPT_TYPE_WEIGHTS]
    return rng.choices(types, weights=weights, k=1)[0]


def _pick_origin_timestamp(
    rng: random.Random,
    time_step: int,
) -> datetime:
    """Pick a random timestamp within the time step window.

    The transaction could have occurred at any point within
    the two-week window for its time step.
    """
    start, end = time_step_to_datetime_range(time_step)
    window_seconds = (end - start).total_seconds()
    offset = rng.uniform(0, window_seconds)
    return start + timedelta(seconds=offset)


def simulate_propagation(
    origin_node_id: int,
    topology: nx.Graph,
    rng: random.Random,
    max_hops: int = 8,
    max_fan_out: int = 4,
) -> list[PropagationHop]:
    """Simulate transaction propagation through the P2P topology.

    Uses a bounded BFS flood from the origin node. At each level,
    each reached node relays to up to ``max_fan_out`` of its
    neighbors that have not yet been reached.

    Args:
        origin_node_id: Node that first broadcasts the transaction.
        topology: The P2P network topology graph.
        rng: Seeded Random instance for reproducibility.
        max_hops: Maximum propagation depth (BFS levels).
        max_fan_out: Maximum number of peers each node relays to.

    Returns:
        List of ``PropagationHop`` records in temporal order.

    Raises:
        KeyError: If origin_node_id is not in the topology.
    """
    if origin_node_id not in topology:
        msg = f"Origin node {origin_node_id} not found in topology"
        raise KeyError(msg)

    hops: list[PropagationHop] = []
    visited: set[int] = {origin_node_id}

    # Queue entries: (node_id, cumulative_delay_ms, current_hop_depth)
    queue: deque[tuple[int, float, int]] = deque()
    queue.append((origin_node_id, 0.0, 0))

    while queue:
        current_node, current_delay, depth = queue.popleft()

        if depth >= max_hops:
            continue

        # Get unvisited neighbors and pick a subset to relay to
        neighbors = [n for n in topology.neighbors(current_node) if n not in visited]
        if not neighbors:
            continue

        # Shuffle and pick up to max_fan_out neighbors
        rng.shuffle(neighbors)
        relay_targets = neighbors[:max_fan_out]

        for target in relay_targets:
            visited.add(target)
            hop_delay = _pick_relay_delay(rng)
            cumulative = current_delay + hop_delay

            hop = PropagationHop(
                from_node_id=current_node,
                to_node_id=target,
                hop_index=len(hops),
                cumulative_delay_ms=cumulative,
            )
            hops.append(hop)
            queue.append((target, cumulative, depth + 1))

    return hops


def generate_observations_for_transaction(
    txid: int,
    time_step: int,
    origin_node_id: int,
    topology: nx.Graph,
    node_lookup: dict[int, NetworkNode],
    rng: random.Random,
    max_hops: int = 8,
    max_fan_out: int = 4,
    generation_run_id: str | None = None,
    scenario_id: str | None = None,
    generator_version: str | None = None,
) -> list[NetworkObservation]:
    """Generate all network observations for a single transaction.

    Combines propagation simulation with temporal mapping to produce
    complete ``NetworkObservation`` records with timestamps, IPs,
    ports, ASN, country, and script type.

    Args:
        txid: Elliptic++ transaction ID.
        time_step: Transaction's time step (1–49).
        origin_node_id: Node that first broadcasts the transaction.
        topology: The P2P network topology graph.
        node_lookup: Mapping of node_id → NetworkNode for metadata.
        rng: Seeded Random instance for reproducibility.
        max_hops: Maximum propagation depth.
        max_fan_out: Maximum peers relayed to per hop.
        generation_run_id: ID of this generation run (provenance).
        scenario_id: Scenario identifier (provenance).
        generator_version: Generator version string (provenance).

    Returns:
        List of ``NetworkObservation`` records for this transaction.
    """
    # Simulate propagation
    hops = simulate_propagation(
        origin_node_id=origin_node_id,
        topology=topology,
        rng=rng,
        max_hops=max_hops,
        max_fan_out=max_fan_out,
    )

    if not hops:
        return []

    # Pick base timestamp and script type for this transaction
    base_timestamp = _pick_origin_timestamp(rng, time_step)
    script_type = _pick_script_type(rng)

    observations: list[NetworkObservation] = []

    for hop in hops:
        src_node = node_lookup[hop.from_node_id]
        dst_node = node_lookup[hop.to_node_id]

        # Compute observation timestamp from base + cumulative delay
        obs_timestamp = base_timestamp + timedelta(milliseconds=hop.cumulative_delay_ms)

        obs = NetworkObservation(
            txid=txid,
            src_ip=src_node.ip,
            dst_ip=dst_node.ip,
            src_port=src_node.port,
            dst_port=dst_node.port,
            protocol="TCP",
            timestamp=obs_timestamp,
            asn=dst_node.asn,
            country=dst_node.country,
            script_type=script_type,
            is_synthetic=True,
            generation_run_id=generation_run_id,
            scenario_id=scenario_id,
            generator_version=generator_version,
        )
        observations.append(obs)

    return observations


def pick_origin_node(
    nodes: list[NetworkNode],
    rng: random.Random,
    prefer_type: NodeType | None = None,
) -> int:
    """Select an origin node for a transaction broadcast.

    If ``prefer_type`` is specified, nodes of that type are weighted
    5× higher. This allows label-driven origin selection (e.g. illicit
    transactions preferring tor_exit nodes).

    Args:
        nodes: The full node pool.
        rng: Seeded Random instance.
        prefer_type: Optional node type to prefer.

    Returns:
        The ``node_id`` of the selected origin node.
    """
    if prefer_type is not None:
        weights = [5.0 if n.node_type == prefer_type else 1.0 for n in nodes]
    else:
        weights = [1.0] * len(nodes)

    chosen = rng.choices(nodes, weights=weights, k=1)[0]
    return chosen.node_id
