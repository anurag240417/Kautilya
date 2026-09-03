"""Transaction-to-IP correlation engine.

Links blockchain transactions with synthetic network observations to identify
candidate origin and relay nodes, compute decoupled correlation confidence
metrics, and produce auditable correlation domain records.

Forensic Rule:
    Correlation does NOT imply attribution. Temporal proximity and network
    observations alone must NOT be treated as proof of wallet or IP ownership.
    Every correlation record preserves ``is_synthetic=True`` and carries an
    explicit evidential disclaimer.
"""

from collections import defaultdict
from dataclasses import dataclass, field

import networkx as nx

from backend.correlation.temporal import (
    TemporalWindowConfig,
    compute_propagation_consistency,
    compute_temporal_proximity,
)
from backend.domain.correlation import (
    CandidateIPRole,
    CorrelationConfidenceMetrics,
    TransactionIPCorrelation,
)
from backend.domain.network import NetworkObservation
from backend.domain.transaction import Transaction
from backend.normalization.temporal import normalize_timestamp


@dataclass(frozen=True)
class CorrelationConfig:
    """Weights and window configuration for correlation confidence scoring."""

    temporal_weight: float = 0.40
    propagation_weight: float = 0.35
    topology_weight: float = 0.25
    temporal_config: TemporalWindowConfig = field(default_factory=TemporalWindowConfig)


DEFAULT_CORRELATION_CONFIG = CorrelationConfig()


def compute_topology_consistency(
    src_ip: str,
    dst_ip: str,
    topology: nx.Graph | None = None,
    ip_to_node_id: dict[str, int] | None = None,
) -> float:
    """Evaluate topological consistency between source and destination IPs.

    If a NetworkX graph of the P2P topology is provided:
    - Direct peer edge in topology: 1.0 (verified adjacent connection)
    - 2-hop neighbor: 0.75
    - 3+ hops neighbor: 0.40
    - Disconnected / no path: 0.15

    If no topology graph is provided:
    - Valid distinct IPs: baseline score 0.8 (standard plausible transmission)
    - Identical src and dst: 0.5 (loopback/self-connection)

    Args:
        src_ip: Source IP address.
        dst_ip: Destination IP address.
        topology: Optional network topology graph.
        ip_to_node_id: Optional mapping of IP string to graph node ID.

    Returns:
        Consistency score in range [0.0, 1.0].
    """
    if not src_ip or not dst_ip:
        return 0.0

    if src_ip == dst_ip:
        return 0.5

    if topology is not None:
        # Determine node identifiers
        u = ip_to_node_id.get(src_ip) if ip_to_node_id else src_ip
        v = ip_to_node_id.get(dst_ip) if ip_to_node_id else dst_ip

        if u in topology and v in topology:
            if topology.has_edge(u, v):
                return 1.0
            try:
                path_len = nx.shortest_path_length(topology, source=u, target=v)
                if path_len == 2:
                    return 0.75
                elif path_len >= 3:
                    return 0.40
            except nx.NetworkXNoPath:
                return 0.15

    # Default heuristic when topology graph is omitted or nodes not in graph
    return 0.80


def compute_confidence_metrics(
    temporal_proximity: float,
    propagation_consistency: float,
    topology_consistency: float,
    config: CorrelationConfig | None = None,
) -> CorrelationConfidenceMetrics:
    """Calculate composite correlation confidence from decomposed sub-metrics.

    All sub-metrics are guaranteed to be in range [0.0, 1.0]. The composite
    confidence is a normalized linear combination using configurable weights.

    Args:
        temporal_proximity: Proximity of observation to synthetic window [0, 1].
        propagation_consistency: Monotonicity and latency consistency [0, 1].
        topology_consistency: Topology path and connection plausibility [0, 1].
        config: Optional correlation weights configuration.

    Returns:
        CorrelationConfidenceMetrics instance.
    """
    cfg = config or DEFAULT_CORRELATION_CONFIG

    s_temp = max(0.0, min(1.0, float(temporal_proximity)))
    s_prop = max(0.0, min(1.0, float(propagation_consistency)))
    s_topo = max(0.0, min(1.0, float(topology_consistency)))

    weight_sum = cfg.temporal_weight + cfg.propagation_weight + cfg.topology_weight
    if weight_sum <= 0:
        composite = 0.0
    else:
        composite = (
            cfg.temporal_weight * s_temp
            + cfg.propagation_weight * s_prop
            + cfg.topology_weight * s_topo
        ) / weight_sum

    composite = max(0.0, min(1.0, float(composite)))

    return CorrelationConfidenceMetrics(
        temporal_proximity=s_temp,
        propagation_consistency=s_prop,
        topology_consistency=s_topo,
        composite_confidence=composite,
    )


def correlate_transaction_to_ips(
    tx: Transaction,
    observations: list[NetworkObservation],
    topology: nx.Graph | None = None,
    ip_to_node_id: dict[str, int] | None = None,
    config: CorrelationConfig | None = None,
) -> list[TransactionIPCorrelation]:
    """Correlate a blockchain transaction with observed network IPs.

    Identifies the candidate broadcast origin IP (the source of the earliest
    hop in the propagation sequence) and relaying nodes. Each correlation is
    assigned a distinct `correlation_confidence` score and decomposed metrics.

    Args:
        tx: Transaction domain record.
        observations: Synthetic network observations.
        topology: Optional P2P network graph.
        ip_to_node_id: Optional mapping of IP string to node ID in topology.
        config: Optional correlation configuration.

    Returns:
        List of TransactionIPCorrelation records with `is_synthetic=True`.
    """
    cfg = config or DEFAULT_CORRELATION_CONFIG
    tx_obs = [o for o in observations if o.txid == tx.txid]

    if not tx_obs:
        return []

    # Sort observations chronologically
    sorted_obs = sorted(tx_obs, key=lambda o: normalize_timestamp(o.timestamp))
    earliest_obs = sorted_obs[0]
    origin_ip = earliest_obs.src_ip

    # Global propagation consistency for this transaction wave
    prop_consistency = compute_propagation_consistency(sorted_obs, cfg.temporal_config)

    # Aggregate observations by IP
    ip_obs_map: dict[str, list[NetworkObservation]] = defaultdict(list)
    ip_roles: dict[str, CandidateIPRole] = {}
    ip_asns: dict[str, int | None] = {}
    ip_countries: dict[str, str | None] = {}

    for obs in sorted_obs:
        # Collect for src_ip
        ip_obs_map[obs.src_ip].append(obs)
        if obs.src_ip == origin_ip:
            ip_roles[obs.src_ip] = CandidateIPRole.ORIGIN
        elif obs.src_ip not in ip_roles:
            ip_roles[obs.src_ip] = CandidateIPRole.RELAY

        # Collect for dst_ip
        ip_obs_map[obs.dst_ip].append(obs)
        if obs.dst_ip not in ip_roles:
            ip_roles[obs.dst_ip] = CandidateIPRole.RELAY

        # Record GeoIP metadata if present
        if obs.asn is not None and obs.dst_ip not in ip_asns:
            ip_asns[obs.dst_ip] = obs.asn
        if obs.country is not None and obs.dst_ip not in ip_countries:
            ip_countries[obs.dst_ip] = obs.country

    results: list[TransactionIPCorrelation] = []

    for ip, obs_list in ip_obs_map.items():
        role = ip_roles.get(ip, CandidateIPRole.RELAY)

        # Compute temporal proximity: mean proximity of observations for this IP
        prox_scores = [
            compute_temporal_proximity(
                normalize_timestamp(o.timestamp), tx.time_step, cfg.temporal_config
            )
            for o in obs_list
        ]
        mean_prox = sum(prox_scores) / len(prox_scores) if prox_scores else 1.0

        # Compute topology consistency across hops involving this IP
        topo_scores = [
            compute_topology_consistency(o.src_ip, o.dst_ip, topology, ip_to_node_id)
            for o in obs_list
        ]
        mean_topo = sum(topo_scores) / len(topo_scores) if topo_scores else 0.8

        metrics = compute_confidence_metrics(
            temporal_proximity=mean_prox,
            propagation_consistency=prop_consistency,
            topology_consistency=mean_topo,
            config=cfg,
        )

        timestamps = [normalize_timestamp(o.timestamp) for o in obs_list]
        earliest_ts = min(timestamps)
        latest_ts = max(timestamps)

        results.append(
            TransactionIPCorrelation(
                txid=tx.txid,
                ip=ip,
                role=role,
                correlation_confidence=metrics.composite_confidence,
                metrics=metrics,
                observations_count=len(obs_list),
                earliest_observation=earliest_ts,
                latest_observation=latest_ts,
                asn=ip_asns.get(ip),
                country=ip_countries.get(ip),
                is_synthetic=True,
            )
        )

    # Sort so ORIGIN is first, then by descending confidence
    results.sort(
        key=lambda c: (0 if c.role == CandidateIPRole.ORIGIN else 1, -c.correlation_confidence)
    )
    return results


def correlate_ip_to_transactions(
    ip: str,
    transactions: list[Transaction],
    observations: list[NetworkObservation],
    topology: nx.Graph | None = None,
    ip_to_node_id: dict[str, int] | None = None,
    config: CorrelationConfig | None = None,
) -> list[TransactionIPCorrelation]:
    """Find all transactions correlated with a given IP address.

    Args:
        ip: Target IP address to investigate.
        transactions: Blockchain transactions to cross-reference.
        observations: Synthetic network observations.
        topology: Optional P2P network graph.
        ip_to_node_id: Optional mapping of IP string to node ID in topology.
        config: Optional correlation configuration.

    Returns:
        List of TransactionIPCorrelation records involving the specified IP.
    """
    tx_by_id = {tx.txid: tx for tx in transactions}
    ip_obs = [o for o in observations if o.src_ip == ip or o.dst_ip == ip]

    tx_ids_seen = {o.txid for o in ip_obs if o.txid in tx_by_id}

    all_correlations: list[TransactionIPCorrelation] = []
    for txid in tx_ids_seen:
        tx = tx_by_id[txid]
        tx_corrs = correlate_transaction_to_ips(
            tx=tx,
            observations=observations,
            topology=topology,
            ip_to_node_id=ip_to_node_id,
            config=config,
        )
        # Filter for only the targeted IP
        for c in tx_corrs:
            if c.ip == ip:
                all_correlations.append(c)

    all_correlations.sort(key=lambda c: -c.correlation_confidence)
    return all_correlations
