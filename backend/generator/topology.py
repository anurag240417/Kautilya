"""Network topology generation.

Creates a synthetic Bitcoin P2P network topology connecting nodes
from the node pool into a realistic scale-free graph structure.

Uses the Barabási–Albert model which produces degree distributions
similar to real P2P networks where some nodes (hubs) have many
more connections than average.

See DATASET.md §9 for generation context.
"""

import random

import networkx as nx

from backend.generator.node_pool import NetworkNode


def build_topology(
    nodes: list[NetworkNode],
    seed: int,
    peers_per_node: int = 8,
) -> nx.Graph:
    """Build a scale-free P2P topology from a node pool.

    Uses the Barabási–Albert preferential attachment model to
    create a connected, undirected graph where node degrees follow
    a power-law distribution — realistic for Bitcoin P2P networks.

    Node attributes (ip, asn, country, node_type) from the
    ``NetworkNode`` objects are stored on each graph node.

    Args:
        nodes: List of ``NetworkNode`` instances forming the pool.
        seed: Random seed for reproducible topology generation.
        peers_per_node: Number of edges each new node creates when
            joining the network (Barabási–Albert ``m`` parameter).
            Must be >= 1 and < len(nodes). Default 8.

    Returns:
        An undirected ``nx.Graph`` with node IDs matching
        ``NetworkNode.node_id`` and node attributes populated.

    Raises:
        ValueError: If the node pool is too small or peers_per_node
            is invalid.
    """
    n = len(nodes)
    if n < 2:
        msg = f"Need at least 2 nodes for a topology, got {n}"
        raise ValueError(msg)

    if peers_per_node < 1:
        msg = f"peers_per_node must be >= 1, got {peers_per_node}"
        raise ValueError(msg)

    # BA model requires m < n. Clamp if the pool is small.
    m = min(peers_per_node, n - 1)

    # Generate the BA graph.
    # NetworkX BA model creates nodes 0..n-1, matching our node_id scheme.
    G = nx.barabasi_albert_graph(n, m, seed=seed)

    # Attach NetworkNode metadata to each graph node.
    node_lookup = {node.node_id: node for node in nodes}
    for node_id in G.nodes():
        net_node = node_lookup[node_id]
        G.nodes[node_id]["ip"] = net_node.ip
        G.nodes[node_id]["port"] = net_node.port
        G.nodes[node_id]["asn"] = net_node.asn
        G.nodes[node_id]["country"] = net_node.country
        G.nodes[node_id]["node_type"] = net_node.node_type

    return G


def get_neighbors(topology: nx.Graph, node_id: int) -> list[int]:
    """Return the neighbor node IDs for a given node.

    Args:
        topology: The P2P topology graph.
        node_id: The node whose neighbors to retrieve.

    Returns:
        List of neighbor node IDs.

    Raises:
        KeyError: If node_id is not in the topology.
    """
    if node_id not in topology:
        msg = f"Node {node_id} not found in topology"
        raise KeyError(msg)
    return list(topology.neighbors(node_id))


def get_topology_stats(topology: nx.Graph) -> dict:
    """Compute basic statistics about the topology.

    Returns:
        Dictionary with node_count, edge_count, avg_degree,
        min_degree, max_degree, and is_connected.
    """
    degrees = [d for _, d in topology.degree()]
    return {
        "node_count": topology.number_of_nodes(),
        "edge_count": topology.number_of_edges(),
        "avg_degree": sum(degrees) / len(degrees) if degrees else 0,
        "min_degree": min(degrees) if degrees else 0,
        "max_degree": max(degrees) if degrees else 0,
        "is_connected": nx.is_connected(topology),
    }
