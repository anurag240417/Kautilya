"""Graph feature extraction.

Computes graph-derived features (degree, centrality, clustering,
PageRank, HITS) for nodes in the investigation graph. These features
feed into the M2 (blockchain + graph) experimental model stage as
described in CONTEXT.md — Experimental Progression.

All functions accept a ``ChainTraceGraph`` and return feature
dictionaries or ``NodeFeatures`` dataclass instances.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from backend.graph.builder import ChainTraceGraph


@dataclass
class NodeFeatures:
    """Graph-derived features for a single node.

    Attributes cover degree statistics, centrality measures, and
    clustering — the standard investigative graph signals.
    """

    node_id: str | int
    node_type: str
    in_degree: int = 0
    out_degree: int = 0
    total_degree: int = 0
    in_degree_by_relationship: dict[str, int] = field(default_factory=dict)
    out_degree_by_relationship: dict[str, int] = field(default_factory=dict)
    pagerank: float = 0.0
    betweenness_centrality: float = 0.0
    clustering_coefficient: float = 0.0
    hub_score: float = 0.0
    authority_score: float = 0.0


# ------------------------------------------------------------------
# Individual feature computations
# ------------------------------------------------------------------


def compute_degree_features(
    graph: ChainTraceGraph, node_id: str | int
) -> dict:
    """Compute degree-based features for a single node.

    Returns in-degree, out-degree, total degree, plus
    per-relationship-type breakdowns.

    Returns an empty dict if the node does not exist.
    """
    G = graph.G
    if node_id not in G:
        return {}

    in_deg: int = G.in_degree(node_id)
    out_deg: int = G.out_degree(node_id)

    # Per-relationship breakdown for incoming edges
    in_by_rel: dict[str, int] = {}
    for _, _, data in G.in_edges(node_id, data=True):
        rel = data.get("relationship", "unknown")
        in_by_rel[rel] = in_by_rel.get(rel, 0) + 1

    # Per-relationship breakdown for outgoing edges
    out_by_rel: dict[str, int] = {}
    for _, _, data in G.out_edges(node_id, data=True):
        rel = data.get("relationship", "unknown")
        out_by_rel[rel] = out_by_rel.get(rel, 0) + 1

    return {
        "in_degree": in_deg,
        "out_degree": out_deg,
        "total_degree": in_deg + out_deg,
        "in_degree_by_relationship": in_by_rel,
        "out_degree_by_relationship": out_by_rel,
    }


def compute_pagerank(
    graph: ChainTraceGraph, alpha: float = 0.85
) -> dict[str | int, float]:
    """Compute PageRank for all nodes in the graph.

    Higher PageRank indicates a node that is more "central" in the
    flow of transactions — potentially a mixing service or high-volume
    intermediary.

    Args:
        graph: The ChainTrace investigation graph.
        alpha: Damping factor (default 0.85).

    Returns:
        Dictionary mapping node IDs to PageRank scores.
    """
    if len(graph.G) == 0:
        return {}
    return nx.pagerank(graph.G, alpha=alpha)


def compute_betweenness_centrality(
    graph: ChainTraceGraph, k: int | None = None
) -> dict[str | int, float]:
    """Compute betweenness centrality for all nodes.

    Identifies nodes that act as bridges between different parts
    of the graph — potentially mixing services or intermediaries.

    Args:
        graph: The ChainTrace investigation graph.
        k: Number of source nodes to sample for approximation.
            ``None`` computes exact centrality.

    Returns:
        Dictionary mapping node IDs to betweenness centrality scores.
    """
    if len(graph.G) == 0:
        return {}
    return nx.betweenness_centrality(graph.G, k=k)


def compute_clustering_coefficients(
    graph: ChainTraceGraph,
) -> dict[str | int, float]:
    """Compute clustering coefficients for all nodes.

    Uses a simple undirected projection of the MultiDiGraph since
    clustering coefficient is defined for undirected graphs.

    Returns:
        Dictionary mapping node IDs to clustering coefficients.
    """
    if len(graph.G) == 0:
        return {}
    # Convert to simple undirected Graph (collapses multi-edges)
    simple_undirected = nx.Graph(graph.G)
    return nx.clustering(simple_undirected)


def compute_hits(
    graph: ChainTraceGraph,
) -> tuple[dict[str | int, float], dict[str | int, float]]:
    """Compute HITS hub and authority scores.

    Hubs point to many authorities (e.g., wallets funding many
    transactions). Authorities are pointed to by many hubs (e.g.,
    transactions receiving from many sources).

    Returns:
        Tuple of (hub_scores, authority_scores).
    """
    if len(graph.G) == 0:
        return {}, {}
    hubs, authorities = nx.hits(graph.G, max_iter=100, normalized=True)
    return hubs, authorities


# ------------------------------------------------------------------
# Composite feature extraction
# ------------------------------------------------------------------


def extract_node_features(
    graph: ChainTraceGraph, node_id: str | int
) -> NodeFeatures | None:
    """Extract all graph-derived features for a single node.

    Combines degree, centrality, clustering, and HITS metrics.
    Returns ``None`` if the node does not exist in the graph.

    Note:
        For bulk extraction across many nodes, use
        ``extract_all_features()`` which computes graph-wide
        metrics only once.
    """
    if node_id not in graph.G:
        return None

    node_type = graph.G.nodes[node_id].get("type", "unknown")
    degree_info = compute_degree_features(graph, node_id)

    pagerank = compute_pagerank(graph)
    betweenness = compute_betweenness_centrality(graph)
    clustering = compute_clustering_coefficients(graph)
    hubs, authorities = compute_hits(graph)

    return NodeFeatures(
        node_id=node_id,
        node_type=node_type,
        in_degree=degree_info["in_degree"],
        out_degree=degree_info["out_degree"],
        total_degree=degree_info["total_degree"],
        in_degree_by_relationship=degree_info["in_degree_by_relationship"],
        out_degree_by_relationship=degree_info["out_degree_by_relationship"],
        pagerank=pagerank.get(node_id, 0.0),
        betweenness_centrality=betweenness.get(node_id, 0.0),
        clustering_coefficient=clustering.get(node_id, 0.0),
        hub_score=hubs.get(node_id, 0.0),
        authority_score=authorities.get(node_id, 0.0),
    )


def extract_all_features(
    graph: ChainTraceGraph,
) -> dict[str | int, NodeFeatures]:
    """Extract graph-derived features for every node in the graph.

    Computes all graph-wide metrics once and distributes results
    across nodes. Much more efficient than calling
    ``extract_node_features()`` per-node.

    Returns:
        Dictionary mapping node IDs to their ``NodeFeatures``.
    """
    if len(graph.G) == 0:
        return {}

    # Compute graph-wide metrics once
    pagerank = compute_pagerank(graph)
    betweenness = compute_betweenness_centrality(graph)
    clustering = compute_clustering_coefficients(graph)
    hubs, authorities = compute_hits(graph)

    result: dict[str | int, NodeFeatures] = {}
    for node_id in graph.G.nodes():
        node_type = graph.G.nodes[node_id].get("type", "unknown")
        degree_info = compute_degree_features(graph, node_id)

        result[node_id] = NodeFeatures(
            node_id=node_id,
            node_type=node_type,
            in_degree=degree_info["in_degree"],
            out_degree=degree_info["out_degree"],
            total_degree=degree_info["total_degree"],
            in_degree_by_relationship=degree_info["in_degree_by_relationship"],
            out_degree_by_relationship=degree_info["out_degree_by_relationship"],
            pagerank=pagerank.get(node_id, 0.0),
            betweenness_centrality=betweenness.get(node_id, 0.0),
            clustering_coefficient=clustering.get(node_id, 0.0),
            hub_score=hubs.get(node_id, 0.0),
            authority_score=authorities.get(node_id, 0.0),
        )

    return result
