"""Graph construction, feature extraction, and query module.

Represents relationships as graph structures. Must load and validate
the four native Elliptic++ edgelists rather than reconstructing edges
from raw records.
"""

from .builder import ChainTraceGraph
from .features import (
    NodeFeatures,
    compute_betweenness_centrality,
    compute_clustering_coefficients,
    compute_degree_features,
    compute_hits,
    compute_pagerank,
    extract_all_features,
    extract_node_features,
)
from .paths import (
    find_all_simple_paths,
    find_common_counterparties,
    find_shortest_path,
    get_connected_wallets,
    get_ego_graph,
    get_transaction_chain,
)

__all__ = [
    # Builder
    "ChainTraceGraph",
    # Feature extraction
    "NodeFeatures",
    "compute_betweenness_centrality",
    "compute_clustering_coefficients",
    "compute_degree_features",
    "compute_hits",
    "compute_pagerank",
    "extract_all_features",
    "extract_node_features",
    # Path / investigation queries
    "find_all_simple_paths",
    "find_common_counterparties",
    "find_shortest_path",
    "get_connected_wallets",
    "get_ego_graph",
    "get_transaction_chain",
]
