"""Graph-derived feature extraction for ML experiments.

Bridges ``graph/features.py`` with the ML pipeline by extracting
graph features into a DataFrame that can be joined with the
transaction feature matrix for M2 model training.

See CONTEXT.md — Experimental Progression (M2: blockchain + graph).
"""

import logging

import pandas as pd

from backend.graph.builder import KautilyaGraph
from backend.graph.features import (
    compute_betweenness_centrality,
    compute_clustering_coefficients,
    compute_degree_features,
    compute_hits,
    compute_pagerank,
)
from backend.ml.feature_registry import GRAPH_FEATURES

logger = logging.getLogger(__name__)


def extract_graph_feature_df(
    graph: KautilyaGraph,
    transaction_ids: list | None = None,
    betweenness_k: int | None = None,
) -> pd.DataFrame:
    """Extract graph features for transactions into a DataFrame.

    Computes all 8 graph-derived features (in_degree, out_degree,
    total_degree, pagerank, betweenness_centrality,
    clustering_coefficient, hub_score, authority_score) for each
    transaction node in the graph.

    Args:
        graph: The Kautilya investigation graph.
        transaction_ids: Optional list of transaction IDs to extract
            features for. If None, extracts for all transaction nodes.
        betweenness_k: Number of source nodes to sample for approximate
            betweenness centrality. If None, computes exact centrality.

    Returns:
        DataFrame with ``txId`` column plus the 8 graph feature columns.
        Transactions not found in the graph are excluded.
    """
    if len(graph.G) == 0:
        logger.warning("Graph is empty; returning empty feature DataFrame")
        return pd.DataFrame(columns=["txId"] + GRAPH_FEATURES)

    # Compute graph-wide metrics once
    pagerank = compute_pagerank(graph)
    betweenness = compute_betweenness_centrality(graph, k=betweenness_k)
    clustering = compute_clustering_coefficients(graph)
    hubs, authorities = compute_hits(graph)

    # Determine target nodes
    if transaction_ids is not None:
        target_ids = [tid for tid in transaction_ids if tid in graph.G]
    else:
        target_ids = graph.get_nodes_by_type("transaction")

    rows: list[dict] = []
    for node_id in target_ids:
        degree_info = compute_degree_features(graph, node_id)
        if not degree_info:
            continue

        rows.append({
            "txId": node_id,
            "in_degree": degree_info["in_degree"],
            "out_degree": degree_info["out_degree"],
            "total_degree": degree_info["total_degree"],
            "pagerank": pagerank.get(node_id, 0.0),
            "betweenness_centrality": betweenness.get(node_id, 0.0),
            "clustering_coefficient": clustering.get(node_id, 0.0),
            "hub_score": hubs.get(node_id, 0.0),
            "authority_score": authorities.get(node_id, 0.0),
        })

    df = pd.DataFrame(rows, columns=["txId"] + GRAPH_FEATURES)

    logger.info(
        "Extracted graph features for %d/%d transaction nodes",
        len(df),
        len(target_ids),
    )

    return df


def merge_graph_features(
    tx_df: pd.DataFrame,
    graph_feature_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge graph features onto the transaction feature DataFrame.

    Performs a left join so that all transactions are preserved.
    Missing graph features (transactions not in the graph) are filled
    with 0 — giving them neutral/baseline graph signal.

    Args:
        tx_df: Transaction DataFrame with a ``txId`` column.
        graph_feature_df: DataFrame from ``extract_graph_feature_df()``.

    Returns:
        The transaction DataFrame augmented with graph feature columns.

    Raises:
        ValueError: If ``txId`` column is missing from either DataFrame.
    """
    if "txId" not in tx_df.columns:
        msg = "tx_df is missing 'txId' column"
        raise ValueError(msg)
    if "txId" not in graph_feature_df.columns:
        msg = "graph_feature_df is missing 'txId' column"
        raise ValueError(msg)

    merged = tx_df.merge(graph_feature_df, on="txId", how="left")

    # Fill NaN for transactions not in the graph with 0 (neutral features)
    for col in GRAPH_FEATURES:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0.0)

    n_missing = len(tx_df) - len(graph_feature_df.merge(
        tx_df[["txId"]], on="txId", how="inner"
    ))
    if n_missing > 0:
        logger.info(
            "%d transactions had no graph features (filled with 0)",
            n_missing,
        )

    return merged
