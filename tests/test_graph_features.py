"""Tests for graph feature extraction."""

from backend.domain.graph import AddrTxEdge, TxAddrEdge, TxTxEdge
from backend.graph.builder import ChainTraceGraph
from backend.graph.features import (
    NodeFeatures,
    compute_betweenness_centrality,
    compute_clustering_coefficients,
    compute_degree_features,
    compute_hits,
    compute_pagerank,
    extract_all_features,
    extract_node_features,
)


def _build_feature_graph() -> ChainTraceGraph:
    """Build a small graph for feature testing.

    Topology:
        addrA --[addr_tx]--> tx1 --[tx_addr]--> addrB
        addrA --[addr_tx]--> tx2 --[tx_addr]--> addrB
        tx1   --[tx_tx]  --> tx2
    """
    cg = ChainTraceGraph()
    cg.add_addr_tx_edges([
        AddrTxEdge(input_address="addrA", txid=1),
        AddrTxEdge(input_address="addrA", txid=2),
    ])
    cg.add_tx_addr_edges([
        TxAddrEdge(txid=1, output_address="addrB"),
        TxAddrEdge(txid=2, output_address="addrB"),
    ])
    cg.add_tx_tx_edges([
        TxTxEdge(source_txid=1, target_txid=2),
    ])
    return cg


# --- compute_degree_features ---


def test_degree_features_wallet():
    """Wallet node degree reflects addr_tx and tx_addr edges."""
    cg = _build_feature_graph()
    deg = compute_degree_features(cg, "addrA")
    assert deg["out_degree"] == 2  # addrA → tx1, addrA → tx2
    assert deg["in_degree"] == 0
    assert deg["total_degree"] == 2
    assert deg["out_degree_by_relationship"]["addr_tx"] == 2


def test_degree_features_transaction():
    """Transaction node degree reflects incoming and outgoing edges."""
    cg = _build_feature_graph()
    deg = compute_degree_features(cg, 1)
    # tx1: in from addrA (addr_tx), out to addrB (tx_addr) + tx2 (tx_tx)
    assert deg["in_degree"] == 1
    assert deg["out_degree"] == 2
    assert deg["total_degree"] == 3
    assert deg["in_degree_by_relationship"]["addr_tx"] == 1
    assert deg["out_degree_by_relationship"]["tx_addr"] == 1
    assert deg["out_degree_by_relationship"]["tx_tx"] == 1


def test_degree_features_missing_node():
    """Returns empty dict for nonexistent node."""
    cg = _build_feature_graph()
    assert compute_degree_features(cg, "nonexistent") == {}


def test_degree_features_empty_graph():
    """Returns empty dict for empty graph."""
    cg = ChainTraceGraph()
    assert compute_degree_features(cg, "anything") == {}


# --- compute_pagerank ---


def test_pagerank_non_empty():
    """PageRank returns scores for all nodes summing to ~1.0."""
    cg = _build_feature_graph()
    pr = compute_pagerank(cg)
    assert len(pr) == 4  # addrA, addrB, tx1, tx2
    assert abs(sum(pr.values()) - 1.0) < 1e-6


def test_pagerank_empty_graph():
    """Returns empty dict for empty graph."""
    cg = ChainTraceGraph()
    assert compute_pagerank(cg) == {}


# --- compute_betweenness_centrality ---


def test_betweenness_centrality():
    """Betweenness centrality returns values for all nodes."""
    cg = _build_feature_graph()
    bc = compute_betweenness_centrality(cg)
    assert len(bc) == 4
    assert all(v >= 0.0 for v in bc.values())


def test_betweenness_centrality_empty():
    """Returns empty dict for empty graph."""
    cg = ChainTraceGraph()
    assert compute_betweenness_centrality(cg) == {}


# --- compute_clustering_coefficients ---


def test_clustering_coefficients():
    """Clustering coefficients returned for all nodes."""
    cg = _build_feature_graph()
    cc = compute_clustering_coefficients(cg)
    assert len(cc) == 4
    assert all(0.0 <= v <= 1.0 for v in cc.values())


def test_clustering_empty():
    """Returns empty dict for empty graph."""
    cg = ChainTraceGraph()
    assert compute_clustering_coefficients(cg) == {}


# --- compute_hits ---


def test_hits_scores():
    """HITS returns hub and authority scores for all nodes."""
    cg = _build_feature_graph()
    hubs, auths = compute_hits(cg)
    assert len(hubs) == 4
    assert len(auths) == 4
    assert all(v >= 0.0 for v in hubs.values())
    assert all(v >= 0.0 for v in auths.values())


def test_hits_empty():
    """Returns empty dicts for empty graph."""
    cg = ChainTraceGraph()
    hubs, auths = compute_hits(cg)
    assert hubs == {}
    assert auths == {}


# --- extract_node_features ---


def test_extract_node_features():
    """Extracts complete NodeFeatures for a valid node."""
    cg = _build_feature_graph()
    features = extract_node_features(cg, "addrA")
    assert features is not None
    assert isinstance(features, NodeFeatures)
    assert features.node_id == "addrA"
    assert features.node_type == "wallet"
    assert features.out_degree == 2
    assert features.in_degree == 0
    assert features.pagerank > 0.0


def test_extract_node_features_missing():
    """Returns None for nonexistent node."""
    cg = _build_feature_graph()
    assert extract_node_features(cg, "nonexistent") is None


def test_extract_node_features_transaction():
    """Works correctly for transaction nodes."""
    cg = _build_feature_graph()
    features = extract_node_features(cg, 1)
    assert features is not None
    assert features.node_type == "transaction"
    assert features.total_degree == 3


# --- extract_all_features ---


def test_extract_all_features():
    """Extracts features for every node in the graph."""
    cg = _build_feature_graph()
    all_feats = extract_all_features(cg)
    assert len(all_feats) == 4
    assert all(isinstance(f, NodeFeatures) for f in all_feats.values())

    # Verify specific node
    assert all_feats["addrA"].node_type == "wallet"
    assert all_feats[1].node_type == "transaction"


def test_extract_all_features_empty():
    """Returns empty dict for empty graph."""
    cg = ChainTraceGraph()
    assert extract_all_features(cg) == {}


def test_extract_all_features_consistency():
    """Bulk extraction matches single-node extraction."""
    cg = _build_feature_graph()
    all_feats = extract_all_features(cg)
    single = extract_node_features(cg, "addrA")
    assert single is not None

    bulk = all_feats["addrA"]
    assert single.in_degree == bulk.in_degree
    assert single.out_degree == bulk.out_degree
    assert abs(single.pagerank - bulk.pagerank) < 1e-6
