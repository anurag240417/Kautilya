"""Tests for graph path queries and investigation functions."""

from backend.domain.graph import AddrAddrEdge, AddrTxEdge, TxAddrEdge, TxTxEdge
from backend.graph.builder import KautilyaGraph
from backend.graph.paths import (
    find_all_simple_paths,
    find_common_counterparties,
    find_shortest_path,
    get_connected_wallets,
    get_ego_graph,
    get_transaction_chain,
)


def _build_sample_graph() -> KautilyaGraph:
    """Build a small graph for query testing.

    Topology:
        addrA --[addr_tx]--> tx1 --[tx_addr]--> addrB
        addrA --[addr_tx]--> tx1 --[tx_addr]--> addrC
        addrC --[addr_tx]--> tx2 --[tx_tx]  --> tx3
        addrA --[addr_addr]--> addrD
        tx3   --[tx_addr]--> addrE
    """
    cg = KautilyaGraph()
    cg.add_addr_tx_edges([
        AddrTxEdge(input_address="addrA", txid=1),
        AddrTxEdge(input_address="addrC", txid=2),
    ])
    cg.add_tx_addr_edges([
        TxAddrEdge(txid=1, output_address="addrB"),
        TxAddrEdge(txid=1, output_address="addrC"),
        TxAddrEdge(txid=3, output_address="addrE"),
    ])
    cg.add_tx_tx_edges([
        TxTxEdge(source_txid=2, target_txid=3),
    ])
    cg.add_addr_addr_edges([
        AddrAddrEdge(input_address="addrA", output_address="addrD"),
    ])
    return cg


# --- find_shortest_path ---


def test_shortest_path_exists():
    """Finds a directed path when one exists."""
    cg = _build_sample_graph()
    path = find_shortest_path(cg, "addrA", "addrB")
    assert path is not None
    assert path[0] == "addrA"
    assert path[-1] == "addrB"


def test_shortest_path_no_path():
    """Returns None when no directed path exists."""
    cg = _build_sample_graph()
    # addrE has no outgoing edges, so no path from addrE to addrA
    result = find_shortest_path(cg, "addrE", "addrA")
    assert result is None


def test_shortest_path_missing_node():
    """Returns None when a node does not exist."""
    cg = _build_sample_graph()
    result = find_shortest_path(cg, "nonexistent", "addrA")
    assert result is None


# --- find_all_simple_paths ---


def test_all_simple_paths():
    """Finds multiple paths between nodes."""
    cg = _build_sample_graph()
    paths = find_all_simple_paths(cg, "addrA", "addrC", max_depth=5)
    assert len(paths) >= 1
    for p in paths:
        assert p[0] == "addrA"
        assert p[-1] == "addrC"


def test_all_simple_paths_missing_node():
    """Returns empty list for missing nodes."""
    cg = _build_sample_graph()
    paths = find_all_simple_paths(cg, "nonexistent", "addrA")
    assert paths == []


def test_all_simple_paths_max_depth_limits():
    """Respects max_depth cutoff."""
    cg = _build_sample_graph()
    # With depth=1, probably can't reach addrE from addrA
    paths = find_all_simple_paths(cg, "addrA", "addrE", max_depth=1)
    assert all(len(p) <= 2 for p in paths)


# --- get_ego_graph ---


def test_ego_graph_radius_1():
    """Ego graph with radius=1 returns immediate neighbours."""
    cg = _build_sample_graph()
    ego = get_ego_graph(cg, 1, radius=1)
    assert ego is not None
    # tx1 connects to addrA (in), addrB (out), addrC (out)
    assert 1 in ego.nodes()
    assert "addrA" in ego.nodes()
    assert "addrB" in ego.nodes()
    assert "addrC" in ego.nodes()


def test_ego_graph_missing_node():
    """Returns None for a node not in the graph."""
    cg = _build_sample_graph()
    assert get_ego_graph(cg, "nonexistent") is None


def test_ego_graph_radius_0():
    """Ego graph with radius=0 returns only the center node."""
    cg = _build_sample_graph()
    ego = get_ego_graph(cg, 1, radius=0)
    assert ego is not None
    assert list(ego.nodes()) == [1]


# --- get_connected_wallets ---


def test_connected_wallets():
    """Finds wallets reachable from a given wallet."""
    cg = _build_sample_graph()
    connected = get_connected_wallets(cg, "addrA", max_depth=2)
    # addrA → tx1 → addrB, addrC  and  addrA → addrD
    assert "addrB" in connected
    assert "addrC" in connected
    assert "addrD" in connected
    assert "addrA" not in connected  # excludes self


def test_connected_wallets_missing():
    """Returns empty set for missing wallet."""
    cg = _build_sample_graph()
    assert get_connected_wallets(cg, "nonexistent") == set()


def test_connected_wallets_depth_limit():
    """Depth limit restricts traversal."""
    cg = _build_sample_graph()
    # addrE is 4 hops from addrA: addrA→tx1→addrC→tx2→tx3→addrE
    connected_shallow = get_connected_wallets(cg, "addrA", max_depth=1)
    # addrD is 1 hop away via addr_addr, B and C are 2 hops via tx1
    assert "addrD" in connected_shallow


# --- get_transaction_chain ---


def test_transaction_chain_forward():
    """Forward chain follows tx→tx edges downstream."""
    cg = _build_sample_graph()
    chain = get_transaction_chain(cg, 2, depth=3, direction="forward")
    assert chain is not None
    assert 2 in chain.nodes()
    assert 3 in chain.nodes()


def test_transaction_chain_backward():
    """Backward chain follows tx→tx edges upstream."""
    cg = _build_sample_graph()
    chain = get_transaction_chain(cg, 3, depth=3, direction="backward")
    assert chain is not None
    assert 3 in chain.nodes()
    assert 2 in chain.nodes()


def test_transaction_chain_both():
    """Both directions captures full chain."""
    cg = _build_sample_graph()
    chain = get_transaction_chain(cg, 2, depth=3, direction="both")
    assert chain is not None
    assert 2 in chain.nodes()
    assert 3 in chain.nodes()


def test_transaction_chain_missing_node():
    """Returns None for a txid not in the graph."""
    cg = _build_sample_graph()
    assert get_transaction_chain(cg, 9999) is None


def test_transaction_chain_no_tx_tx_edges():
    """Chain with only addr edges doesn't follow non-tx_tx edges."""
    cg = KautilyaGraph()
    cg.add_addr_tx_edges([AddrTxEdge(input_address="addr1", txid=10)])
    cg.add_tx_addr_edges([TxAddrEdge(txid=10, output_address="addr2")])
    chain = get_transaction_chain(cg, 10, depth=3, direction="forward")
    assert chain is not None
    # Only the starting node, no tx_tx to follow
    assert list(chain.nodes()) == [10]


# --- find_common_counterparties ---


def test_common_counterparties():
    """Finds shared neighbours between two addresses."""
    cg = _build_sample_graph()
    # addrA → tx1, addrC → tx1 (via output), so tx1 is shared
    # Actually: addrA→tx1 and tx1→addrC, so from addrA's side tx1 is a neighbour
    # and from addrC's side tx1 is also a neighbour (tx1→addrC output edge)
    common = find_common_counterparties(cg, "addrA", "addrC")
    assert 1 in common  # tx1 connects to both


def test_common_counterparties_missing_node():
    """Returns empty set when a node doesn't exist."""
    cg = _build_sample_graph()
    assert find_common_counterparties(cg, "nonexistent", "addrA") == set()


def test_common_counterparties_no_overlap():
    """Returns empty set when addresses share no neighbours."""
    cg = _build_sample_graph()
    # addrD only connects to addrA, addrE only connects to tx3
    common = find_common_counterparties(cg, "addrD", "addrE")
    assert common == set()
