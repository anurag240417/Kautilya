"""Tests for graph builder logic."""

from backend.domain.graph import AddrAddrEdge, AddrTxEdge, TxAddrEdge, TxTxEdge
from backend.graph.builder import ChainTraceGraph


def test_chaintrace_graph_initialization():
    """Graph initializes as an empty MultiDiGraph."""
    cg = ChainTraceGraph()
    assert len(cg.G.nodes) == 0
    assert len(cg.G.edges) == 0


def test_add_tx_tx_edges():
    """Transaction-to-transaction edges properly assign nodes as type 'transaction'."""
    cg = ChainTraceGraph()
    edges = [
        TxTxEdge(source_txid=1, target_txid=2, is_synthetic=False),
        TxTxEdge(source_txid=2, target_txid=3, is_synthetic=True),
    ]
    cg.add_tx_tx_edges(edges)

    assert len(cg.G.nodes) == 3
    assert cg.G.nodes[1]["type"] == "transaction"
    assert cg.G.nodes[2]["type"] == "transaction"
    assert cg.G.nodes[3]["type"] == "transaction"

    assert len(cg.G.edges) == 2

    # Check edge attributes
    edge_data = cg.G.get_edge_data(1, 2)
    assert edge_data[0]["is_synthetic"] is False
    assert edge_data[0]["relationship"] == "tx_tx"

    edge_data = cg.G.get_edge_data(2, 3)
    assert edge_data[0]["is_synthetic"] is True


def test_add_addr_tx_edges():
    """Address-to-transaction edges properly distinguish 'wallet' and 'transaction' types."""
    cg = ChainTraceGraph()
    edges = [AddrTxEdge(input_address="addr1", txid=99, is_synthetic=False)]
    cg.add_addr_tx_edges(edges)

    assert len(cg.G.nodes) == 2
    assert cg.G.nodes["addr1"]["type"] == "wallet"
    assert cg.G.nodes[99]["type"] == "transaction"

    edge_data = cg.G.get_edge_data("addr1", 99)
    assert edge_data[0]["relationship"] == "addr_tx"


def test_add_tx_addr_edges():
    """Transaction-to-address edges properly assign types."""
    cg = ChainTraceGraph()
    edges = [TxAddrEdge(txid=100, output_address="addr2", is_synthetic=False)]
    cg.add_tx_addr_edges(edges)

    assert cg.G.nodes[100]["type"] == "transaction"
    assert cg.G.nodes["addr2"]["type"] == "wallet"

    edge_data = cg.G.get_edge_data(100, "addr2")
    assert edge_data[0]["relationship"] == "tx_addr"


def test_add_addr_addr_edges():
    """Address-to-address edges assign both ends as 'wallet'."""
    cg = ChainTraceGraph()
    edges = [AddrAddrEdge(input_address="addr1", output_address="addr2", is_synthetic=False)]
    cg.add_addr_addr_edges(edges)

    assert cg.G.nodes["addr1"]["type"] == "wallet"
    assert cg.G.nodes["addr2"]["type"] == "wallet"

    edge_data = cg.G.get_edge_data("addr1", "addr2")
    assert edge_data[0]["relationship"] == "addr_addr"


def test_parallel_edges_are_preserved():
    """MultiDiGraph should allow multiple edges between the same two nodes."""
    cg = ChainTraceGraph()
    # addr1 is an input and output of tx 99 (self-change scenario)
    addr_tx = [AddrTxEdge(input_address="addr1", txid=99, is_synthetic=False)]
    tx_addr = [TxAddrEdge(txid=99, output_address="addr1", is_synthetic=False)]

    cg.add_addr_tx_edges(addr_tx)
    cg.add_tx_addr_edges(tx_addr)

    # 2 distinct edges (one in each direction)
    assert len(cg.G.edges) == 2

    # What if two identical directional edges exist?
    cg.add_addr_tx_edges(addr_tx)
    assert len(cg.G.edges) == 3


def test_edge_provenance_attributes_preserved():
    """Edge confidence, provenance, and temporal_context are stored on edges."""
    cg = ChainTraceGraph()
    edges = [
        TxTxEdge(
            source_txid=10,
            target_txid=20,
            is_synthetic=True,
            confidence=0.85,
            provenance="synthetic_generator",
            temporal_context=5,
        )
    ]
    cg.add_tx_tx_edges(edges)

    edge_data = cg.G.get_edge_data(10, 20)[0]
    assert edge_data["is_synthetic"] is True
    assert edge_data["confidence"] == 0.85
    assert edge_data["provenance"] == "synthetic_generator"
    assert edge_data["temporal_context"] == 5
    assert edge_data["relationship"] == "tx_tx"


def test_graph_query_helpers():
    """Introspection and query methods on ChainTraceGraph work correctly."""
    cg = ChainTraceGraph()
    cg.add_addr_tx_edges([
        AddrTxEdge(input_address="addr1", txid=100, is_synthetic=False),
        AddrTxEdge(input_address="addr2", txid=100, is_synthetic=True),
    ])
    cg.add_tx_addr_edges([
        TxAddrEdge(txid=100, output_address="addr3", is_synthetic=False),
    ])

    assert cg.node_count == 4
    assert cg.edge_count == 3
    assert cg.has_node("addr1") is True
    assert cg.has_node("nonexistent") is False
    assert cg.get_node_type("addr1") == "wallet"
    assert cg.get_node_type(100) == "transaction"
    assert cg.get_node_type("nonexistent") is None

    wallets = cg.get_nodes_by_type("wallet")
    assert set(wallets) == {"addr1", "addr2", "addr3"}

    txs = cg.get_nodes_by_type("transaction")
    assert set(txs) == {100}

    # get_node_edges
    edges_100 = cg.get_node_edges(100)
    assert len(edges_100) == 3
    addr_tx_edges = cg.get_node_edges(100, relationship="addr_tx")
    assert len(addr_tx_edges) == 2

    # get_neighbors
    neighbors = cg.get_neighbors(100)
    assert set(neighbors) == {"addr1", "addr2", "addr3"}

    out_neighbors = cg.get_neighbors(100, direction="out")
    assert set(out_neighbors) == {"addr3"}

    in_neighbors = cg.get_neighbors(100, direction="in")
    assert set(in_neighbors) == {"addr1", "addr2"}

    # get_synthetic_edges
    synth_edges = cg.get_synthetic_edges()
    assert len(synth_edges) == 1
    assert synth_edges[0]["source"] == "addr2"

    # get_edges_by_relationship
    tx_addr_edges = cg.get_edges_by_relationship("tx_addr")
    assert len(tx_addr_edges) == 1

