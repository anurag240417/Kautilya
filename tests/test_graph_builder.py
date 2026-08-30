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
