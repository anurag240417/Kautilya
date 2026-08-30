"""Graph builder.

Constructs the in-memory investigation graph representation using
NetworkX. Integrates normalized edgelists into a MultiDiGraph.
"""

from collections.abc import Iterable

import networkx as nx

from backend.domain.graph import (
    AddrAddrEdge,
    AddrTxEdge,
    TxAddrEdge,
    TxTxEdge,
)


class ChainTraceGraph:
    """Core in-memory graph structure for ChainTrace.

    Wraps a networkx.MultiDiGraph to allow for directed,
    possibly parallel edges (e.g. self-change addresses).
    Nodes are explicitly tagged with 'type'="wallet" or 'type'="transaction".
    """

    def __init__(self) -> None:
        """Initialize the empty MultiDiGraph."""
        self.G = nx.MultiDiGraph()

    def add_tx_tx_edges(self, edges: Iterable[TxTxEdge]) -> None:
        """Add transaction-to-transaction edges.

        Ensures source and target nodes exist as 'transaction' type.
        """
        for edge in edges:
            self.G.add_node(edge.source_txid, type="transaction")
            self.G.add_node(edge.target_txid, type="transaction")
            self.G.add_edge(
                edge.source_txid,
                edge.target_txid,
                is_synthetic=edge.is_synthetic,
                relationship="tx_tx",
            )

    def add_addr_tx_edges(self, edges: Iterable[AddrTxEdge]) -> None:
        """Add address-to-transaction edges.

        Ensures input node is 'wallet' and target is 'transaction'.
        """
        for edge in edges:
            self.G.add_node(edge.input_address, type="wallet")
            self.G.add_node(edge.txid, type="transaction")
            self.G.add_edge(
                edge.input_address,
                edge.txid,
                is_synthetic=edge.is_synthetic,
                relationship="addr_tx",
            )

    def add_tx_addr_edges(self, edges: Iterable[TxAddrEdge]) -> None:
        """Add transaction-to-address edges.

        Ensures source node is 'transaction' and target is 'wallet'.
        """
        for edge in edges:
            self.G.add_node(edge.txid, type="transaction")
            self.G.add_node(edge.output_address, type="wallet")
            self.G.add_edge(
                edge.txid,
                edge.output_address,
                is_synthetic=edge.is_synthetic,
                relationship="tx_addr",
            )

    def add_addr_addr_edges(self, edges: Iterable[AddrAddrEdge]) -> None:
        """Add address-to-address edges.

        Ensures source and target nodes exist as 'wallet' type.
        """
        for edge in edges:
            self.G.add_node(edge.input_address, type="wallet")
            self.G.add_node(edge.output_address, type="wallet")
            self.G.add_edge(
                edge.input_address,
                edge.output_address,
                is_synthetic=edge.is_synthetic,
                relationship="addr_addr",
            )
