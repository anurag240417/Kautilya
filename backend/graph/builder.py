"""Graph builder.

Constructs the in-memory investigation graph representation using
NetworkX. Integrates normalized edgelists into a MultiDiGraph.

Edges carry full provenance metadata: ``is_synthetic``, ``confidence``,
``provenance``, ``temporal_context``, and ``relationship`` type.
"""

from collections.abc import Iterable

import networkx as nx

from backend.domain.graph import (
    AddrAddrEdge,
    AddrTxEdge,
    TxAddrEdge,
    TxTxEdge,
)


def _edge_attrs(edge: TxTxEdge | AddrTxEdge | TxAddrEdge | AddrAddrEdge,
                relationship: str) -> dict:
    """Build the attribute dict stored on each NetworkX edge.

    Centralizes provenance propagation so every ``add_*_edges``
    method stores the same set of metadata keys.
    """
    return {
        "relationship": relationship,
        "is_synthetic": edge.is_synthetic,
        "confidence": edge.confidence,
        "provenance": edge.provenance,
        "temporal_context": edge.temporal_context,
    }


class KautilyaGraph:
    """Core in-memory graph structure for Kautilya.

    Wraps a networkx.MultiDiGraph to allow for directed,
    possibly parallel edges (e.g. self-change addresses).
    Nodes are explicitly tagged with 'type'="wallet" or 'type'="transaction".

    Edge attributes always include:
        - ``relationship``: edge type (tx_tx, addr_tx, tx_addr, addr_addr)
        - ``is_synthetic``: whether the edge is synthetic
        - ``confidence``: edge confidence (None if not set)
        - ``provenance``: data source identifier (None if not set)
        - ``temporal_context``: associated time_step (None if not set)
    """

    def __init__(self) -> None:
        """Initialize the empty MultiDiGraph."""
        self.G = nx.MultiDiGraph()

    # ------------------------------------------------------------------
    # Edge insertion
    # ------------------------------------------------------------------

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
                **_edge_attrs(edge, "tx_tx"),
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
                **_edge_attrs(edge, "addr_tx"),
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
                **_edge_attrs(edge, "tx_addr"),
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
                **_edge_attrs(edge, "addr_addr"),
            )

    # ------------------------------------------------------------------
    # Node queries
    # ------------------------------------------------------------------

    @property
    def node_count(self) -> int:
        """Total number of nodes in the graph."""
        return self.G.number_of_nodes()

    @property
    def edge_count(self) -> int:
        """Total number of edges in the graph."""
        return self.G.number_of_edges()

    def has_node(self, node_id: str | int) -> bool:
        """Check whether a node exists in the graph."""
        return node_id in self.G

    def get_node_type(self, node_id: str | int) -> str | None:
        """Return the type ('wallet' or 'transaction') of a node.

        Returns None if the node does not exist.
        """
        if node_id not in self.G:
            return None
        return self.G.nodes[node_id].get("type")

    def get_nodes_by_type(self, type_name: str) -> list[str | int]:
        """Return all node IDs with the given type.

        Args:
            type_name: 'wallet' or 'transaction'.
        """
        return [
            node for node, attrs in self.G.nodes(data=True)
            if attrs.get("type") == type_name
        ]

    # ------------------------------------------------------------------
    # Edge queries
    # ------------------------------------------------------------------

    def get_node_edges(
        self,
        node_id: str | int,
        relationship: str | None = None,
    ) -> list[dict]:
        """Return all edges incident to a node with their attributes.

        Args:
            node_id: The node to query.
            relationship: If provided, filter to edges of this type
                (e.g. 'tx_tx', 'addr_tx', 'tx_addr', 'addr_addr').

        Returns:
            List of dicts with keys: source, target, and all edge attrs.
        """
        if node_id not in self.G:
            return []

        results: list[dict] = []

        # Outgoing edges
        for _, target, data in self.G.out_edges(node_id, data=True):
            if relationship and data.get("relationship") != relationship:
                continue
            results.append({"source": node_id, "target": target, **data})

        # Incoming edges
        for source, _, data in self.G.in_edges(node_id, data=True):
            if relationship and data.get("relationship") != relationship:
                continue
            results.append({"source": source, "target": node_id, **data})

        return results

    def get_neighbors(
        self,
        node_id: str | int,
        relationship: str | None = None,
        direction: str = "both",
    ) -> list[str | int]:
        """Return neighbor node IDs.

        Args:
            node_id: The node to query.
            relationship: If provided, only follow edges of this type.
            direction: 'out' (successors), 'in' (predecessors), or 'both'.

        Returns:
            Deduplicated list of neighbor node IDs.
        """
        if node_id not in self.G:
            return []

        neighbors: set[str | int] = set()

        if direction in ("out", "both"):
            for _, target, data in self.G.out_edges(node_id, data=True):
                if relationship and data.get("relationship") != relationship:
                    continue
                neighbors.add(target)

        if direction in ("in", "both"):
            for source, _, data in self.G.in_edges(node_id, data=True):
                if relationship and data.get("relationship") != relationship:
                    continue
                neighbors.add(source)

        return list(neighbors)

    def get_edges_by_relationship(self, relationship: str) -> list[dict]:
        """Return all edges of a given relationship type.

        Args:
            relationship: Edge type (e.g. 'tx_tx', 'addr_tx').

        Returns:
            List of dicts with source, target, and edge attributes.
        """
        results: list[dict] = []
        for source, target, data in self.G.edges(data=True):
            if data.get("relationship") == relationship:
                results.append({"source": source, "target": target, **data})
        return results

    def get_synthetic_edges(self) -> list[dict]:
        """Return all edges marked as synthetic."""
        results: list[dict] = []
        for source, target, data in self.G.edges(data=True):
            if data.get("is_synthetic"):
                results.append({"source": source, "target": target, **data})
        return results
