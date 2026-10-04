"""Path finding and investigation-oriented graph queries.

Implements shortest-path, ego-graph, connected-wallet, and
transaction-chain queries over the ``KautilyaGraph``. These
support the investigation workflow questions listed in
ARCHITECTURE.md — Graph Design.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import networkx as nx

if TYPE_CHECKING:
    from backend.graph.builder import KautilyaGraph


def find_shortest_path(
    graph: KautilyaGraph,
    source: str | int,
    target: str | int,
) -> list[str | int] | None:
    """Find the shortest directed path between two nodes.

    Args:
        graph: The Kautilya investigation graph.
        source: Source node ID.
        target: Target node ID.

    Returns:
        Ordered list of node IDs forming the path, or ``None`` if
        no path exists or a node is missing.
    """
    try:
        return nx.shortest_path(graph.G, source, target)
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None


def find_all_simple_paths(
    graph: KautilyaGraph,
    source: str | int,
    target: str | int,
    max_depth: int = 6,
) -> list[list[str | int]]:
    """Find all simple (non-repeating) paths up to a maximum depth.

    WARNING: Can be expensive on large graphs. Always supply a
    reasonable ``max_depth``.

    Args:
        graph: The Kautilya investigation graph.
        source: Source node ID.
        target: Target node ID.
        max_depth: Maximum path length (default 6).

    Returns:
        List of paths, each being a list of node IDs.
    """
    try:
        return list(nx.all_simple_paths(graph.G, source, target, cutoff=max_depth))
    except nx.NodeNotFound:
        return []


def get_ego_graph(
    graph: KautilyaGraph,
    node_id: str | int,
    radius: int = 1,
) -> nx.MultiDiGraph | None:
    """Extract the ego graph (local neighbourhood) around a node.

    Returns the subgraph of all nodes within ``radius`` hops
    (ignoring edge direction for neighbour discovery).

    Args:
        graph: The Kautilya investigation graph.
        node_id: Center node.
        radius: Number of hops from center (default 1).

    Returns:
        A new ``MultiDiGraph`` subgraph, or ``None`` if the node
        does not exist.
    """
    if node_id not in graph.G:
        return None
    return nx.ego_graph(graph.G, node_id, radius=radius, undirected=True)


def get_connected_wallets(
    graph: KautilyaGraph,
    wallet_address: str,
    max_depth: int = 2,
) -> set[str]:
    """Find all wallets reachable from a given wallet within ``max_depth`` hops.

    Traverses through transactions to discover related wallets.

    Args:
        graph: The Kautilya investigation graph.
        wallet_address: Starting wallet address.
        max_depth: Maximum traversal depth (default 2).

    Returns:
        Set of connected wallet addresses (excluding the input wallet).
    """
    if wallet_address not in graph.G:
        return set()

    ego = nx.ego_graph(graph.G, wallet_address, radius=max_depth, undirected=True)
    return {
        node for node in ego.nodes()
        if ego.nodes[node].get("type") == "wallet" and node != wallet_address
    }


def get_transaction_chain(
    graph: KautilyaGraph,
    txid: int,
    depth: int = 3,
    direction: str = "both",
) -> nx.MultiDiGraph | None:
    """Extract a chain of transactions following tx→tx money-flow edges.

    Args:
        graph: The Kautilya investigation graph.
        txid: Starting transaction ID.
        depth: Maximum chain depth (default 3).
        direction: ``"forward"`` (downstream), ``"backward"`` (upstream),
            or ``"both"`` (default).

    Returns:
        Subgraph containing the transaction chain, or ``None`` if
        ``txid`` is not in the graph.
    """
    if txid not in graph.G:
        return None

    chain_nodes: set[int] = {txid}

    if direction in ("forward", "both"):
        _traverse_tx_chain(graph.G, txid, depth, "forward", chain_nodes)
    if direction in ("backward", "both"):
        _traverse_tx_chain(graph.G, txid, depth, "backward", chain_nodes)

    return graph.G.subgraph(chain_nodes).copy()


def _traverse_tx_chain(
    G: nx.MultiDiGraph,
    start: int,
    depth: int,
    direction: str,
    visited: set[int],
) -> None:
    """Recursively follow tx→tx edges in one direction.

    Only follows edges with ``relationship == "tx_tx"`` so that
    address-related edges are skipped.
    """
    if depth <= 0:
        return

    if direction == "forward":
        edge_iter = G.out_edges(start, data=True)
    else:
        edge_iter = G.in_edges(start, data=True)

    for u, v, data in edge_iter:
        if data.get("relationship") != "tx_tx":
            continue
        neighbor = v if direction == "forward" else u
        if neighbor not in visited:
            visited.add(neighbor)
            _traverse_tx_chain(G, neighbor, depth - 1, direction, visited)


def find_common_counterparties(
    graph: KautilyaGraph,
    addr1: str,
    addr2: str,
) -> set[str | int]:
    """Find entities that both addresses have transacted with.

    Useful for identifying shared intermediaries or mixing services.

    Args:
        graph: The Kautilya investigation graph.
        addr1: First wallet address.
        addr2: Second wallet address.

    Returns:
        Set of node IDs that are neighbours of both addresses.
    """
    if addr1 not in graph.G or addr2 not in graph.G:
        return set()

    undirected = graph.G.to_undirected()
    neighbors1 = set(undirected.neighbors(addr1))
    neighbors2 = set(undirected.neighbors(addr2))

    return neighbors1 & neighbors2
