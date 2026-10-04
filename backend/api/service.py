"""Investigation service layer.

Decouples the API presentation layer from underlying storage, graph queries,
and ML risk engines per ARCHITECTURE.md §api/ and CODING_CONVENTIONS.md.
"""

from typing import Any

from backend.api.schemas import (
    GraphEdge,
    GraphNode,
    GraphPathResponse,
    GraphResponse,
    StatisticsResponse,
    TransactionResponse,
    WalletResponse,
)
from backend.domain.alert import AlertFilter, InvestigativeAlert
from backend.domain.graph import AddrAddrEdge, AddrTxEdge, TxAddrEdge, TxTxEdge
from backend.domain.risk import EvidenceLedger, RiskScore
from backend.domain.transaction import Transaction
from backend.domain.types import AlertStatus, EntityClass, PriorityTier
from backend.domain.wallet import StatsSummary, Wallet
from backend.graph.builder import KautilyaGraph
from backend.graph.paths import find_shortest_path, get_ego_graph
from backend.api.demo_data import build_demo_dataset
from backend.risk.aggregation import AggregationMethod, aggregate_transaction_scores
from backend.risk.ranking import filter_and_prioritize_alerts, generate_alert


class InvestigationService:
    """Core domain service for investigation lookups and triage."""

    def __init__(
        self,
        graph: KautilyaGraph | None = None,
        transactions: dict[int, Transaction] | None = None,
        wallets: dict[str, Wallet] | None = None,
        risk_scores: dict[str, RiskScore] | None = None,
        alerts: dict[str, InvestigativeAlert] | None = None,
        correlations: dict[int, list[dict[str, Any]]] | None = None,
    ) -> None:
        self.graph = graph or KautilyaGraph()
        self.transactions: dict[int, Transaction] = transactions or {}
        self.wallets: dict[str, Wallet] = wallets or {}
        self.risk_scores: dict[str, RiskScore] = risk_scores or {}
        self.alerts: dict[str, InvestigativeAlert] = alerts or {}
        self.correlations: dict[int, list[dict[str, Any]]] = correlations or {}
        self.simulation_step: int = 6

    def get_transaction(self, txid: int) -> TransactionResponse | None:
        """Fetch transaction details, interpretable features, and forensic risk signals."""
        tx = self.transactions.get(txid)
        if not tx:
            return None

        interpretable = {
            "total_btc": tx.total_btc,
            "fees": tx.fees,
            "size": tx.size,
            "num_input_addresses": tx.num_input_addresses,
            "num_output_addresses": tx.num_output_addresses,
            "in_txs_degree": tx.in_txs_degree,
            "out_txs_degree": tx.out_txs_degree,
            "in_btc_min": tx.in_btc_min,
            "in_btc_max": tx.in_btc_max,
            "in_btc_mean": tx.in_btc_mean,
            "in_btc_median": tx.in_btc_median,
            "in_btc_total": tx.in_btc_total,
            "out_btc_min": tx.out_btc_min,
            "out_btc_max": tx.out_btc_max,
            "out_btc_mean": tx.out_btc_mean,
            "out_btc_median": tx.out_btc_median,
            "out_btc_total": tx.out_btc_total,
        }

        # Connected wallets from graph
        inputs: list[str] = []
        outputs: list[str] = []
        if self.graph.has_node(txid):
            for edge in self.graph.get_node_edges(txid):
                rel = edge.get("relationship")
                if rel == "addr_tx" and edge.get("target") == txid:
                    inputs.append(str(edge.get("source")))
                elif rel == "tx_addr" and edge.get("source") == txid:
                    outputs.append(str(edge.get("target")))

        risk = self.risk_scores.get(str(txid))
        corr = self.correlations.get(txid, [])
        is_synthetic = (risk.contains_synthetic_input if risk else False) or any(
            c.get("is_synthetic", False) for c in corr
        )

        return TransactionResponse(
            txid=tx.txid,
            time_step=tx.time_step,
            label=tx.label,
            interpretable_features=interpretable,
            risk_score=risk,
            correlations=corr,
            connected_wallets={"inputs": sorted(set(inputs)), "outputs": sorted(set(outputs))},
            is_synthetic=is_synthetic,
        )

    def get_wallet(
        self,
        address: str,
        aggregation_method: AggregationMethod = AggregationMethod.MAX,
    ) -> WalletResponse | None:
        """Fetch wallet summary and aggregated risk score preserving non-accusation rules."""
        wallet = self.wallets.get(address)
        if not wallet:
            return None

        stats: dict[str, Any] = {
            "num_txs_as_sender": wallet.num_txs_as_sender,
            "num_txs_as_receiver": wallet.num_txs_as_receiver,
            "first_block_appeared_in": wallet.first_block_appeared_in,
            "last_block_appeared_in": wallet.last_block_appeared_in,
            "lifetime_in_blocks": wallet.lifetime_in_blocks,
            "total_txs": wallet.total_txs,
            "num_timesteps_appeared_in": wallet.num_timesteps_appeared_in,
            "num_addr_transacted_multiple": wallet.num_addr_transacted_multiple,
        }
        if wallet.btc_transacted:
            stats["btc_transacted"] = wallet.btc_transacted.model_dump()
        if wallet.btc_sent:
            stats["btc_sent"] = wallet.btc_sent.model_dump()
        if wallet.btc_received:
            stats["btc_received"] = wallet.btc_received.model_dump()
        if wallet.fees:
            stats["fees"] = wallet.fees.model_dump()

        # Find associated transactions for aggregation
        associated_tx_ids: list[int] = []
        counterparties: list[str] = []

        if self.graph.has_node(address):
            for edge in self.graph.get_node_edges(address):
                rel = edge.get("relationship")
                src = edge.get("source")
                tgt = edge.get("target")
                if rel in ("addr_tx", "tx_addr"):
                    tx_cand = tgt if rel == "addr_tx" else src
                    if isinstance(tx_cand, int):
                        associated_tx_ids.append(tx_cand)
                elif rel == "addr_addr":
                    cp = tgt if src == address else src
                    counterparties.append(str(cp))

        # Check for risk scores on transactions
        tx_scores: list[RiskScore] = []
        for tid in set(associated_tx_ids):
            rs = self.risk_scores.get(str(tid))
            if rs:
                tx_scores.append(rs)
            elif tid in self.transactions:
                # Provide a baseline score if not evaluated
                t = self.transactions[tid]
                score_val = 80.0 if t.label == EntityClass.ILLICIT else 10.0
                tier = (
                    PriorityTier.HIGH
                    if t.label == EntityClass.ILLICIT
                    else PriorityTier.LOW
                )
                tx_scores.append(
                    RiskScore(
                        entity_id=str(tid),
                        entity_type="transaction",
                        score=score_val,
                        priority_tier=tier,
                    )
                )

        aggregation = None
        if tx_scores:
            vol_weights = {}
            if aggregation_method == AggregationMethod.VOLUME_WEIGHTED:
                for rs in tx_scores:
                    tx_obj = self.transactions.get(int(rs.entity_id))
                    vol_weights[rs.entity_id] = (
                        tx_obj.total_btc if tx_obj and tx_obj.total_btc else 1.0
                    )
            aggregation = aggregate_transaction_scores(
                entity_id=address,
                transaction_scores=tx_scores,
                method=aggregation_method,
                volume_weights=vol_weights if vol_weights else None,
            )

        is_synthetic = (
            aggregation.contains_synthetic_input if aggregation else False
        )

        return WalletResponse(
            address=wallet.address,
            time_step=wallet.time_step,
            label=wallet.label,
            stats=stats,
            aggregation=aggregation,
            counterparties=sorted(set(counterparties)),
            is_synthetic=is_synthetic,
        )

    def list_alerts(
        self,
        filter_criteria: AlertFilter | None = None,
        offset: int = 0,
    ) -> tuple[int, int, list[InvestigativeAlert]]:
        """List and triage alerts matching filter criteria with pagination."""
        all_alerts = list(self.alerts.values())
        filtered = filter_and_prioritize_alerts(all_alerts, filter_criteria)
        total = len(all_alerts)
        filtered_count = len(filtered)

        # Apply offset and limit
        crit = filter_criteria or AlertFilter()
        limit = crit.limit or 50
        paged = filtered[offset : offset + limit]

        return total, filtered_count, paged

    def get_alert(self, alert_id: str) -> InvestigativeAlert | None:
        """Fetch an individual investigative alert by ID."""
        return self.alerts.get(alert_id)

    def update_alert_status(
        self,
        alert_id: str,
        new_status: AlertStatus,
        reviewer_notes: str | None = None,
    ) -> InvestigativeAlert | None:
        """Update lifecycle status and reviewer notes for an alert."""
        alert = self.alerts.get(alert_id)
        if not alert:
            return None

        updated_dict = alert.model_dump()
        updated_dict["status"] = new_status
        if reviewer_notes:
            meta = updated_dict.get("metadata", {})
            meta["reviewer_notes"] = reviewer_notes
            updated_dict["metadata"] = meta

        updated_alert = InvestigativeAlert(**updated_dict)
        self.alerts[alert_id] = updated_alert
        return updated_alert

    def get_statistics(self) -> StatisticsResponse:
        """Return aggregate inventory, alert counts, and triage metrics."""
        total_txs = len(self.transactions)
        total_wallets = len(self.wallets)
        total_alerts = len(self.alerts)

        tier_counts = {"critical": 0, "high": 0, "medium": 0, "low": 0}
        for a in self.alerts.values():
            tier_key = a.priority_tier.value.lower()
            if tier_key in tier_counts:
                tier_counts[tier_key] += 1

        status_counts = {
            "new": 0,
            "triaged": 0,
            "in_review": 0,
            "escalated": 0,
            "dismissed": 0,
            "closed": 0,
        }
        for a in self.alerts.values():
            st_key = a.status.value.lower()
            if st_key in status_counts:
                status_counts[st_key] += 1
                if st_key == "dismissed":
                    status_counts["closed"] += 1

        synthetic_count = sum(
            1 for a in self.alerts.values() if a.contains_synthetic_input
        )
        synthetic_pct = (
            (synthetic_count / total_alerts * 100.0) if total_alerts > 0 else 0.0
        )

        return StatisticsResponse(
            total_transactions=total_txs,
            total_wallets=total_wallets,
            total_alerts=total_alerts,
            tier_counts=tier_counts,
            status_counts=status_counts,
            synthetic_alerts_count=synthetic_count,
            synthetic_alerts_percentage=round(synthetic_pct, 1),
            is_offline_mode=True,
        )

    def get_subgraph(
        self,
        entity_id: str | int,
        depth: int = 1,
        relationship: str | None = None,
        max_nodes: int = 100,
    ) -> GraphResponse | None:
        """Extract ego subgraph around an entity up to depth hops."""
        # Convert node id to match graph node representation
        target_node = entity_id
        if (
            not self.graph.has_node(target_node)
            and isinstance(entity_id, str)
            and entity_id.isdigit()
        ):
            target_node = int(entity_id)

        if not self.graph.has_node(target_node):
            return None

        ego = get_ego_graph(self.graph, target_node, radius=depth)
        if ego is None:
            return None

        node_type = self.graph.get_node_type(target_node) or "unknown"

        # Build nodes
        nodes: list[GraphNode] = []
        for n in list(ego.nodes())[:max_nodes]:
            ntype = ego.nodes[n].get("type", "unknown")
            # Lookup risk score
            rs = self.risk_scores.get(str(n))
            risk_score_val = rs.score if rs else None
            tier_val = rs.priority_tier.value if rs else None

            # Label lookup
            lbl = None
            if ntype == "transaction":
                tx = self.transactions.get(int(n) if isinstance(n, int) else 0)
                if tx and tx.label:
                    lbl = tx.label.name
            elif ntype == "wallet":
                w = self.wallets.get(str(n))
                if w and w.label:
                    lbl = w.label.name

            nodes.append(
                GraphNode(
                    id=str(n),
                    type=ntype,
                    label=lbl,
                    priority_tier=tier_val,
                    risk_score=risk_score_val,
                )
            )

        # Build edges
        node_id_set = {n.id for n in nodes}
        edges: list[GraphEdge] = []
        for u, v, data in ego.edges(data=True):
            if str(u) in node_id_set and str(v) in node_id_set:
                rel = data.get("relationship", "unknown")
                if relationship and rel != relationship:
                    continue
                edges.append(
                    GraphEdge(
                        source=str(u),
                        target=str(v),
                        relationship=rel,
                        is_synthetic=data.get("is_synthetic", False),
                        confidence=data.get("confidence"),
                        temporal_context=data.get("temporal_context"),
                        provenance=data.get("provenance"),
                    )
                )

        return GraphResponse(
            entity_id=str(entity_id),
            entity_type=node_type,
            depth=depth,
            node_count=len(nodes),
            edge_count=len(edges),
            nodes=nodes,
            edges=edges,
        )

    def find_entity_path(
        self,
        source: str | int,
        target: str | int,
    ) -> GraphPathResponse | None:
        """Find the shortest path connecting two investigative entities."""
        src = int(source) if isinstance(source, str) and source.isdigit() else source
        dst = int(target) if isinstance(target, str) and target.isdigit() else target

        if not self.graph.has_node(src) or not self.graph.has_node(dst):
            return None

        path = find_shortest_path(self.graph, src, dst)
        if not path:
            return GraphPathResponse(
                source=str(source),
                target=str(target),
                found=False,
                path_length=None,
                path_nodes=[],
                path_edges=[],
            )

        # Extract edges along path
        edges: list[GraphEdge] = []
        for i in range(len(path) - 1):
            u, v = path[i], path[i + 1]
            edge_data_dict = self.graph.G.get_edge_data(u, v)
            if edge_data_dict:
                # MultiDiGraph returns a dict of keys -> attrs
                first_edge = next(iter(edge_data_dict.values()))
                edges.append(
                    GraphEdge(
                        source=str(u),
                        target=str(v),
                        relationship=first_edge.get("relationship", "unknown"),
                        is_synthetic=first_edge.get("is_synthetic", False),
                        confidence=first_edge.get("confidence"),
                        temporal_context=first_edge.get("temporal_context"),
                        provenance=first_edge.get("provenance"),
                    )
                )

        return GraphPathResponse(
            source=str(source),
            target=str(target),
            found=True,
            path_length=len(path) - 1,
            path_nodes=[str(p) for p in path],
            path_edges=edges,
        )

    def seed_sample_data(self) -> None:
        """Seed representative base investigation scenario (Operation Shadow Mixer)."""
        demo = build_demo_dataset()
        self.transactions.clear()
        self.wallets.clear()
        self.risk_scores.clear()
        self.correlations.clear()
        self.alerts.clear()
        self.graph = KautilyaGraph()

        self.transactions.update(demo.transactions)
        self.wallets.update(demo.wallets)
        self.risk_scores.update(demo.risk_scores)
        self.correlations.update(demo.correlations)
        self.alerts.update(demo.alerts)

        self.graph.add_addr_tx_edges(demo.graph_edges["addr_tx"])
        self.graph.add_tx_addr_edges(demo.graph_edges["tx_addr"])
        self.graph.add_tx_tx_edges(demo.graph_edges["tx_tx"])
        self.graph.add_addr_addr_edges(demo.graph_edges["addr_addr"])
        self.simulation_step = 7

    def seed_expanded_data(self) -> None:
        """Seed 110 transactions across 5 forensic clusters for rich live demonstration."""
        from backend.api.expanded_data import build_expanded_dataset
        exp = build_expanded_dataset()
        self.transactions.clear()
        self.wallets.clear()
        self.risk_scores.clear()
        self.correlations.clear()
        self.alerts.clear()
        self.graph = KautilyaGraph()

        self.transactions.update(exp.transactions)
        self.wallets.update(exp.wallets)
        self.risk_scores.update(exp.risk_scores)
        self.correlations.update(exp.correlations)
        self.alerts.update(exp.alerts)

        self.graph.add_addr_tx_edges(exp.graph_edges["addr_tx"])
        self.graph.add_tx_addr_edges(exp.graph_edges["tx_addr"])
        self.graph.add_tx_tx_edges(exp.graph_edges["tx_tx"])
        self.graph.add_addr_addr_edges(exp.graph_edges["addr_addr"])
        self.simulation_step = len(exp.transactions)

    def reset_simulation(self, mode: str = "baseline") -> dict[str, Any]:
        """Reset the service state to baseline or full scenario."""
        from backend.api.simulation import get_simulation_engine
        engine = get_simulation_engine()
        return engine.reset(self, mode=mode)

    def inject_simulation_step(self) -> dict[str, Any]:
        """Inject the next step in the investigation scenario."""
        from backend.api.simulation import get_simulation_engine
        engine = get_simulation_engine()
        return engine.inject_next(self)

    def inject_simulation_batch(self, batch_size: int = 10) -> dict[str, Any]:
        """Inject a batch of transactions into the simulation."""
        from backend.api.simulation import get_simulation_engine
        engine = get_simulation_engine()
        return engine.inject_batch(self, batch_size=batch_size)

    def get_simulation_status(self) -> dict[str, Any]:
        """Return the current simulation progress and next step."""
        from backend.api.simulation import get_simulation_engine
        engine = get_simulation_engine()
        return engine.get_status(self)


# Default singleton instance
_DEFAULT_SERVICE: InvestigationService | None = None


def get_investigation_service() -> InvestigationService:
    """Return or initialize the singleton InvestigationService."""
    global _DEFAULT_SERVICE
    if _DEFAULT_SERVICE is None:
        _DEFAULT_SERVICE = InvestigationService()
        _DEFAULT_SERVICE.seed_sample_data()
    return _DEFAULT_SERVICE


def set_investigation_service(service: InvestigationService) -> None:
    """Override singleton service (useful for tests)."""
    global _DEFAULT_SERVICE
    _DEFAULT_SERVICE = service
