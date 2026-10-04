"""Simulation engine for realistic Bitcoin transaction streaming and mempool injection.

Supports 110 transactions, 50 wallets, 250+ graph edges, and forensic telemetry:
- Step-by-step or automated stream across darknet, ransomware, mixer, and commerce clusters
- Rich Bitcoin node telemetry (Block #854,230, fee rates, mempool backlog)
- Instant triage updates in Overview, Alert Queue, and Investigation Graph
"""

from typing import Any

from backend.api.demo_data import (
    TX_1001,
    TX_1002,
    TX_1003,
    TX_1004,
    TX_1005,
    TX_1006,
    TX_1007,
)
from backend.api.expanded_data import build_expanded_dataset
from backend.domain.alert import InvestigativeAlert
from backend.domain.graph import AddrAddrEdge, AddrTxEdge, TxAddrEdge, TxTxEdge
from backend.domain.risk import RiskScore
from backend.domain.transaction import Transaction
from backend.domain.wallet import Wallet
from backend.graph.builder import KautilyaGraph

# Bitcoin node telemetry constants for professional UI display
BLOCK_HEIGHT = 854230
MEMPOOL_BACKLOG = 1420
MEDIAN_FEE_RATE = 14.2  # sat/vB
NETWORK_HASHRATE = "642.8 EH/s"
NODE_STATUS = "SYNCED (LOCAL REPLICA)"


class SimulationEngine:
    """Manages streaming and injection of 110 transactions."""

    def __init__(self) -> None:
        self._dataset = build_expanded_dataset()
        self._all_transactions = self._dataset.transactions
        self._all_wallets = self._dataset.wallets
        self._all_risk_scores = self._dataset.risk_scores
        self._all_correlations = self._dataset.correlations
        self._all_edges = self._dataset.graph_edges
        self._all_alerts = self._dataset.alerts

        # Order transactions logically:
        # 1. Baseline control (1005)
        # 2-7. Core Shadow Mixer scenario (1001, 1002, 1003, 1004, 1006, 1007)
        # 8-110. Expanded clusters (2001 through 2103)
        core_txs = [TX_1005, TX_1001, TX_1002, TX_1003, TX_1004, TX_1006, TX_1007]
        extra_txs = sorted([txid for txid in self._all_transactions.keys() if txid not in core_txs])
        self._ordered_txids: list[int] = core_txs + extra_txs

    @property
    def total_transactions_count(self) -> int:
        return len(self._ordered_txids)

    def _get_tx_summary(self, txid: int) -> dict[str, str]:
        """Generate descriptive forensic headline for each transaction."""
        if txid == TX_1005:
            return {
                "title": "TX 1005: Baseline Licit Control Payment",
                "headline": "0.50 BTC merchant payment between verified compliant wallets.",
                "narrative": "System baseline: Normal commerce flow with zero deviance flags.",
            }
        elif txid == TX_1001:
            return {
                "title": "TX 1001: Darknet Market Influx (14.5 BTC)",
                "headline": "14.50 BTC transferred from darknet vendor to mixing service hub.",
                "narrative": "Supervised Random Forest classifier evaluated: 94.2% illicit probability. CRITICAL alert generated.",
            }
        elif txid == TX_1002:
            return {
                "title": "TX 1002: Mixer Peel Split A (7.2 BTC)",
                "headline": "Mixer peels 7.20 BTC to intermediate layering hop.",
                "narrative": "Isolation Forest flags statistical anomaly (0.88 deviance). Hub topology forming.",
            }
        elif txid == TX_1003:
            return {
                "title": "TX 1003: Mixer Parallel Peel B (7.0 BTC)",
                "headline": "Secondary parallel peel dispatched within 2-minute processing window.",
                "narrative": "High-frequency churn confirms automated algorithmic mixing service.",
            }
        elif txid == TX_1004:
            return {
                "title": "TX 1004: Layering Convergence (13.8 BTC)",
                "headline": "Peels converge into consolidation transfer towards OTC cash-out desk.",
                "narrative": "End-to-end 4-hop laundering path established in forensic graph.",
            }
        elif txid == TX_1006:
            return {
                "title": "TX 1006: Corroborating Influx (8.3 BTC)",
                "headline": "Second dirty transfer from darknet vendor to mixer.",
                "narrative": "Network correlation confirms identical source IP 198.51.100.42.",
            }
        elif txid == TX_1007:
            return {
                "title": "TX 1007: Fast Churn Liquidation (8.1 BTC)",
                "headline": "Direct churn transfer from mixer to cash-out desk completes scenario.",
                "narrative": "Rapid turnaround cash-out verified across mempool blocks.",
            }
        elif 2001 <= txid <= 2025:
            return {
                "title": f"TX {txid}: Darknet Vendor Payment ({self._all_transactions[txid].total_btc} BTC)",
                "headline": f"Darknet marketplace transaction flagged with elevated fee structure.",
                "narrative": "Classified as high-risk illicit flow; multi-sig inputs identified.",
            }
        elif 2026 <= txid <= 2050:
            return {
                "title": f"TX {txid}: Ransomware Peel Chain ({self._all_transactions[txid].total_btc} BTC)",
                "headline": f"Extortion payout hop peeling funds into secondary sub-wallets.",
                "narrative": "Sequential peel chain pattern with tight temporal clustering.",
            }
        elif 2051 <= txid <= 2075:
            return {
                "title": f"TX {txid}: Mixing Pool Fan-Out ({self._all_transactions[txid].total_btc} BTC)",
                "headline": f"CoinJoin mixing pool fan-out detected with 4+ output addresses.",
                "narrative": "High fan-out anomaly score registered by unsupervised model.",
            }
        else:
            return {
                "title": f"TX {txid}: Compliant Merchant Commerce ({self._all_transactions[txid].total_btc} BTC)",
                "headline": "Routine commercial transaction via regulated payment processor.",
                "narrative": "Normal behavioral baseline; zero risk signals triggered.",
            }

    def apply_tx(self, service: Any, txid: int) -> dict[str, Any]:
        """Inject a single transaction and its linked entities into service memory."""
        tx = self._all_transactions.get(txid)
        if not tx:
            return {}

        service.transactions[txid] = tx

        # Add linked graph edges
        edges_added = 0
        for edge in self._all_edges.get("addr_tx", []):
            if edge.txid == txid:
                service.graph.add_addr_tx_edges([edge])
                if edge.input_address in self._all_wallets:
                    service.wallets[edge.input_address] = self._all_wallets[edge.input_address]
                edges_added += 1

        for edge in self._all_edges.get("tx_addr", []):
            if edge.txid == txid:
                service.graph.add_tx_addr_edges([edge])
                if edge.output_address in self._all_wallets:
                    service.wallets[edge.output_address] = self._all_wallets[edge.output_address]
                edges_added += 1

        for edge in self._all_edges.get("tx_tx", []):
            if edge.source_txid == txid or edge.target_txid == txid:
                if edge.source_txid in service.transactions and edge.target_txid in service.transactions:
                    service.graph.add_tx_tx_edges([edge])
                    edges_added += 1

        for edge in self._all_edges.get("addr_addr", []):
            if edge.input_address in service.wallets and edge.output_address in service.wallets:
                service.graph.add_addr_addr_edges([edge])
                edges_added += 1

        # Add risk score
        rs_key = f"tx-{txid}"
        if rs_key in self._all_risk_scores:
            service.risk_scores[rs_key] = self._all_risk_scores[rs_key]

        # Add correlations
        if txid in self._all_correlations:
            service.correlations[txid] = self._all_correlations[txid]

        # Add alerts
        added_alerts = []
        for alert_id, alert in self._all_alerts.items():
            if alert.entity_id == str(txid):
                service.alerts[alert_id] = alert
                added_alerts.append(alert_id)

        meta = self._get_tx_summary(txid)
        return {
            "txid": txid,
            "title": meta["title"],
            "headline": meta["headline"],
            "narrative": meta["narrative"],
            "amount_btc": tx.total_btc,
            "label": tx.label.value if hasattr(tx.label, "value") else str(tx.label),
            "added_alerts": added_alerts,
            "edges_added_count": edges_added,
        }

    def reset(self, service: Any, mode: str = "baseline") -> dict[str, Any]:
        """Reset service state to baseline (1 control tx) or full (110 txs)."""
        service.transactions.clear()
        service.wallets.clear()
        service.risk_scores.clear()
        service.correlations.clear()
        service.alerts.clear()
        service.graph = KautilyaGraph()

        if mode == "full":
            # Apply all 110 transactions
            for txid in self._ordered_txids:
                self.apply_tx(service, txid)
            service.simulation_step = len(self._ordered_txids)
        else:
            # Baseline: apply Step 0 (TX 1005) only
            self.apply_tx(service, self._ordered_txids[0])
            service.simulation_step = 1

        return self.get_status(service)

    def inject_next(self, service: Any) -> dict[str, Any]:
        """Advance simulation by one transaction."""
        current = getattr(service, "simulation_step", 1)
        if current >= len(self._ordered_txids):
            return {
                "success": False,
                "message": f"All {len(self._ordered_txids)} transactions already ingested from mempool.",
                "status": self.get_status(service),
            }

        next_txid = self._ordered_txids[current]
        injected = self.apply_tx(service, next_txid)
        service.simulation_step = current + 1

        return {
            "success": True,
            "injected": injected,
            "status": self.get_status(service),
        }

    def inject_batch(self, service: Any, batch_size: int = 10) -> dict[str, Any]:
        """Advance simulation by a batch of transactions."""
        current = getattr(service, "simulation_step", 1)
        injected_list = []
        for _ in range(batch_size):
            if current >= len(self._ordered_txids):
                break
            txid = self._ordered_txids[current]
            res = self.apply_tx(service, txid)
            injected_list.append(res)
            current += 1

        service.simulation_step = current
        return {
            "success": True,
            "batch_count": len(injected_list),
            "injected": injected_list[-1] if injected_list else None,
            "batch": injected_list,
            "status": self.get_status(service),
        }

    def get_status(self, service: Any) -> dict[str, Any]:
        """Return current status and Bitcoin network telemetry."""
        current = getattr(service, "simulation_step", len(self._ordered_txids))
        total = len(self._ordered_txids)

        last_txid = self._ordered_txids[min(current - 1, total - 1)] if current > 0 else self._ordered_txids[0]
        next_txid = self._ordered_txids[current] if current < total else None

        last_meta = self._get_tx_summary(last_txid)
        next_meta = self._get_tx_summary(next_txid) if next_txid else None

        return {
            "current_step": current,
            "total_steps": total,
            "block_height": BLOCK_HEIGHT,
            "mempool_backlog": MEMPOOL_BACKLOG,
            "median_fee_rate": MEDIAN_FEE_RATE,
            "network_hashrate": NETWORK_HASHRATE,
            "node_status": NODE_STATUS,
            "step_title": last_meta["title"],
            "step_headline": last_meta["headline"],
            "narrative": last_meta["narrative"],
            "can_inject_next": current < total,
            "next_txid": next_txid,
            "next_step_title": next_meta["title"] if next_meta else "Mempool Drain Complete",
            "total_transactions": len(service.transactions),
            "total_wallets": len(service.wallets),
            "total_alerts": len(service.alerts),
        }


# Singleton engine instance
_SIMULATION_ENGINE: SimulationEngine | None = None


def get_simulation_engine() -> SimulationEngine:
    """Return singleton SimulationEngine instance."""
    global _SIMULATION_ENGINE
    if _SIMULATION_ENGINE is None:
        _SIMULATION_ENGINE = SimulationEngine()
    return _SIMULATION_ENGINE
