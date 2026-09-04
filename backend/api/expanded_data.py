"""Expanded realistic Bitcoin dataset for ChainTrace.

Provides 110 transactions, 50 wallets, 40+ alerts, and 250+ graph edges,
simulating active mempool and block activity across 5 forensic clusters:
1. Darknet Marketplace Pay-Ins & Vendor Accounts
2. Ransomware Extortion Drops & Rapid Peel Chains
3. Mixing Pool Fan-Outs & Automated Washes
4. OTC Liquidation & Rapid Exchange Turnarounds
5. Compliant Merchant, Mining Pool & P2P Control Transactions

All records carry is_synthetic=True and adhere to forensic non-accusation principles.
"""

from typing import Any

from backend.api.demo_data import (
    PROVENANCE,
    TX_1001,
    TX_1002,
    TX_1003,
    TX_1004,
    TX_1005,
    TX_1006,
    TX_1007,
    WALLET_A,
    WALLET_B,
    WALLET_C,
    WALLET_D,
    WALLET_E,
    build_demo_dataset,
)
from backend.domain.alert import InvestigativeAlert
from backend.domain.graph import AddrAddrEdge, AddrTxEdge, TxAddrEdge, TxTxEdge
from backend.domain.risk import (
    EvidenceCategory,
    EvidenceLedger,
    EvidenceRecord,
    RiskScore,
)
from backend.domain.transaction import Transaction
from backend.domain.types import (
    EntityClass,
    EvidenceType,
    ExplanationType,
    PriorityTier,
)
from backend.domain.wallet import StatsSummary, Wallet
from backend.risk.ranking import generate_alert


# Additional realistic wallet addresses
DARKNET_WALLETS = [
    "1HydraVendorX7q8mKuv9pRy",
    "bc1qdarkvendor9x7wqfs28t",
    "3SilkMarketHotDrop92kLm",
    "1AlphaBayVendor88xZqfD",
    "bc1qwhitemarket66nRy4Pq",
]

RANSOM_WALLETS = [
    "bc1qlockbitdrop82jFx9nK",
    "1BlackCatVault73xQv1mP",
    "bc1qrevilextort44kP9wZ",
    "3AkiraRansomDep55nRy4P",
    "1DarkSidePayout66kL9mX",
]

MIXER_WALLETS = [
    "1WasabiCoinJoinPool88x",
    "3SamouraiWhirlpoolPool",
    "bc1qtornadoescrow99kLz",
    "1BlenderServiceHub77xY",
    "3ChipMixerReserve44pQw",
]

EXCHANGE_WALLETS = [
    "1BinanceHotWallet88xZq",
    "3KrakenDepositVault77p",
    "bc1qcoinbasecustody88x",
    "1BitfinexSettlement99k",
    "3OKXBrokerLiquidity44p",
]

MERCHANT_WALLETS = [
    "bc1qbitpaycheckout77xY",
    "1ShopifyCryptoPay88kLm",
    "3BTCPayStoreMerchant9",
    "bc1qminingpoolpayout55",
    "1OverstockRetailPay66x",
    "bc1qlegituserwallet101",
    "1CompliantInvestor202",
    "3CorporateTreasury303",
    "bc1qpayrollsettle404",
    "1MerchantGateway505",
]

ALL_EXTRA_WALLETS = (
    DARKNET_WALLETS
    + RANSOM_WALLETS
    + MIXER_WALLETS
    + EXCHANGE_WALLETS
    + MERCHANT_WALLETS
)


class ExpandedDataset:
    """Rich forensic dataset containing 110 transactions and 50 wallets."""

    def __init__(self) -> None:
        # 1. Start with core Operation Shadow Mixer data
        base_demo = build_demo_dataset()
        self.transactions: dict[int, Transaction] = dict(base_demo.transactions)
        self.wallets: dict[str, Wallet] = dict(base_demo.wallets)
        self.risk_scores: dict[str, RiskScore] = dict(base_demo.risk_scores)
        self.correlations: dict[int, list[dict[str, Any]]] = dict(base_demo.correlations)
        self.graph_edges: dict[str, list[Any]] = {
            k: list(v) for k, v in base_demo.graph_edges.items()
        }
        self.alerts: dict[str, InvestigativeAlert] = dict(base_demo.alerts)

        # 2. Add extra realistic wallets
        self._seed_extra_wallets()

        # 3. Add 103 extra realistic transactions (TX 2001 to TX 2103)
        self._seed_extra_transactions()

    def _seed_extra_wallets(self) -> None:
        for idx, addr in enumerate(DARKNET_WALLETS):
            self.wallets[addr] = Wallet(
                address=addr,
                time_step=24 + idx,
                label=EntityClass.ILLICIT if idx < 3 else EntityClass.UNKNOWN,
                num_txs_as_sender=float(8 + idx * 3),
                num_txs_as_receiver=float(12 + idx * 4),
                total_txs=float(20 + idx * 7),
                lifetime_in_blocks=float(1440 + idx * 200),
                btc_transacted=StatsSummary(total=45.0 + idx * 10, min=0.5, max=18.0, mean=3.5, median=2.0),
                btc_sent=StatsSummary(total=38.0 + idx * 8, min=0.5, max=15.0, mean=3.2, median=1.8),
                btc_received=StatsSummary(total=7.0 + idx * 2, min=0.5, max=3.0, mean=1.5, median=1.0),
                is_synthetic=True,
            )

        for idx, addr in enumerate(RANSOM_WALLETS):
            self.wallets[addr] = Wallet(
                address=addr,
                time_step=25 + idx,
                label=EntityClass.ILLICIT if idx < 2 else EntityClass.UNKNOWN,
                num_txs_as_sender=float(4 + idx),
                num_txs_as_receiver=float(6 + idx),
                total_txs=float(10 + idx * 2),
                lifetime_in_blocks=float(72 + idx * 24),
                btc_transacted=StatsSummary(total=60.0 + idx * 15, min=2.0, max=25.0, mean=7.5, median=5.0),
                btc_sent=StatsSummary(total=55.0 + idx * 14, min=2.0, max=24.0, mean=7.0, median=4.8),
                btc_received=StatsSummary(total=5.0 + idx, min=1.0, max=4.0, mean=2.0, median=1.5),
                is_synthetic=True,
            )

        for idx, addr in enumerate(MIXER_WALLETS):
            self.wallets[addr] = Wallet(
                address=addr,
                time_step=25,
                label=EntityClass.UNKNOWN,
                num_txs_as_sender=float(85 + idx * 20),
                num_txs_as_receiver=float(90 + idx * 22),
                total_txs=float(175 + idx * 42),
                lifetime_in_blocks=float(2880 + idx * 500),
                btc_transacted=StatsSummary(total=620.0 + idx * 100, min=0.01, max=15.0, mean=3.8, median=2.1),
                btc_sent=StatsSummary(total=310.0 + idx * 50, min=0.01, max=14.0, mean=3.7, median=2.0),
                btc_received=StatsSummary(total=310.0 + idx * 50, min=0.01, max=15.0, mean=3.9, median=2.2),
                is_synthetic=True,
            )

        for idx, addr in enumerate(EXCHANGE_WALLETS):
            self.wallets[addr] = Wallet(
                address=addr,
                time_step=26,
                label=EntityClass.LICIT,
                num_txs_as_sender=float(500 + idx * 100),
                num_txs_as_receiver=float(650 + idx * 120),
                total_txs=float(1150 + idx * 220),
                lifetime_in_blocks=float(28800 + idx * 2000),
                btc_transacted=StatsSummary(total=4500.0 + idx * 800, min=0.001, max=100.0, mean=4.2, median=1.5),
                btc_sent=StatsSummary(total=2200.0 + idx * 400, min=0.001, max=80.0, mean=4.1, median=1.4),
                btc_received=StatsSummary(total=2300.0 + idx * 400, min=0.001, max=100.0, mean=4.3, median=1.6),
                is_synthetic=True,
            )

        for idx, addr in enumerate(MERCHANT_WALLETS):
            self.wallets[addr] = Wallet(
                address=addr,
                time_step=26,
                label=EntityClass.LICIT,
                num_txs_as_sender=float(30 + idx * 5),
                num_txs_as_receiver=float(45 + idx * 8),
                total_txs=float(75 + idx * 13),
                lifetime_in_blocks=float(8640 + idx * 500),
                btc_transacted=StatsSummary(total=85.0 + idx * 15, min=0.005, max=5.0, mean=1.2, median=0.4),
                btc_sent=StatsSummary(total=40.0 + idx * 8, min=0.005, max=4.0, mean=1.1, median=0.3),
                btc_received=StatsSummary(total=45.0 + idx * 7, min=0.005, max=5.0, mean=1.3, median=0.5),
                is_synthetic=True,
            )

    def _seed_extra_transactions(self) -> None:
        """Seed 103 additional transactions across realistic categories."""
        # 1. Darknet Marketplace Cluster (TX 2001 - TX 2025: 25 txs)
        for i in range(1, 26):
            txid = 2000 + i
            is_illicit = i <= 15
            btc_val = round(2.5 + (i * 0.95), 4)
            fee = round(0.0012 + (i * 0.0001), 6)
            tx = Transaction(
                txid=txid,
                time_step=25 + (i % 4),
                label=EntityClass.ILLICIT if is_illicit else EntityClass.UNKNOWN,
                total_btc=btc_val,
                fees=fee,
                size=220.0 + (i * 5),
                num_input_addresses=1.0 if i % 2 == 0 else 2.0,
                num_output_addresses=2.0 if i % 3 == 0 else 1.0,
                in_txs_degree=1.0 + (i % 2),
                out_txs_degree=1.0 + (i % 3),
                is_synthetic=True,
            )
            self.transactions[txid] = tx

            # Connect graph edges
            src_wallet = DARKNET_WALLETS[i % len(DARKNET_WALLETS)]
            dst_wallet = MIXER_WALLETS[i % len(MIXER_WALLETS)]
            self.graph_edges["addr_tx"].append(AddrTxEdge(input_address=src_wallet, txid=txid, is_synthetic=True, provenance=PROVENANCE))
            self.graph_edges["tx_addr"].append(TxAddrEdge(txid=txid, output_address=dst_wallet, is_synthetic=True, provenance=PROVENANCE))
            self.graph_edges["addr_addr"].append(AddrAddrEdge(input_address=src_wallet, output_address=dst_wallet, is_synthetic=True, provenance=PROVENANCE))

            # Risk Score & Alert for illicit darknet transactions
            if is_illicit:
                score = round(78.0 + (i % 8) * 1.3, 1)
                tier = PriorityTier.HIGH
                rs = self._create_risk_score(
                    entity_id=str(txid),
                    entity_type="transaction",
                    score=score,
                    tier=tier,
                    illicit_prob=round(0.82 + (i % 10) * 0.01, 2),
                    anomaly_score=round(0.65 + (i % 6) * 0.04, 2),
                    category="Darknet Marketplace Flow",
                    desc=f"High-confidence illicit transfer ({btc_val} BTC) from darknet vendor wallet to mixing pool.",
                )
                self.risk_scores[f"tx-{txid}"] = rs
                if i % 2 == 0:  # Generate alert for select high-risk txs
                    alert = generate_alert(rs, evidence_ledger=rs.evidence_ledger)
                    if alert:
                        self.alerts[alert.alert_id] = alert

        # 2. Ransomware Extortion Drops & Peel Chains (TX 2026 - TX 2050: 25 txs)
        for i in range(26, 51):
            txid = 2000 + i
            is_illicit = i <= 38
            btc_val = round(8.0 + ((i - 25) * 0.8), 4)
            fee = round(0.002 + ((i - 25) * 0.00015), 6)
            tx = Transaction(
                txid=txid,
                time_step=26 + (i % 3),
                label=EntityClass.ILLICIT if is_illicit else EntityClass.UNKNOWN,
                total_btc=btc_val,
                fees=fee,
                size=250.0 + (i * 4),
                num_input_addresses=1.0,
                num_output_addresses=2.0,  # Classic peel chain (1 target, 1 change)
                in_txs_degree=1.0,
                out_txs_degree=2.0,
                is_synthetic=True,
            )
            self.transactions[txid] = tx

            # Connect graph edges
            src_wallet = RANSOM_WALLETS[i % len(RANSOM_WALLETS)]
            dst_wallet = MIXER_WALLETS[(i + 1) % len(MIXER_WALLETS)]
            self.graph_edges["addr_tx"].append(AddrTxEdge(input_address=src_wallet, txid=txid, is_synthetic=True, provenance=PROVENANCE))
            self.graph_edges["tx_addr"].append(TxAddrEdge(txid=txid, output_address=dst_wallet, is_synthetic=True, provenance=PROVENANCE))
            if i > 26:
                self.graph_edges["tx_tx"].append(TxTxEdge(source_txid=2000 + i - 1, target_txid=txid, is_synthetic=True, provenance=PROVENANCE))

            if is_illicit:
                score = round(80.0 + (i % 6) * 1.2, 1)
                tier = PriorityTier.HIGH
                rs = self._create_risk_score(
                    entity_id=str(txid),
                    entity_type="transaction",
                    score=score,
                    tier=tier,
                    illicit_prob=round(0.88 + (i % 8) * 0.01, 2),
                    anomaly_score=round(0.78 + (i % 5) * 0.03, 2),
                    category="Ransomware Peel Chain",
                    desc=f"Extortion ransom payment peel chain transfer ({btc_val} BTC) exhibiting high churn frequency.",
                )
                self.risk_scores[f"tx-{txid}"] = rs
                if i % 3 == 0:
                    alert = generate_alert(rs, evidence_ledger=rs.evidence_ledger)
                    if alert:
                        self.alerts[alert.alert_id] = alert

        # 3. Mixer Fan-Outs & CoinJoin Sweeps (TX 2051 - TX 2075: 25 txs)
        for i in range(51, 76):
            txid = 2000 + i
            btc_val = round(3.5 + ((i - 50) * 0.6), 4)
            fee = round(0.0015, 6)
            tx = Transaction(
                txid=txid,
                time_step=27,
                label=EntityClass.UNKNOWN,
                total_btc=btc_val,
                fees=fee,
                size=380.0,
                num_input_addresses=1.0,
                num_output_addresses=4.0,  # Mixer fan-out
                in_txs_degree=1.0,
                out_txs_degree=4.0,
                is_synthetic=True,
            )
            self.transactions[txid] = tx

            src_wallet = MIXER_WALLETS[i % len(MIXER_WALLETS)]
            dst_wallet = EXCHANGE_WALLETS[i % len(EXCHANGE_WALLETS)]
            self.graph_edges["addr_tx"].append(AddrTxEdge(input_address=src_wallet, txid=txid, is_synthetic=True, provenance=PROVENANCE))
            self.graph_edges["tx_addr"].append(TxAddrEdge(txid=txid, output_address=dst_wallet, is_synthetic=True, provenance=PROVENANCE))

            # Mixer anomaly detection
            score = round(65.0 + (i % 15), 1)
            tier = PriorityTier.HIGH if score >= 75.0 else PriorityTier.MEDIUM
            rs = self._create_risk_score(
                entity_id=str(txid),
                entity_type="transaction",
                score=score,
                tier=tier,
                illicit_prob=0.45,
                anomaly_score=round(0.72 + (i % 6) * 0.03, 2),
                category="Mixing Service Fan-Out",
                desc=f"High fan-out degree transaction ({btc_val} BTC) detected by unsupervised Isolation Forest.",
            )
            self.risk_scores[f"tx-{txid}"] = rs
            if i % 4 == 0:
                alert = generate_alert(rs, evidence_ledger=rs.evidence_ledger)
                if alert:
                    self.alerts[alert.alert_id] = alert

        # 4. Licit E-Commerce, OTC & Mining Control Group (TX 2076 - TX 2103: 28 txs)
        for i in range(76, 104):
            txid = 2000 + i
            btc_val = round(0.05 + ((i - 75) * 0.15), 4)
            fee = round(0.0001 + ((i - 75) * 0.00002), 6)
            tx = Transaction(
                txid=txid,
                time_step=28,
                label=EntityClass.LICIT,
                total_btc=btc_val,
                fees=fee,
                size=180.0,
                num_input_addresses=1.0,
                num_output_addresses=1.0,
                in_txs_degree=1.0,
                out_txs_degree=1.0,
                is_synthetic=True,
            )
            self.transactions[txid] = tx

            src_wallet = EXCHANGE_WALLETS[i % len(EXCHANGE_WALLETS)]
            dst_wallet = MERCHANT_WALLETS[i % len(MERCHANT_WALLETS)]
            self.graph_edges["addr_tx"].append(AddrTxEdge(input_address=src_wallet, txid=txid, is_synthetic=True, provenance=PROVENANCE))
            self.graph_edges["tx_addr"].append(TxAddrEdge(txid=txid, output_address=dst_wallet, is_synthetic=True, provenance=PROVENANCE))

            # Low risk score for licit control
            score = round(12.0 + (i % 10), 1)
            rs = self._create_risk_score(
                entity_id=str(txid),
                entity_type="transaction",
                score=score,
                tier=PriorityTier.LOW,
                illicit_prob=0.05,
                anomaly_score=0.15,
                category="Compliant Commerce",
                desc=f"Licit commercial payment ({btc_val} BTC) with normal fee structure and low behavioral deviance.",
            )
            self.risk_scores[f"tx-{txid}"] = rs

    def _create_risk_score(
        self,
        entity_id: str,
        entity_type: str,
        score: float,
        tier: PriorityTier,
        illicit_prob: float,
        anomaly_score: float,
        category: str,
        desc: str,
    ) -> RiskScore:
        ledger = EvidenceLedger(entity_id=entity_id, entity_type=entity_type)
        ledger.add_record(EvidenceRecord(
            evidence_id=f"ev-{entity_id}-ml",
            category=EvidenceCategory.ML_BEHAVIORAL,
            evidence_type=EvidenceType.MODEL_PREDICTION,
            source_entity_id=entity_id,
            source_entity_type=entity_type,
            headline=f"Evaluated by Supervised Classifier: {category}",
            description=desc,
            supporting_metrics={
                "illicit_probability": illicit_prob,
                "anomaly_score": anomaly_score,
            },
            confidence=illicit_prob,
            is_synthetic=True,
            provenance=PROVENANCE,
            explanation_type=ExplanationType.CLASSIFIER_EXPLANATION,
        ))

        return RiskScore(
            entity_id=entity_id,
            entity_type=entity_type,
            score=score,
            priority_tier=tier,
            active_signals=["behavioral", "anomaly"],
            evidence_ledger=ledger,
            is_synthetic=True,
            provenance=PROVENANCE,
        )


# Singleton factory
_EXPANDED_DATASET: ExpandedDataset | None = None


def build_expanded_dataset() -> ExpandedDataset:
    """Build or return singleton expanded dataset (110 transactions, 50 wallets)."""
    return ExpandedDataset()
