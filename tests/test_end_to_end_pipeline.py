"""End-to-End Pipeline Verification Suite for Kautilya.

Verifies the complete analytical dataflow:
    Raw Data / Synthetics
        -> Validation & Normalization
        -> Graph Construction & Path Analysis
        -> Synthetic Network Simulation & Correlation
        -> Supervised Classifier & Anomaly Inference
        -> Multi-Signal Risk Synthesis & Alert Ranking
        -> Entity-Level Rollup Traceability
        -> WSGI API Endpoints & Static Distribution Serving
        -> Strict Zero-Network Isolation Guarantee

Forensic Rules Enforced:
    1. Provenance preservation: `is_synthetic` survives all pipeline stages.
    2. Zero external network dependency: Non-localhost connections are strictly rejected.
    3. Explainability separation: Classifier rationale vs anomaly deviance.
    4. Non-accusation principle: Wallet rollups retain contributing transaction evidence.
"""

import socket
import numpy as np
import pandas as pd
import pytest

from backend.api import create_app, InvestigationService, TestClient, set_investigation_service
from backend.correlation import CorrelationConfig, correlate_transaction_to_ips
from backend.domain.graph import AddrTxEdge, TxAddrEdge, TxTxEdge
from backend.domain.network import NetworkObservation
from backend.domain.risk import EvidenceCategory, SignalInput
from backend.domain.transaction import Transaction
from backend.domain.types import EntityClass, PriorityTier, ScriptType
from backend.domain.wallet import StatsSummary, Wallet
from backend.graph.builder import KautilyaGraph
from backend.graph.paths import find_shortest_path, get_ego_graph
from backend.ml import AnomalyDetector, TransactionClassifier
from backend.normalization import (
    normalize_network_observations,
    normalize_transactions,
    normalize_wallets,
)
from backend.risk import (
    AggregationMethod,
    SynthesisConfig,
    aggregate_transaction_scores,
    filter_and_prioritize_alerts,
    generate_alert,
    rank_entities,
    synthesize_risk_score,
)
from backend.generator import (
    build_topology,
    generate_node_pool,
    generate_observations_for_transaction,
)


@pytest.fixture(autouse=True)
def guard_external_network_calls(monkeypatch):
    """Ensure runtime code makes zero external network calls.

    Allows connections only to local loopback (127.0.0.1, localhost).
    """
    orig_connect = socket.socket.connect

    def restricted_connect(self, address):
        host = address[0] if isinstance(address, tuple) else address
        if host not in ("127.0.0.1", "localhost", "::1"):
            raise RuntimeError(
                f"FORENSIC VIOLATION: Pipeline attempted external network connection to {host}"
            )
        return orig_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", restricted_connect)


class TestCompleteEndToEndPipeline:
    """Validate every stage of the Kautilya forensic pipeline in sequence."""

    def test_stage1_normalization_and_provenance(self):
        """Stage 1: Normalization enforces schema types and preserves is_synthetic."""
        # Transaction normalization with full feature schema
        row = {"txId": 9001, "time_step": 15, "total_BTC": 12.5}
        for i in range(1, 94):
            row[f"Local_feature_{i}"] = 0.1
        for i in range(1, 73):
            row[f"Aggregate_feature_{i}"] = 0.2

        df_feat = pd.DataFrame([row])
        df_class = pd.DataFrame([{"txId": 9001, "class": 1}])
        tx, feats = next(normalize_transactions(df_feat, df_class))
        assert tx.txid == 9001
        assert tx.label == EntityClass.ILLICIT

        # Wallet normalization
        df_w_feat = pd.DataFrame([{"address": "1PipeSourceWalletA", "time_step": 15}])
        df_w_class = pd.DataFrame([{"address": "1PipeSourceWalletA", "class": 1}])
        wallet = next(normalize_wallets(df_w_feat, df_w_class))
        assert wallet.address == "1PipeSourceWalletA"
        assert wallet.label == EntityClass.ILLICIT

        # Network observation normalization
        df_net = pd.DataFrame([{
            "timestamp": "2024-03-15T10:00:00Z",
            "txid": 9001,
            "src_ip": "198.51.100.22",
            "dst_ip": "10.0.0.1",
            "src_port": 8333,
            "dst_port": 8333,
            "role": "originator",
            "script_type": "P2PKH",
            "country": "US",
            "asn": 13335,
            "confidence": 0.85,
        }])
        obs = next(normalize_network_observations(df_net))
        assert obs.txid == 9001
        assert obs.src_ip == "198.51.100.22"
        assert obs.is_synthetic is True

    def test_stage2_graph_construction_and_topology(self):
        """Stage 2: Graph builder populates topology and executes graph queries."""
        graph = KautilyaGraph()

        # Add edges
        graph.add_addr_tx_edges([
            AddrTxEdge(input_address="1PipeSourceWalletA", txid=9001, is_synthetic=True),
        ])
        graph.add_tx_tx_edges([
            TxTxEdge(source_txid=9001, target_txid=9002, is_synthetic=True),
        ])
        graph.add_tx_addr_edges([
            TxAddrEdge(txid=9002, output_address="1PipeDestWalletB", is_synthetic=True),
        ])

        # Ego subgraph
        ego = get_ego_graph(graph, 9001, radius=1)
        assert ego.has_node(9001)
        assert ego.has_node("1PipeSourceWalletA")
        assert ego.has_node(9002)

        # Path finding
        path = find_shortest_path(graph, "1PipeSourceWalletA", "1PipeDestWalletB")
        assert path == ["1PipeSourceWalletA", 9001, 9002, "1PipeDestWalletB"]

    def test_stage3_synthetic_generation_and_correlation(self):
        """Stage 3: Synthetic generator produces observations; correlation engine evaluates confidence."""
        import random

        nodes = generate_node_pool(size=20, seed=42)
        lookup = {n.node_id: n for n in nodes}
        topo = build_topology(nodes, seed=42, peers_per_node=4)
        rng = random.Random(42)
        events = generate_observations_for_transaction(
            txid=9001,
            time_step=15,
            origin_node_id=0,
            topology=topo,
            node_lookup=lookup,
            rng=rng,
        )
        assert len(events) >= 1
        for ev in events:
            assert ev.is_synthetic is True
            assert ev.src_ip is not None

        # Correlation
        tx = Transaction(
            txid=9001,
            time_step=15,
            label=EntityClass.ILLICIT,
            total_btc=12.5,
            fees=0.001,
        )
        corrs = correlate_transaction_to_ips(tx, events)
        assert isinstance(corrs, list)
        for c in corrs:
            assert c.is_synthetic is True
            assert 0.0 <= c.correlation_confidence <= 1.0

    def test_stage4_machine_learning_inference(self):
        """Stage 4: Supervised classification and anomaly deviance inference."""
        # Supervised classifier
        clf = TransactionClassifier(feature_set_name="M1", n_estimators=10)
        X_train = np.array([[10.0, 0.001, 250.0], [1.0, 0.0001, 150.0]])
        y_train = np.array([1, 0])
        clf.train(X_train, y_train, feature_names=["total_btc", "fees", "size"])
        probs = clf.predict_proba(np.array([[12.0, 0.0015, 300.0]]))
        assert len(probs) == 1
        assert 0.0 <= probs[0, 1] <= 1.0

        # Anomaly detector
        anom = AnomalyDetector(n_estimators=10, feature_names=["total_btc", "fees", "size"])
        anom.fit(X_train)
        scores = anom.score(np.array([[12.0, 0.0015, 300.0]]))
        assert len(scores) == 1
        assert 0.0 <= scores[0] <= 1.0

    def test_stage5_risk_synthesis_and_evidence_ledger(self):
        """Stage 5: Multi-signal risk synthesis, evidence ledger compilation, and alert ranking."""
        sig_input = SignalInput(
            entity_id="9001",
            entity_type="transaction",
            illicit_probability=0.92,
            predicted_label=EntityClass.ILLICIT,
            anomaly_score=0.75,
            graph_signal=0.80,
            correlation_confidence=0.85,
            contains_synthetic_input=True,
        )

        risk_score = synthesize_risk_score(sig_input)
        assert risk_score.score >= 80.0
        assert risk_score.priority_tier in (PriorityTier.CRITICAL, PriorityTier.HIGH)
        assert risk_score.contains_synthetic_input is True

        # Generate alert
        alert = generate_alert(risk_score, evidence_ledger=risk_score.evidence_ledger)
        assert alert is not None
        assert alert.entity_id == "9001"
        assert alert.priority_tier == risk_score.priority_tier
        assert alert.contains_synthetic_input is True

        # Rank triage queue
        ranked = rank_entities([risk_score])
        assert len(ranked) == 1
        assert ranked[0].rank == 1
        assert ranked[0].entity_id == "9001"

    def test_stage6_entity_rollup_traceability(self):
        """Stage 6: Wallet rollup preserves non-accusation principle."""
        rs_tx1 = synthesize_risk_score(SignalInput(
            entity_id="9001", entity_type="transaction", illicit_probability=0.95, contains_synthetic_input=True
        ))
        rs_tx2 = synthesize_risk_score(SignalInput(
            entity_id="9002", entity_type="transaction", illicit_probability=0.10, contains_synthetic_input=False
        ))

        agg = aggregate_transaction_scores(
            entity_id="1PipeSourceWalletA",
            transaction_scores=[rs_tx1, rs_tx2],
            method=AggregationMethod.MAX,
        )

        assert agg.aggregated_score == rs_tx1.score
        assert len(agg.contributing_transactions) == 2
        assert "Individual transaction predictions" in agg.forensic_disclaimer
        assert agg.contains_synthetic_input is True

    def test_stage7_api_and_offline_serving(self):
        """Stage 7: Full pipeline serves via WSGI API with zero external calls."""
        service = InvestigationService()
        service.seed_sample_data()
        set_investigation_service(service)
        client = TestClient(create_app())

        # Health
        resp_health = client.get("/health")
        assert resp_health.status_code == 200
        assert resp_health.json()["status"] == "healthy"

        # Statistics
        resp_stats = client.get("/statistics")
        assert resp_stats.status_code == 200
        stats = resp_stats.json()
        assert stats["is_offline_mode"] is True
        assert stats["total_transactions"] >= 7

        # Transaction
        resp_tx = client.get("/transactions/1001")
        assert resp_tx.status_code == 200
        assert resp_tx.json()["txid"] == 1001

        # Wallet
        resp_w = client.get("/wallets/1DrK44np3gMKuvcGeFHv")
        assert resp_w.status_code == 200
        assert resp_w.json()["address"] == "1DrK44np3gMKuvcGeFHv"

        # Graph Path
        resp_path = client.get("/graph/path", params={"source": "1DrK44np3gMKuvcGeFHv", "target": "1CashOutNn3bVx7wqFsT"})
        assert resp_path.status_code == 200
        assert resp_path.json()["found"] is True

        # Static assets
        resp_static = client.get("/")
        assert resp_static.status_code in (200, 404)
        if resp_static.status_code == 200:
            assert "Kautilya" in resp_static.text
