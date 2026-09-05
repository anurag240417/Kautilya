"""Unit and integration tests for ChainTrace Investigation API.

Tests all endpoints:
- GET /health
- GET /transactions/{txid}
- GET /wallets/{address}
- GET /alerts
- GET /alerts/{alert_id}
- PATCH /alerts/{alert_id}
- GET /graph/{entity_id}
- GET /graph/path
- CORS, pagination, error envelopes, and forensic principles
"""

import pytest

from backend.api import (
    InvestigationService,
    TestClient,
    create_app,
    set_investigation_service,
)
from backend.domain.alert import InvestigativeAlert
from backend.domain.graph import AddrAddrEdge, AddrTxEdge, TxAddrEdge, TxTxEdge
from backend.domain.risk import EvidenceLedger, RiskScore
from backend.domain.transaction import Transaction
from backend.domain.types import EntityClass, PriorityTier
from backend.domain.wallet import StatsSummary, Wallet
from backend.graph.builder import ChainTraceGraph
from backend.risk.ranking import generate_alert


@pytest.fixture(autouse=True)
def setup_api_service():
    """Build a clean, deterministic InvestigationService fixture for tests."""
    graph = ChainTraceGraph()

    # 1. Transactions
    tx1 = Transaction(
        txid=1001,
        time_step=10,
        label=EntityClass.ILLICIT,
        total_btc=5.0,
        fees=0.001,
        size=250.0,
        num_input_addresses=1.0,
        num_output_addresses=2.0,
        in_txs_degree=1.0,
        out_txs_degree=2.0,
        in_btc_total=5.001,
        out_btc_total=5.0,
    )
    tx2 = Transaction(
        txid=1002,
        time_step=10,
        label=EntityClass.UNKNOWN,
        total_btc=4.999,
        fees=0.0005,
        size=220.0,
        num_input_addresses=1.0,
        num_output_addresses=1.0,
    )
    tx_isolated = Transaction(
        txid=9999,
        time_step=12,
        label=EntityClass.LICIT,
        total_btc=0.2,
        fees=0.00005,
    )

    transactions = {tx1.txid: tx1, tx2.txid: tx2, tx_isolated.txid: tx_isolated}

    # 2. Wallets
    w1 = Wallet(
        address="addr_alpha_111",
        time_step=10,
        label=EntityClass.ILLICIT,
        num_txs_as_sender=10.0,
        num_txs_as_receiver=2.0,
        total_txs=12.0,
        lifetime_in_blocks=500.0,
        btc_transacted=StatsSummary(total=20.0, min=0.1, max=5.0, mean=1.66, median=1.0),
        btc_sent=StatsSummary(total=18.0, min=0.1, max=5.0, mean=1.8, median=1.0),
        btc_received=StatsSummary(total=2.0, min=0.5, max=1.5, mean=1.0, median=1.0),
    )
    w2 = Wallet(
        address="addr_beta_222",
        time_step=10,
        label=EntityClass.UNKNOWN,
        num_txs_as_sender=1.0,
        num_txs_as_receiver=3.0,
        total_txs=4.0,
        btc_transacted=StatsSummary(total=5.0, min=1.0, max=4.999, mean=2.5, median=2.5),
    )
    wallets = {w1.address: w1, w2.address: w2}

    # 3. Edges
    graph.add_addr_tx_edges([AddrTxEdge(input_address=w1.address, txid=tx1.txid)])
    graph.add_tx_tx_edges([TxTxEdge(source_txid=tx1.txid, target_txid=tx2.txid)])
    graph.add_tx_addr_edges([TxAddrEdge(txid=tx2.txid, output_address=w2.address)])
    graph.add_addr_addr_edges([
        AddrAddrEdge(input_address=w1.address, output_address=w2.address)
    ])
    # Isolated node in graph
    graph.G.add_node("addr_isolated_999", type="wallet")

    # 4. Risk Scores
    rs1 = RiskScore(
        entity_id=str(tx1.txid),
        entity_type="transaction",
        score=90.0,
        priority_tier=PriorityTier.CRITICAL,
        tier_description="Critical Priority: Illicit corroboration",
        behavioral_signal=0.92,
        graph_signal=0.85,
        anomaly_signal=0.75,
        active_signals=["behavioral", "graph", "anomaly"],
        explanation="Illicit transfer flagged by supervised classifier.",
        evidence_ledger=EvidenceLedger(
            entity_id=str(tx1.txid),
            entity_type="transaction",
        ),
        contains_synthetic_input=True,
    )
    rs2 = RiskScore(
        entity_id=str(tx2.txid),
        entity_type="transaction",
        score=45.0,
        priority_tier=PriorityTier.MEDIUM,
        tier_description="Medium Priority",
        behavioral_signal=0.35,
        anomaly_signal=0.55,
        active_signals=["anomaly"],
        explanation="Unusual transaction size.",
        contains_synthetic_input=False,
    )
    risk_scores = {str(tx1.txid): rs1, str(tx2.txid): rs2}

    # 5. Correlations
    correlations = {
        tx1.txid: [
            {
                "candidate_ip": "198.51.100.1",
                "role": "originator",
                "correlation_confidence": 0.88,
                "timestamp": "2024-02-10T12:00:00Z",
                "asn": 15169,
                "country": "US",
                "script_type": "P2PKH",
                "is_synthetic": True,
            }
        ]
    }

    # 6. Alerts
    alert1 = generate_alert(rs1, evidence_ledger=rs1.evidence_ledger)
    alert2 = generate_alert(rs2)
    alerts: dict[str, InvestigativeAlert] = {}
    if alert1:
        alerts[alert1.alert_id] = alert1
    if alert2:
        alerts[alert2.alert_id] = alert2

    service = InvestigationService(
        graph=graph,
        transactions=transactions,
        wallets=wallets,
        risk_scores=risk_scores,
        alerts=alerts,
        correlations=correlations,
    )
    set_investigation_service(service)
    yield
    set_investigation_service(None)


@pytest.fixture
def client():
    """Create a TestClient with the configured application."""
    app = create_app()
    return TestClient(app)


class TestHealthEndpoint:
    """Verify system health check and CORS preflight."""

    def test_health_check(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert data["version"] == "0.1.0"
        assert resp.headers["Access-Control-Allow-Origin"] == "*"

    def test_cors_preflight(self, client):
        resp = client.options("/health")
        assert resp.status_code == 204
        assert resp.headers["Access-Control-Allow-Origin"] == "*"
        assert "GET, POST, PATCH, OPTIONS" in resp.headers["Access-Control-Allow-Methods"]


class TestTransactionEndpoint:
    """Verify GET /transactions/{txid}."""

    def test_get_valid_transaction(self, client):
        resp = client.get("/transactions/1001")
        assert resp.status_code == 200
        data = resp.json()
        assert data["txid"] == 1001
        assert data["time_step"] == 10
        assert data["label"] == EntityClass.ILLICIT
        assert data["interpretable_features"]["total_btc"] == 5.0
        assert data["interpretable_features"]["fees"] == 0.001
        assert "Local_feature_1" not in data["interpretable_features"]

        # Risk score and evidence ledger
        assert data["risk_score"] is not None
        assert data["risk_score"]["score"] == 90.0
        assert data["risk_score"]["priority_tier"] == "critical"

        # Correlations
        assert len(data["correlations"]) == 1
        assert data["correlations"][0]["candidate_ip"] == "198.51.100.1"
        assert data["correlations"][0]["is_synthetic"] is True
        assert data["is_synthetic"] is True

        # Connected wallets
        assert "addr_alpha_111" in data["connected_wallets"]["inputs"]

    def test_get_transaction_not_found(self, client):
        resp = client.get("/transactions/888888")
        assert resp.status_code == 404
        data = resp.json()
        assert data["error"]["code"] == "NOT_FOUND"
        assert "888888" in data["error"]["message"]

    def test_get_transaction_invalid_id(self, client):
        resp = client.get("/transactions/not-an-int")
        assert resp.status_code == 400
        data = resp.json()
        assert data["error"]["code"] == "INVALID_TXID"

    def test_get_transaction_negative_id(self, client):
        resp = client.get("/transactions/-99")
        assert resp.status_code == 400
        data = resp.json()
        assert data["error"]["code"] == "INVALID_TXID"


class TestWalletEndpoint:
    """Verify GET /wallets/{address}."""

    def test_get_valid_wallet(self, client):
        resp = client.get("/wallets/addr_alpha_111")
        assert resp.status_code == 200
        data = resp.json()
        assert data["address"] == "addr_alpha_111"
        assert data["time_step"] == 10
        assert data["label"] == EntityClass.ILLICIT
        assert data["stats"]["total_txs"] == 12.0
        assert data["stats"]["btc_transacted"]["total"] == 20.0

        # Aggregation
        assert data["aggregation"] is not None
        assert data["aggregation"]["aggregation_method"] == "max"
        assert data["aggregation"]["aggregated_score"] == 90.0
        assert data["aggregation"]["priority_tier"] == "critical"
        assert len(data["aggregation"]["contributing_transactions"]) >= 1

        # Forensic disclaimer
        assert "Individual transaction predictions" in data["forensic_disclaimer"]
        assert "investigative prioritization" in data["forensic_disclaimer"]

        # Counterparties
        assert "addr_beta_222" in data["counterparties"]

    def test_get_wallet_with_mean_aggregation(self, client):
        resp = client.get("/wallets/addr_alpha_111", params={"aggregation_method": "mean"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["aggregation"]["aggregation_method"] == "mean"

    def test_get_wallet_with_frequency_aggregation(self, client):
        resp = client.get("/wallets/addr_alpha_111", params={"aggregation_method": "frequency"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["aggregation"]["aggregation_method"] == "frequency"

    def test_get_wallet_invalid_aggregation_method(self, client):
        resp = client.get("/wallets/addr_alpha_111", params={"aggregation_method": "bogus_method"})
        assert resp.status_code == 400
        data = resp.json()
        assert data["error"]["code"] == "INVALID_AGGREGATION_METHOD"

    def test_get_wallet_not_found(self, client):
        resp = client.get("/wallets/addr_unknown_999999")
        assert resp.status_code == 404
        data = resp.json()
        assert data["error"]["code"] == "NOT_FOUND"


class TestAlertsEndpoint:
    """Verify GET /alerts, GET /alerts/{alert_id}, and PATCH /alerts/{alert_id}."""

    def test_get_alerts_list(self, client):
        resp = client.get("/alerts")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total"] >= 2
        assert data["filtered_count"] >= 2
        assert len(data["alerts"]) >= 2
        assert "investigative triage prioritization" in data["forensic_disclaimer"]

        # Alerts should be sorted with Critical before Medium
        first_alert = data["alerts"][0]
        assert first_alert["priority_tier"] == "critical"

    def test_filter_alerts_by_min_tier(self, client):
        resp = client.get("/alerts", params={"min_tier": "critical"})
        assert resp.status_code == 200
        data = resp.json()
        for alert in data["alerts"]:
            assert alert["priority_tier"] == "critical"

    def test_filter_alerts_by_status(self, client):
        resp = client.get("/alerts", params={"status": "new"})
        assert resp.status_code == 200
        data = resp.json()
        for alert in data["alerts"]:
            assert alert["status"] == "new"

    def test_filter_alerts_by_min_score(self, client):
        resp = client.get("/alerts", params={"min_score": "80.0"})
        assert resp.status_code == 200
        data = resp.json()
        for alert in data["alerts"]:
            assert alert["risk_score"] >= 80.0

    def test_filter_alerts_synthetic(self, client):
        resp = client.get("/alerts", params={"include_synthetic": "false"})
        assert resp.status_code == 200
        data = resp.json()
        for alert in data["alerts"]:
            assert alert["contains_synthetic_input"] is False

    def test_filter_alerts_pagination(self, client):
        resp = client.get("/alerts", params={"limit": "1", "offset": "0"})
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["alerts"]) == 1

    def test_filter_alerts_invalid_tier(self, client):
        resp = client.get("/alerts", params={"min_tier": "ultra_danger"})
        assert resp.status_code == 400
        data = resp.json()
        assert data["error"]["code"] == "INVALID_FILTER"

    def test_filter_alerts_invalid_status(self, client):
        resp = client.get("/alerts", params={"status": "deleted"})
        assert resp.status_code == 400
        data = resp.json()
        assert data["error"]["code"] == "INVALID_FILTER"

    def test_filter_alerts_invalid_min_score(self, client):
        resp = client.get("/alerts", params={"min_score": "150.0"})
        assert resp.status_code == 400
        data = resp.json()
        assert data["error"]["code"] == "INVALID_FILTER"

    def test_filter_alerts_invalid_limit(self, client):
        resp = client.get("/alerts", params={"limit": "-1"})
        assert resp.status_code == 400
        data = resp.json()
        assert data["error"]["code"] == "INVALID_PAGINATION"

    def test_get_alert_by_id(self, client):
        # Fetch alert ID from list
        list_resp = client.get("/alerts")
        alert_id = list_resp.json()["alerts"][0]["alert_id"]

        resp = client.get(f"/alerts/{alert_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["alert_id"] == alert_id
        assert data["risk_score"] > 0

    def test_get_alert_by_id_not_found(self, client):
        resp = client.get("/alerts/alt-nonexistent-1234")
        assert resp.status_code == 404
        data = resp.json()
        assert data["error"]["code"] == "NOT_FOUND"

    def test_patch_alert_status(self, client):
        list_resp = client.get("/alerts")
        alert_id = list_resp.json()["alerts"][0]["alert_id"]

        # Update status to in_review
        patch_resp = client.patch(
            f"/alerts/{alert_id}",
            json={"status": "in_review", "reviewer_notes": "Forensic analyst assigned"},
        )
        assert patch_resp.status_code == 200
        updated = patch_resp.json()
        assert updated["status"] == "in_review"
        assert updated["metadata"]["reviewer_notes"] == "Forensic analyst assigned"

        # Check subsequent GET reflects status
        get_resp = client.get(f"/alerts/{alert_id}")
        assert get_resp.json()["status"] == "in_review"

        # Update with uppercase status (e.g. IN_REVIEW, ESCALATED)
        patch_upper = client.patch(
            f"/alerts/{alert_id}",
            json={"status": "ESCALATED"},
        )
        assert patch_upper.status_code == 200
        assert patch_upper.json()["status"] == "escalated"

        # Update with CLOSED alias (maps to dismissed)
        patch_closed = client.patch(
            f"/alerts/{alert_id}",
            json={"status": "CLOSED"},
        )
        assert patch_closed.status_code == 200
        assert patch_closed.json()["status"] == "dismissed"

    def test_patch_alert_invalid_status(self, client):
        list_resp = client.get("/alerts")
        alert_id = list_resp.json()["alerts"][0]["alert_id"]

        patch_resp = client.patch(
            f"/alerts/{alert_id}",
            json={"status": "bogus_status"},
        )
        assert patch_resp.status_code == 400
        assert patch_resp.json()["error"]["code"] == "VALIDATION_ERROR"

    def test_patch_alert_not_found(self, client):
        patch_resp = client.patch(
            "/alerts/alt-missing-9999",
            json={"status": "triaged"},
        )
        assert patch_resp.status_code == 404
        assert patch_resp.json()["error"]["code"] == "NOT_FOUND"


class TestGraphEndpoint:
    """Verify GET /graph/{entity_id} and GET /graph/path."""

    def test_get_subgraph_by_transaction(self, client):
        resp = client.get("/graph/1001", params={"depth": "1"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["entity_id"] == "1001"
        assert data["entity_type"] == "transaction"
        assert data["depth"] == 1
        assert data["node_count"] >= 2
        assert data["edge_count"] >= 1

        node_ids = [n["id"] for n in data["nodes"]]
        assert "1001" in node_ids
        assert "1002" in node_ids or "addr_alpha_111" in node_ids

    def test_get_subgraph_by_wallet(self, client):
        resp = client.get("/graph/addr_alpha_111", params={"depth": "1"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["entity_id"] == "addr_alpha_111"
        assert data["entity_type"] == "wallet"
        assert data["node_count"] >= 2

    def test_get_subgraph_relationship_filter(self, client):
        resp = client.get(
            "/graph/1001",
            params={"depth": "1", "relationship": "tx_tx"},
        )
        assert resp.status_code == 200
        data = resp.json()
        for edge in data["edges"]:
            assert edge["relationship"] == "tx_tx"

    def test_get_subgraph_invalid_depth(self, client):
        resp = client.get("/graph/1001", params={"depth": "10"})
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "INVALID_DEPTH"

    def test_get_subgraph_invalid_relationship(self, client):
        resp = client.get("/graph/1001", params={"relationship": "invalid_rel"})
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "INVALID_RELATIONSHIP"

    def test_get_subgraph_not_found(self, client):
        resp = client.get("/graph/addr_nonexistent_9999")
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "NOT_FOUND"

    def test_get_graph_path_success(self, client):
        resp = client.get("/graph/path", params={"source": "addr_alpha_111", "target": "1002"})
        assert resp.status_code == 200
        data = resp.json()
        assert data["source"] == "addr_alpha_111"
        assert data["target"] == "1002"
        assert data["found"] is True
        assert data["path_length"] == 2
        assert data["path_nodes"] == ["addr_alpha_111", "1001", "1002"]
        assert len(data["path_edges"]) == 2

    def test_get_graph_path_missing_params(self, client):
        resp = client.get("/graph/path", params={"source": "addr_alpha_111"})
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "MISSING_QUERY_PARAMS"

    def test_get_graph_path_nonexistent_node(self, client):
        resp = client.get(
            "/graph/path",
            params={"source": "addr_alpha_111", "target": "missing_node"},
        )
        assert resp.status_code == 404
        assert resp.json()["error"]["code"] == "NOT_FOUND"

    def test_get_graph_path_no_path_found(self, client):
        resp = client.get(
            "/graph/path",
            params={"source": "addr_alpha_111", "target": "addr_isolated_999"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["found"] is False
        assert data["path_length"] is None
        assert data["path_nodes"] == []


class TestAPIGeneralBehavior:
    """Verify generic routing, 404, 405, and error handling behaviors."""

    def test_unmatched_route_returns_404(self, client):
        resp = client.get("/unmatched/path")
        assert resp.status_code == 404
        data = resp.json()
        assert data["error"]["code"] == "NOT_FOUND"
        assert "Path not found" in data["error"]["message"]

    def test_method_not_allowed_returns_405(self, client):
        resp = client.post("/transactions/1001")
        assert resp.status_code == 405
        data = resp.json()
        assert data["error"]["code"] == "METHOD_NOT_ALLOWED"
        assert "POST" in data["error"]["message"]

    def test_error_envelope_consistency(self, client):
        resp = client.get("/transactions/invalid-id")
        data = resp.json()
        assert "error" in data
        assert "code" in data["error"]
        assert "message" in data["error"]
        assert resp.headers["Access-Control-Allow-Origin"] == "*"


class TestStatisticsEndpoint:
    """Tests for GET /statistics dashboard aggregate counts."""

    def test_get_statistics_success(self, client):
        resp = client.get("/statistics")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_transactions" in data
        assert "total_wallets" in data
        assert "total_alerts" in data
        assert "tier_counts" in data
        assert "status_counts" in data
        assert "synthetic_alerts_count" in data
        assert "synthetic_alerts_percentage" in data
        assert data["is_offline_mode"] is True
        assert data["total_transactions"] >= 2
        assert data["total_wallets"] >= 2
        assert data["total_alerts"] >= 2


class TestStaticFrontendServing:
    """Tests for offline static SPA distribution serving."""

    def test_serves_index_html_for_root(self, client):
        resp = client.get("/")
        if resp.status_code == 200:
            assert "ChainTrace" in resp.text
            assert "text/html" in resp.headers.get("Content-Type", "")

    def test_custom_static_dir(self, tmp_path):
        html_file = tmp_path / "index.html"
        html_file.write_text("<!DOCTYPE html><html><body>ChainTrace Offline Test</body></html>")
        from backend.api.app import ChainTraceAPI
        app = ChainTraceAPI(title="Test", static_dir=tmp_path)
        test_client = TestClient(app)
        resp = test_client.get("/")
        assert resp.status_code == 200
        assert "ChainTrace Offline Test" in resp.text


