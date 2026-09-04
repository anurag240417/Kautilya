"""Automated end-to-end test suite for 'Operation Shadow Mixer' Investigation Scenario.

Validates the full investigator workflow defined in docs/INVESTIGATION_SCENARIO.md:
1. Alert triage & intake (Critical priority alert on TX-1001).
2. Evidence Trail decomposition & forensic explainability checks.
3. Link analysis ego-subgraph & 4-hop money flow path from source to cash-out.
4. Network layer cross-correlation (same candidate originator IP across TX-1001 and TX-1006).
5. Entity-level aggregation & non-accusation principle on mixer wallet.
6. Alert triage status transitions (NEW -> IN_REVIEW -> ESCALATED).
"""

import pytest

from backend.api import create_app, InvestigationService, TestClient, set_investigation_service
from backend.api.demo_data import (
    TX_1001,
    TX_1004,
    TX_1006,
    WALLET_A,
    WALLET_B,
    WALLET_C,
    WALLET_D,
)
from backend.domain.types import AlertStatus, ExplanationType, PriorityTier


@pytest.fixture(scope="module")
def client():
    """Setup client backed by service seeded with Operation Shadow Mixer dataset."""
    service = InvestigationService()
    service.seed_sample_data()
    set_investigation_service(service)
    app = create_app()
    return TestClient(app)


class TestInvestigationScenarioE2E:
    """Execute each phase of the Operation Shadow Mixer scenario playbook."""

    def test_phase1_alert_triage_intake(self, client):
        """Phase 1: Alert triage on overview dashboard."""
        resp = client.get("/alerts", params={"min_tier": "critical"})
        assert resp.status_code == 200
        data = resp.json()

        assert data["total"] >= 2
        assert len(data["alerts"]) >= 2

        # First alert must be TX-1001 with Critical priority
        top_alert = data["alerts"][0]
        assert top_alert["entity_id"] == str(TX_1001)
        assert top_alert["priority_tier"] == PriorityTier.CRITICAL
        assert top_alert["risk_score"] >= 90.0
        assert "behavioral" in top_alert["active_signals"]
        assert "correlation" in top_alert["active_signals"]
        assert top_alert["contains_synthetic_input"] is True
        assert top_alert["status"] == AlertStatus.NEW

    def test_phase2_evidence_trail_reveal(self, client):
        """Phase 2: Evidence trail decomposition on investigation page."""
        resp = client.get(f"/transactions/{TX_1001}")
        assert resp.status_code == 200
        data = resp.json()

        # Risk score synthesis
        risk = data["risk_score"]
        assert risk["score"] == 92.5
        assert risk["priority_tier"] == "critical"
        assert len(risk["active_signals"]) == 4

        # Evidence records
        ledger = risk["evidence_ledger"]
        assert ledger is not None
        records = ledger["records"]
        assert len(records) == 4

        categories = {r["category"] for r in records}
        assert "ml_behavioral" in categories
        assert "graph_structural" in categories
        assert "anomaly" in categories
        assert "synthetic_network" in categories

        # Forensic explainability rules
        for r in records:
            # Rule: No anonymized feature name leakage
            assert "Local_feature_" not in r["headline"]
            assert "Local_feature_" not in r["description"]
            assert "Aggregate_feature_" not in r["headline"]
            assert "Aggregate_feature_" not in r["description"]

        # Explanation type separation
        ml_rec = next(r for r in records if r["category"] == "ml_behavioral")
        assert ml_rec["explanation_type"] == ExplanationType.CLASSIFIER_EXPLANATION
        assert ml_rec["supporting_metrics"]["illicit_probability"] == 0.94

        anom_rec = next(r for r in records if r["category"] == "anomaly")
        assert anom_rec["explanation_type"] == ExplanationType.ANOMALY_EXPLANATION
        assert "statistical observation" in anom_rec["description"].lower()

    def test_phase3_graph_link_analysis_and_path(self, client):
        """Phase 3: Link analysis ego-subgraph and 4-hop money flow path."""
        # Ego subgraph around TX-1001
        resp_ego = client.get(f"/graph/{TX_1001}")
        assert resp_ego.status_code == 200
        ego_data = resp_ego.json()
        ego_node_ids = {n["id"] for n in ego_data["nodes"]}
        assert str(TX_1001) in ego_node_ids
        assert WALLET_A in ego_node_ids
        assert WALLET_B in ego_node_ids

        # Path query 1: Source wallet -> Layering hop (through mixer)
        resp_layer = client.get("/graph/path", params={"source": WALLET_A, "target": WALLET_C})
        assert resp_layer.status_code == 200
        layer_path = resp_layer.json()
        assert layer_path["found"] is True
        assert layer_path["path_nodes"] == [WALLET_A, WALLET_B, WALLET_C]

        # Path query 2: Source wallet -> Cash-out wallet (via bypass)
        resp_path = client.get("/graph/path", params={"source": WALLET_A, "target": WALLET_D})
        assert resp_path.status_code == 200
        path_data = resp_path.json()
        assert path_data["found"] is True
        assert path_data["path_nodes"][0] == WALLET_A
        assert path_data["path_nodes"][-1] == WALLET_D
        assert WALLET_B in path_data["path_nodes"]

        # Path query 3: Transaction 1001 -> Consolidation Transaction 1004
        resp_tx_path = client.get("/graph/path", params={"source": "1001", "target": "1004"})
        assert resp_tx_path.status_code == 200
        tx_path = resp_tx_path.json()
        assert tx_path["found"] is True
        assert tx_path["path_nodes"][0] == "1001"
        assert tx_path["path_nodes"][-1] == "1004"

    def test_phase4_network_cross_correlation(self, client):
        """Phase 4: Network layer cross-correlation across TX-1001 and TX-1006."""
        # Query TX-1001
        resp1 = client.get(f"/transactions/{TX_1001}")
        corr1 = resp1.json()["correlations"][0]
        assert corr1["candidate_ip"] == "198.51.100.45"
        assert corr1["is_synthetic"] is True

        # Query TX-1006
        resp2 = client.get(f"/transactions/{TX_1006}")
        corr2 = resp2.json()["correlations"][0]
        assert corr2["candidate_ip"] == "198.51.100.45"
        assert corr2["is_synthetic"] is True

        # Independent transactions share the same candidate originator IP
        assert corr1["candidate_ip"] == corr2["candidate_ip"]
        assert corr1["asn"] == corr2["asn"]
        assert corr1["country"] == corr2["country"]

    def test_phase5_entity_level_aggregation_non_accusation(self, client):
        """Phase 5: Mixer wallet entity profile preserves non-accusation principle."""
        resp = client.get(f"/wallets/{WALLET_B}")
        assert resp.status_code == 200
        data = resp.json()

        assert data["address"] == WALLET_B
        assert data["stats"]["total_txs"] == 80.0

        # Mandatory non-accusation disclaimer
        disclaimer = data["forensic_disclaimer"]
        assert "Individual transaction predictions" in disclaimer
        assert "investigative prioritization" in disclaimer

        # Aggregation traceability
        agg = data["aggregation"]
        assert agg is not None
        assert agg["aggregation_method"] == "max"
        assert "contributing_transactions" in agg
        assert len(agg["contributing_transactions"]) >= 1

    def test_phase6_alert_status_lifecycle(self, client):
        """Phase 6: Update alert lifecycle status during investigation."""
        # Find alert for TX-1001
        resp = client.get("/alerts")
        alerts = resp.json()["alerts"]
        target_alert = next(a for a in alerts if a["entity_id"] == str(TX_1001))
        alert_id = target_alert["alert_id"]

        # Move to IN_REVIEW
        resp_patch1 = client.patch(f"/alerts/{alert_id}", json={"status": "in_review"})
        assert resp_patch1.status_code == 200
        assert resp_patch1.json()["status"] == AlertStatus.IN_REVIEW

        # Escalate
        resp_patch2 = client.patch(f"/alerts/{alert_id}", json={"status": "escalated"})
        assert resp_patch2.status_code == 200
        assert resp_patch2.json()["status"] == AlertStatus.ESCALATED

        # Verify status in list
        resp_get = client.get(f"/alerts/{alert_id}")
        assert resp_get.status_code == 200
        assert resp_get.json()["status"] == AlertStatus.ESCALATED
