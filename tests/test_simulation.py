"""Tests for live transaction simulation and streaming engine with 110 transactions."""

import pytest
from backend.api import create_app, TestClient
from backend.api.service import InvestigationService, set_investigation_service
from backend.api.simulation import get_simulation_engine


@pytest.fixture
def test_client():
    service = InvestigationService()
    service.seed_expanded_data()
    set_investigation_service(service)
    app = create_app()
    return TestClient(app)


def test_simulation_status(test_client):
    resp = test_client.get("/simulation/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "current_step" in data
    assert "total_steps" in data
    assert data["total_steps"] == 110
    assert data["total_transactions"] == 110
    assert data["block_height"] == 854230
    assert data["node_status"] == "SYNCED (LOCAL REPLICA)"


def test_simulation_reset_baseline(test_client):
    resp = test_client.post("/simulation/reset", json={"mode": "baseline"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["current_step"] == 1
    assert data["total_transactions"] == 1
    assert data["total_alerts"] == 0
    assert data["can_inject_next"] is True
    assert data["next_txid"] == 1001


def test_simulation_step_by_step_injection(test_client):
    # Reset to baseline
    test_client.post("/simulation/reset", json={"mode": "baseline"})

    # Step 1: TX 1001 (Darknet Influx)
    r1 = test_client.post("/simulation/inject")
    assert r1.status_code == 200
    d1 = r1.json()
    assert d1["success"] is True
    assert d1["injected"]["txid"] == 1001
    assert d1["injected"]["amount_btc"] == 14.5
    assert d1["status"]["current_step"] == 2
    assert d1["status"]["total_transactions"] == 2
    assert d1["status"]["total_alerts"] >= 1

    # Step 2: TX 1002 (Mixer Split A)
    r2 = test_client.post("/simulation/inject")
    assert r2.status_code == 200
    d2 = r2.json()
    assert d2["injected"]["txid"] == 1002
    assert d2["status"]["current_step"] == 3
    assert d2["status"]["total_transactions"] == 3

    # Step 3: TX 1003 (Mixer Split B)
    r3 = test_client.post("/simulation/inject")
    assert r3.status_code == 200
    d3 = r3.json()
    assert d3["injected"]["txid"] == 1003
    assert d3["status"]["current_step"] == 4

    # Step 4: TX 1004 (Consolidation & Cash-out)
    r4 = test_client.post("/simulation/inject")
    assert r4.status_code == 200
    d4 = r4.json()
    assert d4["injected"]["txid"] == 1004
    assert d4["status"]["current_step"] == 5

    # Step 5: TX 1006 (Corroborating Influx)
    r5 = test_client.post("/simulation/inject")
    assert r5.status_code == 200
    d5 = r5.json()
    assert d5["injected"]["txid"] == 1006
    assert d5["status"]["current_step"] == 6

    # Step 6: TX 1007 (Rapid Churn Liquidation)
    r6 = test_client.post("/simulation/inject")
    assert r6.status_code == 200
    d6 = r6.json()
    assert d6["injected"]["txid"] == 1007
    assert d6["status"]["current_step"] == 7
    assert d6["status"]["total_transactions"] == 7


def test_simulation_reset_full(test_client):
    test_client.post("/simulation/reset", json={"mode": "baseline"})
    resp = test_client.post("/simulation/reset", json={"mode": "full"})
    assert resp.status_code == 200
    data = resp.json()
    assert data["current_step"] == 110
    assert data["total_transactions"] == 110
    assert data["total_alerts"] >= 20
    assert data["can_inject_next"] is False


def test_simulation_batch_injection(test_client):
    test_client.post("/simulation/reset", json={"mode": "baseline"})
    resp = test_client.post("/simulation/inject?batch=10")
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is True
    assert data["batch_count"] == 10
    assert data["status"]["total_transactions"] == 11
    assert data["status"]["can_inject_next"] is True
