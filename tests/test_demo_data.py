"""Tests for Phase 10 Step 1 — Operation Shadow Mixer Demonstration Dataset.

Verifies:
1. Entity integrity (7 transactions, 5 wallets, full label coverage).
2. Multi-signal risk scores across all 4 priority tiers (Critical, High, Medium, Low).
3. Structured Evidence Ledgers with both classifier and anomaly explanations.
4. Compliance with forensic explainability rules (no anonymized Elliptic feature names).
5. Graph topology and money-flow path from source wallet to cash-out wallet.
6. Synthetic network provenance flags and multi-transaction IP corroboration.
7. Default InvestigationService seeding and API endpoints integration.
"""

import pytest

from backend.api import create_app, InvestigationService, TestClient, set_investigation_service
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
from backend.domain.types import EntityClass, ExplanationType, PriorityTier
from backend.graph.paths import find_shortest_path


@pytest.fixture
def demo():
    """Return fresh build of demo dataset."""
    return build_demo_dataset()


class TestDemoDatasetIntegrity:
    """Validate completeness and schema integrity of the demo dataset."""

    def test_transactions_coverage(self, demo):
        """Must have 7 transactions spanning Illicit, Unknown, and Licit."""
        assert len(demo.transactions) == 7
        tx_ids = set(demo.transactions.keys())
        expected_ids = {TX_1001, TX_1002, TX_1003, TX_1004, TX_1005, TX_1006, TX_1007}
        assert tx_ids == expected_ids

        labels = {tx.label for tx in demo.transactions.values()}
        assert EntityClass.ILLICIT in labels
        assert EntityClass.UNKNOWN in labels
        assert EntityClass.LICIT in labels

        # TX-1001 and TX-1006 must be Illicit
        assert demo.transactions[TX_1001].label == EntityClass.ILLICIT
        assert demo.transactions[TX_1006].label == EntityClass.ILLICIT
        # TX-1005 is control Licit
        assert demo.transactions[TX_1005].label == EntityClass.LICIT

    def test_wallets_coverage(self, demo):
        """Must have 5 wallets covering source, mixer, layer, cashout, and control."""
        assert len(demo.wallets) == 5
        addrs = set(demo.wallets.keys())
        expected_addrs = {WALLET_A, WALLET_B, WALLET_C, WALLET_D, WALLET_E}
        assert addrs == expected_addrs

        # WALLET_B is high-volume mixer
        mixer = demo.wallets[WALLET_B]
        assert mixer.total_txs >= 50
        assert mixer.btc_transacted.total > 100.0

    def test_risk_score_tiers(self, demo):
        """Must include entities across all priority tiers: Critical, High, Medium, Low."""
        tiers = {rs.priority_tier for rs in demo.risk_scores.values()}
        assert PriorityTier.CRITICAL in tiers
        assert PriorityTier.HIGH in tiers
        assert PriorityTier.MEDIUM in tiers
        assert PriorityTier.LOW in tiers

        # TX-1001 is critical
        assert demo.risk_scores[str(TX_1001)].priority_tier == PriorityTier.CRITICAL
        assert demo.risk_scores[str(TX_1001)].score >= 90.0
        # TX-1005 is low
        assert demo.risk_scores[str(TX_1005)].priority_tier == PriorityTier.LOW
        assert demo.risk_scores[str(TX_1005)].score < 20.0

    def test_synthetic_provenance_flags(self, demo):
        """All demo records must carry explicit synthetic markers."""
        # Risk scores for synthetic scenarios
        for txid in [TX_1001, TX_1002, TX_1003, TX_1004, TX_1006, TX_1007]:
            assert demo.risk_scores[str(txid)].contains_synthetic_input is True

        # Graph edges
        for category, edges in demo.graph_edges.items():
            for edge in edges:
                assert edge.is_synthetic is True
                assert edge.provenance == PROVENANCE

        # Correlations
        for txid, corr_list in demo.correlations.items():
            for c in corr_list:
                assert c["is_synthetic"] is True

    def test_corroborating_network_correlations(self, demo):
        """TX-1001 and TX-1006 share IP 198.51.100.45; TX-1004 and TX-1007 share relay IP."""
        c1001 = demo.correlations[TX_1001][0]
        c1006 = demo.correlations[TX_1006][0]
        assert c1001["candidate_ip"] == "198.51.100.45"
        assert c1006["candidate_ip"] == "198.51.100.45"

        c1004 = demo.correlations[TX_1004][0]
        c1007 = demo.correlations[TX_1007][0]
        assert c1004["candidate_ip"] == "203.0.113.88"
        assert c1007["candidate_ip"] == "203.0.113.88"

    def test_forensic_explainability_rules(self, demo):
        """Forensic rule: No anonymized feature names in descriptions/headlines.

        Distinguish classifier explanations from anomaly explanations.
        """
        for rs in demo.risk_scores.values():
            if rs.evidence_ledger:
                for rec in rs.evidence_ledger.records:
                    # Check forbidden raw column patterns
                    assert "Local_feature_" not in rec.headline
                    assert "Local_feature_" not in rec.description
                    assert "Aggregate_feature_" not in rec.headline
                    assert "Aggregate_feature_" not in rec.description

                    # Check explanation type correctness
                    if rec.explanation_type == ExplanationType.CLASSIFIER_EXPLANATION:
                        assert "classifier" in rec.headline.lower() or "probability" in rec.headline.lower() or "illicit" in rec.headline.lower()
                    elif rec.explanation_type == ExplanationType.ANOMALY_EXPLANATION:
                        assert "deviance" in rec.description.lower() or "anomaly" in rec.description.lower() or "baseline" in rec.description.lower()

    def test_alert_generation(self, demo):
        """Only entities meeting medium threshold produce alerts; low tier excluded."""
        assert len(demo.alerts) >= 4
        alert_entity_ids = {a.entity_id for a in demo.alerts.values()}
        assert str(TX_1001) in alert_entity_ids
        assert str(TX_1006) in alert_entity_ids
        assert str(TX_1004) in alert_entity_ids
        assert str(TX_1005) not in alert_entity_ids  # Low tier, excluded

    def test_graph_path_connectivity(self, demo):
        """Graph must contain a money flow path from WALLET_A (source) to WALLET_D (cash-out)."""
        from backend.graph.builder import KautilyaGraph

        graph = KautilyaGraph()
        graph.add_addr_tx_edges(demo.graph_edges["addr_tx"])
        graph.add_tx_addr_edges(demo.graph_edges["tx_addr"])
        graph.add_tx_tx_edges(demo.graph_edges["tx_tx"])
        graph.add_addr_addr_edges(demo.graph_edges["addr_addr"])

        # Path between wallet addresses
        path = find_shortest_path(graph, WALLET_A, WALLET_D)
        assert path is not None
        assert path[0] == WALLET_A
        assert path[-1] == WALLET_D


class TestInvestigationServiceWithDemoData:
    """Test full API service seeded with demo data."""

    @pytest.fixture(autouse=True)
    def setup_service(self):
        svc = InvestigationService()
        svc.seed_sample_data()
        set_investigation_service(svc)
        self.client = TestClient(create_app())

    def test_get_demo_transaction(self):
        resp = self.client.get(f"/transactions/{TX_1001}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["txid"] == TX_1001
        assert data["interpretable_features"]["total_btc"] == 14.5
        assert data["risk_score"] is not None
        assert data["risk_score"]["priority_tier"] == "critical"
        assert len(data["correlations"]) == 1
        assert data["correlations"][0]["candidate_ip"] == "198.51.100.45"

    def test_get_demo_wallet(self):
        resp = self.client.get(f"/wallets/{WALLET_B}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["address"] == WALLET_B
        assert data["stats"]["total_txs"] == 80.0

    def test_get_demo_alerts(self):
        resp = self.client.get("/alerts")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["alerts"]) >= 4

    def test_get_graph_path(self):
        resp = self.client.get("/graph/path", params={"source": WALLET_A, "target": WALLET_D})
        assert resp.status_code == 200
        data = resp.json()
        assert data["found"] is True
        assert data["path_nodes"][0] == WALLET_A
        assert data["path_nodes"][-1] == WALLET_D

    def test_statistics_reflects_demo_dataset(self):
        resp = self.client.get("/statistics")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_transactions"] == 7
        assert data["total_wallets"] == 5
        assert data["total_alerts"] >= 4
        assert data["tier_counts"]["critical"] >= 2
        assert data["tier_counts"]["high"] >= 2
