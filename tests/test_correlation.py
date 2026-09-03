"""Tests for Phase 6 Correlation Engine.

Verifies:
1. Timestamp normalization (ISO strings, epoch s/ms, datetimes, pandas Timestamps).
2. Deterministic time_step → datetime and reverse mapping.
3. Temporal correlation windows and proximity scoring.
4. Propagation timing consistency across hops.
5. Topology consistency checks with and without NetworkX graph.
6. Transaction-to-IP correlation and candidate role assignment (Origin vs Relay).
7. Decoupled correlation confidence metrics [0.0, 1.0].
8. Strict preservation of `is_synthetic=True` and mandatory forensic disclaimers.
9. Entity resolution linking wallets to candidate IPs.
"""

from datetime import UTC, datetime, timedelta, timezone

import networkx as nx
import pandas as pd
import pytest

from backend.correlation.entity_resolution import (
    resolve_ip_to_candidate_wallets,
    resolve_wallet_to_candidate_ips,
)
from backend.correlation.temporal import (
    compute_propagation_consistency,
    compute_temporal_proximity,
    correlate_temporally,
    get_temporal_window,
    is_in_temporal_window,
)
from backend.correlation.tx_ip import (
    CandidateIPRole,
    compute_confidence_metrics,
    compute_topology_consistency,
    correlate_ip_to_transactions,
    correlate_transaction_to_ips,
)
from backend.domain.correlation import (
    CorrelationConfidenceMetrics,
    TransactionIPCorrelation,
)
from backend.domain.network import NetworkObservation
from backend.domain.transaction import Transaction
from backend.generator.temporal import (
    _EPOCH,
    datetime_to_time_step,
    time_step_to_datetime,
    time_step_to_datetime_range,
)
from backend.normalization.temporal import (
    normalize_time_step,
    normalize_timestamp,
)


class TestTimestampNormalization:
    """Tests for timestamp and time_step normalization."""

    def test_normalize_iso_string_utc(self):
        ts = "2019-01-07T12:30:00Z"
        dt = normalize_timestamp(ts)
        assert dt == datetime(2019, 1, 7, 12, 30, 0, tzinfo=UTC)

    def test_normalize_iso_string_with_offset(self):
        ts = "2019-01-07T14:30:00+02:00"
        dt = normalize_timestamp(ts)
        assert dt == datetime(2019, 1, 7, 12, 30, 0, tzinfo=UTC)

    def test_normalize_space_separated_string(self):
        ts = "2019-01-07 12:30:00"
        dt = normalize_timestamp(ts)
        assert dt == datetime(2019, 1, 7, 12, 30, 0, tzinfo=UTC)

    def test_normalize_epoch_seconds(self):
        # 1546819200 is 2019-01-07 00:00:00 UTC
        dt = normalize_timestamp(1546819200)
        assert dt == datetime(2019, 1, 7, 0, 0, 0, tzinfo=UTC)

    def test_normalize_epoch_milliseconds(self):
        dt = normalize_timestamp(1546819200000)
        assert dt == datetime(2019, 1, 7, 0, 0, 0, tzinfo=UTC)

    def test_normalize_naive_datetime(self):
        naive = datetime(2019, 1, 7, 10, 0, 0)
        dt = normalize_timestamp(naive)
        assert dt.tzinfo == UTC
        assert dt == datetime(2019, 1, 7, 10, 0, 0, tzinfo=UTC)

    def test_normalize_aware_datetime(self):
        est = timezone(timedelta(hours=-5))
        aware = datetime(2019, 1, 7, 5, 0, 0, tzinfo=est)
        dt = normalize_timestamp(aware)
        assert dt.tzinfo == UTC
        assert dt == datetime(2019, 1, 7, 10, 0, 0, tzinfo=UTC)

    def test_normalize_pandas_timestamp(self):
        pts = pd.Timestamp("2019-01-07 15:00:00")
        dt = normalize_timestamp(pts)
        assert dt == datetime(2019, 1, 7, 15, 0, 0, tzinfo=UTC)

    def test_normalize_invalid_type_raises(self):
        with pytest.raises(TypeError):
            normalize_timestamp(None)
        with pytest.raises(TypeError):
            normalize_timestamp(["invalid"])

    def test_normalize_invalid_string_raises(self):
        with pytest.raises(ValueError):
            normalize_timestamp("not-a-timestamp")
        with pytest.raises(ValueError):
            normalize_timestamp("")

    def test_normalize_time_step_valid(self):
        assert normalize_time_step(1) == 1
        assert normalize_time_step(49) == 49
        assert normalize_time_step("25") == 25
        assert normalize_time_step(10.0) == 10

    def test_normalize_time_step_out_of_bounds(self):
        with pytest.raises(ValueError):
            normalize_time_step(0)
        with pytest.raises(ValueError):
            normalize_time_step(50)
        with pytest.raises(ValueError):
            normalize_time_step(-5)

    def test_normalize_time_step_invalid_type(self):
        with pytest.raises(TypeError):
            normalize_time_step(None)
        with pytest.raises(ValueError):
            normalize_time_step("not_a_number")


class TestTemporalMapping:
    """Tests for consumption of deterministic time_step mapping and reverse lookup."""

    def test_time_step_to_datetime_step_1(self):
        start = time_step_to_datetime(1)
        assert start == datetime(2019, 1, 7, 0, 0, 0, tzinfo=UTC)

    def test_time_step_to_datetime_step_49(self):
        start, end = time_step_to_datetime_range(49)
        assert (end - start) == timedelta(days=14)
        # step 49 start = epoch + 48 * 14 days
        expected_start = datetime(2019, 1, 7, 0, 0, 0, tzinfo=UTC) + timedelta(days=48 * 14)
        assert start == expected_start

    def test_datetime_to_time_step_exact_and_within(self):
        # Beginning of step 1
        assert datetime_to_time_step(_EPOCH) == 1
        # Middle of step 1
        assert datetime_to_time_step(_EPOCH + timedelta(days=7)) == 1
        # Start of step 2
        assert datetime_to_time_step(_EPOCH + timedelta(days=14)) == 2
        # Middle of step 49
        step_49_mid = _EPOCH + timedelta(days=48 * 14 + 5)
        assert datetime_to_time_step(step_49_mid) == 49

    def test_datetime_to_time_step_out_of_bounds(self):
        # Prior to epoch
        assert datetime_to_time_step(_EPOCH - timedelta(seconds=1)) is None
        # After step 49
        assert datetime_to_time_step(_EPOCH + timedelta(days=49 * 14)) is None


class TestTemporalCorrelation:
    """Tests for temporal correlation scoring and windows."""

    def test_window_retrieval(self):
        start, end = get_temporal_window(1)
        assert (end - start) == timedelta(days=14)

        buf_start, buf_end = get_temporal_window(1, boundary_tolerance_seconds=3600)
        assert buf_start == start - timedelta(seconds=3600)
        assert buf_end == end + timedelta(seconds=3600)

    def test_is_in_temporal_window(self):
        start, end = time_step_to_datetime_range(1)
        assert is_in_temporal_window(start, 1) is True
        assert is_in_temporal_window(start + timedelta(days=5), 1) is True
        assert is_in_temporal_window(end, 1) is False  # Half-open [start, end)
        assert is_in_temporal_window(start - timedelta(seconds=10), 1) is False

        # With tolerance
        assert (
            is_in_temporal_window(
                start - timedelta(seconds=10), 1, boundary_tolerance_seconds=60
            )
            is True
        )

    def test_temporal_proximity_inside_window(self):
        start, _ = time_step_to_datetime_range(2)
        score = compute_temporal_proximity(start + timedelta(days=1), 2)
        assert score == 1.0

    def test_temporal_proximity_decay_outside_window(self):
        start, end = time_step_to_datetime_range(2)
        # Exactly 24 hours before start (1 half-life with default 24h config)
        score_24h = compute_temporal_proximity(start - timedelta(hours=24), 2)
        assert 0.49 <= score_24h <= 0.51  # Approx 0.5

        # 48 hours before start (2 half-lives)
        score_48h = compute_temporal_proximity(start - timedelta(hours=48), 2)
        assert 0.24 <= score_48h <= 0.26  # Approx 0.25

        # Far away (> 1 year)
        score_far = compute_temporal_proximity(start - timedelta(days=400), 2)
        assert score_far < 1e-4

    def test_correlate_temporally(self):
        tx = Transaction(txid=101, time_step=1)
        start, _ = time_step_to_datetime_range(1)

        obs1 = NetworkObservation(
            txid=101,
            src_ip="10.0.0.1",
            dst_ip="10.0.0.2",
            src_port=8333,
            dst_port=8333,
            timestamp=start + timedelta(days=2),
            is_synthetic=True,
        )
        obs2 = NetworkObservation(
            txid=101,
            src_ip="10.0.0.2",
            dst_ip="10.0.0.3",
            src_port=8333,
            dst_port=8333,
            timestamp=start - timedelta(hours=24),
            is_synthetic=True,
        )

        corrs = correlate_temporally(tx, [obs1, obs2])
        assert len(corrs) == 2

        assert corrs[0].temporal_proximity == 1.0
        assert corrs[0].delta_seconds == 0.0
        assert corrs[0].is_synthetic is True
        assert "ownership" in corrs[0].disclaimer

        assert 0.49 <= corrs[1].temporal_proximity <= 0.51
        assert corrs[1].delta_seconds == 86400.0


class TestPropagationConsistency:
    """Tests for propagation timing consistency evaluation."""

    def test_empty_observations(self):
        assert compute_propagation_consistency([]) == 0.0

    def test_single_observation(self):
        obs = NetworkObservation(
            txid=200,
            src_ip="192.168.1.1",
            dst_ip="192.168.1.2",
            src_port=8333,
            dst_port=8333,
            timestamp=datetime(2019, 1, 10, 0, 0, 0, tzinfo=UTC),
            is_synthetic=True,
        )
        assert compute_propagation_consistency([obs]) == 1.0

    def test_realistic_propagation_wave(self):
        base_t = datetime(2019, 1, 10, 12, 0, 0, tzinfo=UTC)
        obs1 = NetworkObservation(
            txid=200,
            src_ip="10.0.0.1",
            dst_ip="10.0.0.2",
            src_port=8333,
            dst_port=8333,
            timestamp=base_t,
            is_synthetic=True,
        )
        obs2 = NetworkObservation(
            txid=200,
            src_ip="10.0.0.2",
            dst_ip="10.0.0.3",
            src_port=8333,
            dst_port=8333,
            timestamp=base_t + timedelta(milliseconds=150),
            is_synthetic=True,
        )
        obs3 = NetworkObservation(
            txid=200,
            src_ip="10.0.0.3",
            dst_ip="10.0.0.4",
            src_port=8333,
            dst_port=8333,
            timestamp=base_t + timedelta(milliseconds=350),
            is_synthetic=True,
        )

        score = compute_propagation_consistency([obs1, obs2, obs3])
        assert score == 1.0

    def test_excessive_propagation_duration_penalized(self):
        base_t = datetime(2019, 1, 10, 12, 0, 0, tzinfo=UTC)
        obs1 = NetworkObservation(
            txid=200,
            src_ip="10.0.0.1",
            dst_ip="10.0.0.2",
            src_port=8333,
            dst_port=8333,
            timestamp=base_t,
            is_synthetic=True,
        )
        # Second hop occurs 2 hours later
        obs2 = NetworkObservation(
            txid=200,
            src_ip="10.0.0.2",
            dst_ip="10.0.0.3",
            src_port=8333,
            dst_port=8333,
            timestamp=base_t + timedelta(hours=2),
            is_synthetic=True,
        )

        score = compute_propagation_consistency([obs1, obs2])
        assert score < 0.6  # Significantly penalized for taking hours


class TestTopologyConsistency:
    """Tests for topology consistency evaluation."""

    def test_same_src_and_dst(self):
        score = compute_topology_consistency("10.0.0.1", "10.0.0.1")
        assert score == 0.5

    def test_default_heuristic_without_graph(self):
        score = compute_topology_consistency("10.0.0.1", "10.0.0.2")
        assert score == 0.80

    def test_networkx_topology_direct_edge(self):
        g = nx.Graph()
        g.add_edge("10.0.0.1", "10.0.0.2")
        g.add_edge("10.0.0.2", "10.0.0.3")

        # Direct edge: 1.0
        assert compute_topology_consistency("10.0.0.1", "10.0.0.2", topology=g) == 1.0
        # 2-hop edge: 0.75
        assert compute_topology_consistency("10.0.0.1", "10.0.0.3", topology=g) == 0.75

    def test_networkx_topology_disconnected(self):
        g = nx.Graph()
        g.add_node("10.0.0.1")
        g.add_node("10.0.0.99")

        score = compute_topology_consistency("10.0.0.1", "10.0.0.99", topology=g)
        assert score == 0.15


class TestTransactionIPCorrelation:
    """Tests for Transaction-to-IP correlation engine."""

    @pytest.fixture
    def sample_data(self):
        tx = Transaction(txid=5001, time_step=3)
        start, _ = time_step_to_datetime_range(3)

        obs1 = NetworkObservation(
            txid=5001,
            src_ip="198.51.100.1",  # Origin
            dst_ip="198.51.100.2",  # Relay 1
            src_port=8333,
            dst_port=8333,
            timestamp=start + timedelta(days=1),
            asn=13335,
            country="US",
            is_synthetic=True,
        )
        obs2 = NetworkObservation(
            txid=5001,
            src_ip="198.51.100.2",  # Relay 1 forwarding
            dst_ip="198.51.100.3",  # Relay 2
            src_port=8333,
            dst_port=8333,
            timestamp=start + timedelta(days=1, milliseconds=120),
            asn=15169,
            country="DE",
            is_synthetic=True,
        )
        return tx, [obs1, obs2]

    def test_origin_and_relay_role_detection(self, sample_data):
        tx, observations = sample_data
        corrs = correlate_transaction_to_ips(tx, observations)

        assert len(corrs) >= 2
        # Origin IP should be 198.51.100.1 (earliest src_ip)
        origin_corr = corrs[0]
        assert origin_corr.ip == "198.51.100.1"
        assert origin_corr.role == CandidateIPRole.ORIGIN
        assert origin_corr.is_synthetic is True
        assert 0.0 <= origin_corr.correlation_confidence <= 1.0

        # Disclaimers present
        assert "ownership" in origin_corr.disclaimer

        # Decomposed metrics
        metrics = origin_corr.metrics
        assert isinstance(metrics, CorrelationConfidenceMetrics)
        assert metrics.temporal_proximity == 1.0
        assert metrics.propagation_consistency == 1.0
        assert metrics.topology_consistency == 0.80  # heuristic default
        assert 0.90 <= origin_corr.correlation_confidence <= 1.0

    def test_unrelated_transaction_filtered(self, sample_data):
        tx, observations = sample_data
        unrelated_tx = Transaction(txid=9999, time_step=3)
        corrs = correlate_transaction_to_ips(unrelated_tx, observations)
        assert corrs == []

    def test_correlate_ip_to_transactions(self, sample_data):
        tx, observations = sample_data
        corrs = correlate_ip_to_transactions("198.51.100.1", [tx], observations)
        assert len(corrs) == 1
        assert corrs[0].txid == 5001
        assert corrs[0].role == CandidateIPRole.ORIGIN


class TestEntityResolution:
    """Tests for entity resolution and wallet-to-IP candidate linking."""

    def test_resolve_wallet_to_candidate_ips(self):
        # Setup mock tx-ip correlations
        ts = datetime(2019, 1, 8, 12, 0, 0, tzinfo=UTC)
        metrics = CorrelationConfidenceMetrics(
            temporal_proximity=1.0,
            propagation_consistency=1.0,
            topology_consistency=0.8,
            composite_confidence=0.95,
        )
        c1 = TransactionIPCorrelation(
            txid=1001,
            ip="198.51.100.1",
            role=CandidateIPRole.ORIGIN,
            correlation_confidence=0.95,
            metrics=metrics,
            observations_count=1,
            earliest_observation=ts,
            latest_observation=ts,
            is_synthetic=True,
        )
        c2 = TransactionIPCorrelation(
            txid=1002,
            ip="198.51.100.1",
            role=CandidateIPRole.ORIGIN,
            correlation_confidence=0.90,
            metrics=metrics,
            observations_count=1,
            earliest_observation=ts,
            latest_observation=ts,
            is_synthetic=True,
        )

        wallet_tx_map = {"addr_alice": [1001, 1002]}
        wallet_corrs = resolve_wallet_to_candidate_ips("addr_alice", [c1, c2], wallet_tx_map)

        assert len(wallet_corrs) == 1
        w_corr = wallet_corrs[0]
        assert w_corr.address == "addr_alice"
        assert w_corr.ip == "198.51.100.1"
        assert w_corr.associated_txids == [1001, 1002]
        assert w_corr.candidate_roles == [CandidateIPRole.ORIGIN]
        assert 0.90 <= w_corr.correlation_confidence <= 0.95
        assert w_corr.is_synthetic is True
        assert "ownership" in w_corr.disclaimer

    def test_resolve_ip_to_candidate_wallets(self):
        ts = datetime(2019, 1, 8, 12, 0, 0, tzinfo=UTC)
        metrics = CorrelationConfidenceMetrics(
            temporal_proximity=1.0,
            propagation_consistency=1.0,
            topology_consistency=0.8,
            composite_confidence=0.95,
        )
        c1 = TransactionIPCorrelation(
            txid=2001,
            ip="192.0.2.55",
            role=CandidateIPRole.RELAY,
            correlation_confidence=0.85,
            metrics=metrics,
            observations_count=1,
            earliest_observation=ts,
            latest_observation=ts,
            is_synthetic=True,
        )
        tx_wallet_map = {2001: ["addr_bob", "addr_carol"]}

        ip_wallets = resolve_ip_to_candidate_wallets("192.0.2.55", [c1], tx_wallet_map)
        assert len(ip_wallets) == 2
        addresses = {w.address for w in ip_wallets}
        assert addresses == {"addr_bob", "addr_carol"}
        for w in ip_wallets:
            assert w.is_synthetic is True
            assert "ownership" in w.disclaimer


class TestForensicDecoupling:
    """Forensic integrity tests verifying clear separation of signals."""

    def test_correlation_confidence_decoupled_from_risk(self):
        """Verify correlation records do not contain risk score or illicit probability."""
        metrics = compute_confidence_metrics(
            temporal_proximity=1.0,
            propagation_consistency=1.0,
            topology_consistency=1.0,
        )
        # Even with perfect correlation confidence (1.0)
        assert metrics.composite_confidence == 1.0

        # Check fields of TransactionIPCorrelation
        fields = TransactionIPCorrelation.model_fields
        # Correlation confidence exists
        assert "correlation_confidence" in fields
        # Risk score and illicit probability MUST NOT be present
        assert "risk_score" not in fields
        assert "final_risk_score" not in fields
        assert "illicit_probability" not in fields
        assert "predicted_label" not in fields

        # Synthetic flag MUST be present and default to True
        assert fields["is_synthetic"].default is True
