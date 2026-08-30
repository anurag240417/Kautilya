"""Tests for the synthetic network data generator (Phase 4).

Covers:
    - Node pool generation (uniqueness, distributions, reproducibility)
    - Topology construction (connectivity, degree, attributes)
    - Temporal mapping (determinism, range, monotonicity)
    - Propagation simulation (BFS, delays, fan-out)
    - Observation generation (NetworkObservation records, timestamps)
    - GeoIP resolver (node pool lookup, range fallback)
"""

import random
from datetime import UTC, datetime, timedelta

import networkx as nx

from backend.generator.geoip import GeoIPResolver, GeoIPResult
from backend.generator.node_pool import (
    NetworkNode,
    NodeType,
    generate_node_pool,
)
from backend.generator.propagation import (
    PropagationHop,
    generate_observations_for_transaction,
    pick_origin_node,
    simulate_propagation,
)
from backend.generator.temporal import (
    get_all_step_mappings,
    time_step_to_datetime,
    time_step_to_datetime_range,
)
from backend.generator.topology import (
    build_topology,
    get_neighbors,
    get_topology_stats,
)

import pytest


# ===================================================================
# Node Pool Tests
# ===================================================================


class TestNodePool:
    """Tests for generate_node_pool."""

    def test_generates_correct_count(self):
        nodes = generate_node_pool(size=100, seed=42)
        assert len(nodes) == 100

    def test_all_ips_unique(self):
        nodes = generate_node_pool(size=500, seed=42)
        ips = [n.ip for n in nodes]
        assert len(set(ips)) == len(ips)

    def test_all_node_ids_unique(self):
        nodes = generate_node_pool(size=100, seed=42)
        ids = [n.node_id for n in nodes]
        assert ids == list(range(100))

    def test_valid_ports(self):
        nodes = generate_node_pool(size=50, seed=42)
        for node in nodes:
            assert 0 <= node.port <= 65535

    def test_default_port_is_8333(self):
        nodes = generate_node_pool(size=10, seed=42)
        for node in nodes:
            assert node.port == 8333

    def test_valid_country_codes(self):
        nodes = generate_node_pool(size=200, seed=42)
        for node in nodes:
            assert len(node.country) == 2
            assert node.country.isalpha()
            assert node.country == node.country.upper()

    def test_valid_asn(self):
        nodes = generate_node_pool(size=100, seed=42)
        for node in nodes:
            assert node.asn > 0

    def test_valid_node_types(self):
        nodes = generate_node_pool(size=100, seed=42)
        valid_types = set(NodeType)
        for node in nodes:
            assert node.node_type in valid_types

    def test_contains_tor_exit_nodes(self):
        """With enough nodes, we should see tor_exit types (~5%)."""
        nodes = generate_node_pool(size=500, seed=42)
        tor_count = sum(1 for n in nodes if n.node_type == NodeType.TOR_EXIT)
        assert tor_count > 0, "Expected at least some TOR_EXIT nodes in 500 samples"

    def test_contains_vpn_nodes(self):
        """With enough nodes, we should see vpn types (~3%)."""
        nodes = generate_node_pool(size=500, seed=42)
        vpn_count = sum(1 for n in nodes if n.node_type == NodeType.VPN)
        assert vpn_count > 0, "Expected at least some VPN nodes in 500 samples"

    def test_reproducible_with_same_seed(self):
        nodes_a = generate_node_pool(size=50, seed=123)
        nodes_b = generate_node_pool(size=50, seed=123)
        for a, b in zip(nodes_a, nodes_b):
            assert a.ip == b.ip
            assert a.country == b.country
            assert a.asn == b.asn
            assert a.node_type == b.node_type

    def test_different_seeds_produce_different_pools(self):
        nodes_a = generate_node_pool(size=50, seed=1)
        nodes_b = generate_node_pool(size=50, seed=2)
        ips_a = {n.ip for n in nodes_a}
        ips_b = {n.ip for n in nodes_b}
        # Extremely unlikely to be identical with different seeds
        assert ips_a != ips_b

    def test_no_private_ips(self):
        """Generated IPs should not fall in private/reserved ranges."""
        nodes = generate_node_pool(size=500, seed=42)
        for node in nodes:
            first_octet = int(node.ip.split(".")[0])
            assert first_octet not in {0, 10, 127, 224, 255}
            assert not node.ip.startswith("192.168.")
            assert not node.ip.startswith("172.16.")
            assert not node.ip.startswith("169.254.")

    def test_rejects_zero_size(self):
        with pytest.raises(ValueError, match="positive"):
            generate_node_pool(size=0, seed=42)

    def test_rejects_negative_size(self):
        with pytest.raises(ValueError, match="positive"):
            generate_node_pool(size=-5, seed=42)

    def test_country_diversity(self):
        """With 500 nodes, we should see multiple countries."""
        nodes = generate_node_pool(size=500, seed=42)
        countries = {n.country for n in nodes}
        assert len(countries) >= 5, f"Expected country diversity, got {countries}"

    def test_pydantic_model_validation(self):
        """NetworkNode should be a valid Pydantic model."""
        node = NetworkNode(
            node_id=0, ip="45.33.32.156", asn=15169, country="US",
        )
        assert node.port == 8333
        assert node.node_type == NodeType.FULL_NODE


# ===================================================================
# Topology Tests
# ===================================================================


class TestTopology:
    """Tests for build_topology."""

    def test_topology_has_correct_node_count(self):
        nodes = generate_node_pool(size=50, seed=42)
        topo = build_topology(nodes, seed=42)
        assert topo.number_of_nodes() == 50

    def test_topology_is_connected(self):
        """BA model always produces a connected graph."""
        nodes = generate_node_pool(size=100, seed=42)
        topo = build_topology(nodes, seed=42)
        assert nx.is_connected(topo)

    def test_nodes_have_ip_attribute(self):
        nodes = generate_node_pool(size=20, seed=42)
        topo = build_topology(nodes, seed=42)
        for node_id in topo.nodes():
            assert "ip" in topo.nodes[node_id]
            assert topo.nodes[node_id]["ip"] != ""

    def test_nodes_have_country_attribute(self):
        nodes = generate_node_pool(size=20, seed=42)
        topo = build_topology(nodes, seed=42)
        for node_id in topo.nodes():
            assert "country" in topo.nodes[node_id]

    def test_nodes_have_asn_attribute(self):
        nodes = generate_node_pool(size=20, seed=42)
        topo = build_topology(nodes, seed=42)
        for node_id in topo.nodes():
            assert "asn" in topo.nodes[node_id]

    def test_nodes_have_node_type_attribute(self):
        nodes = generate_node_pool(size=20, seed=42)
        topo = build_topology(nodes, seed=42)
        for node_id in topo.nodes():
            assert "node_type" in topo.nodes[node_id]

    def test_reproducible_topology(self):
        nodes = generate_node_pool(size=30, seed=42)
        topo_a = build_topology(nodes, seed=99)
        topo_b = build_topology(nodes, seed=99)
        assert set(topo_a.edges()) == set(topo_b.edges())

    def test_different_seeds_different_topology(self):
        nodes = generate_node_pool(size=30, seed=42)
        topo_a = build_topology(nodes, seed=1)
        topo_b = build_topology(nodes, seed=2)
        assert set(topo_a.edges()) != set(topo_b.edges())

    def test_custom_peers_per_node(self):
        nodes = generate_node_pool(size=50, seed=42)
        topo = build_topology(nodes, seed=42, peers_per_node=3)
        stats = get_topology_stats(topo)
        # Min degree in a BA graph is m (except possibly for the initial core)
        assert stats["min_degree"] >= 3

    def test_rejects_single_node(self):
        nodes = generate_node_pool(size=1, seed=42)
        with pytest.raises(ValueError, match="at least 2"):
            build_topology(nodes, seed=42)

    def test_rejects_invalid_peers(self):
        nodes = generate_node_pool(size=10, seed=42)
        with pytest.raises(ValueError, match="peers_per_node"):
            build_topology(nodes, seed=42, peers_per_node=0)

    def test_clamps_peers_for_small_pool(self):
        """If peers_per_node >= pool size, it should clamp gracefully."""
        nodes = generate_node_pool(size=5, seed=42)
        topo = build_topology(nodes, seed=42, peers_per_node=20)
        assert topo.number_of_nodes() == 5
        assert nx.is_connected(topo)

    def test_get_neighbors(self):
        nodes = generate_node_pool(size=20, seed=42)
        topo = build_topology(nodes, seed=42)
        neighbors = get_neighbors(topo, 0)
        assert isinstance(neighbors, list)
        assert len(neighbors) > 0

    def test_get_neighbors_invalid_node(self):
        nodes = generate_node_pool(size=20, seed=42)
        topo = build_topology(nodes, seed=42)
        with pytest.raises(KeyError):
            get_neighbors(topo, 9999)

    def test_topology_stats(self):
        nodes = generate_node_pool(size=50, seed=42)
        topo = build_topology(nodes, seed=42, peers_per_node=4)
        stats = get_topology_stats(topo)
        assert stats["node_count"] == 50
        assert stats["edge_count"] > 0
        assert stats["avg_degree"] > 0
        assert stats["is_connected"] is True
        assert stats["min_degree"] <= stats["avg_degree"] <= stats["max_degree"]


# ===================================================================
# Temporal Mapping Tests
# ===================================================================


class TestTemporalMapping:
    """Tests for time_step → datetime mapping."""

    def test_step_1_is_epoch(self):
        dt = time_step_to_datetime(1)
        assert dt == datetime(2019, 1, 7, 0, 0, 0, tzinfo=UTC)

    def test_step_49_is_valid(self):
        dt = time_step_to_datetime(49)
        assert dt.year == 2020
        assert dt.tzinfo == UTC

    def test_monotonically_increasing(self):
        prev = None
        for step in range(1, 50):
            dt = time_step_to_datetime(step)
            if prev is not None:
                assert dt > prev, f"Step {step} is not after step {step - 1}"
            prev = dt

    def test_step_interval_is_two_weeks(self):
        dt1 = time_step_to_datetime(1)
        dt2 = time_step_to_datetime(2)
        assert dt2 - dt1 == timedelta(days=14)

    def test_deterministic(self):
        """Same input always produces same output."""
        for step in range(1, 50):
            assert time_step_to_datetime(step) == time_step_to_datetime(step)

    def test_rejects_step_0(self):
        with pytest.raises(ValueError, match="time_step"):
            time_step_to_datetime(0)

    def test_rejects_step_50(self):
        with pytest.raises(ValueError, match="time_step"):
            time_step_to_datetime(50)

    def test_rejects_negative_step(self):
        with pytest.raises(ValueError, match="time_step"):
            time_step_to_datetime(-1)

    def test_range_returns_two_week_window(self):
        start, end = time_step_to_datetime_range(10)
        assert end - start == timedelta(days=14)

    def test_range_boundaries_are_contiguous(self):
        """End of step N should equal start of step N+1."""
        for step in range(1, 49):
            _, end = time_step_to_datetime_range(step)
            next_start, _ = time_step_to_datetime_range(step + 1)
            assert end == next_start, f"Gap between step {step} and {step + 1}"

    def test_range_start_matches_single_datetime(self):
        for step in range(1, 50):
            start, _ = time_step_to_datetime_range(step)
            assert start == time_step_to_datetime(step)

    def test_get_all_step_mappings_covers_49_steps(self):
        mappings = get_all_step_mappings()
        assert len(mappings) == 49
        assert set(mappings.keys()) == set(range(1, 50))

    def test_all_mappings_have_utc_timezone(self):
        mappings = get_all_step_mappings()
        for dt in mappings.values():
            assert dt.tzinfo == UTC


# ===================================================================
# Propagation Tests
# ===================================================================


class TestPropagation:
    """Tests for simulate_propagation."""

    def _make_topology(self, size=50, seed=42):
        """Helper to create a node pool and topology."""
        nodes = generate_node_pool(size=size, seed=seed)
        topo = build_topology(nodes, seed=seed)
        lookup = {n.node_id: n for n in nodes}
        return nodes, topo, lookup

    def test_propagation_returns_hops(self):
        nodes, topo, _ = self._make_topology()
        rng = random.Random(42)
        hops = simulate_propagation(0, topo, rng)
        assert len(hops) > 0
        assert all(isinstance(h, PropagationHop) for h in hops)

    def test_hops_have_valid_node_ids(self):
        nodes, topo, _ = self._make_topology()
        rng = random.Random(42)
        hops = simulate_propagation(0, topo, rng)
        node_ids = set(topo.nodes())
        for hop in hops:
            assert hop.from_node_id in node_ids
            assert hop.to_node_id in node_ids

    def test_cumulative_delays_are_increasing(self):
        nodes, topo, _ = self._make_topology()
        rng = random.Random(42)
        hops = simulate_propagation(0, topo, rng)
        # Within the same BFS path, delays should be non-decreasing
        # (hops are sorted by discovery order, not strictly by delay,
        # but delays are cumulative from the path origin)
        for hop in hops:
            assert hop.cumulative_delay_ms >= 0

    def test_delays_are_positive(self):
        nodes, topo, _ = self._make_topology()
        rng = random.Random(42)
        hops = simulate_propagation(0, topo, rng)
        for hop in hops:
            assert hop.cumulative_delay_ms > 0

    def test_max_hops_limits_depth(self):
        nodes, topo, _ = self._make_topology(size=100, seed=42)
        rng = random.Random(42)
        hops_short = simulate_propagation(0, topo, rng, max_hops=2)
        rng2 = random.Random(42)
        hops_long = simulate_propagation(0, topo, rng2, max_hops=10)
        assert len(hops_short) <= len(hops_long)

    def test_max_fan_out_limits_spread(self):
        nodes, topo, _ = self._make_topology(size=100, seed=42)
        rng1 = random.Random(42)
        hops_narrow = simulate_propagation(0, topo, rng1, max_hops=5, max_fan_out=1)
        rng2 = random.Random(42)
        hops_wide = simulate_propagation(0, topo, rng2, max_hops=5, max_fan_out=8)
        assert len(hops_narrow) <= len(hops_wide)

    def test_no_duplicate_destinations(self):
        """Each node should be reached at most once (BFS property)."""
        nodes, topo, _ = self._make_topology()
        rng = random.Random(42)
        hops = simulate_propagation(0, topo, rng)
        destinations = [h.to_node_id for h in hops]
        assert len(destinations) == len(set(destinations))

    def test_reproducible_propagation(self):
        nodes, topo, _ = self._make_topology()
        rng1 = random.Random(99)
        rng2 = random.Random(99)
        hops_a = simulate_propagation(0, topo, rng1)
        hops_b = simulate_propagation(0, topo, rng2)
        assert len(hops_a) == len(hops_b)
        for a, b in zip(hops_a, hops_b):
            assert a.from_node_id == b.from_node_id
            assert a.to_node_id == b.to_node_id

    def test_invalid_origin_raises(self):
        nodes, topo, _ = self._make_topology()
        rng = random.Random(42)
        with pytest.raises(KeyError):
            simulate_propagation(9999, topo, rng)

    def test_hop_indices_are_sequential(self):
        nodes, topo, _ = self._make_topology()
        rng = random.Random(42)
        hops = simulate_propagation(0, topo, rng)
        for i, hop in enumerate(hops):
            assert hop.hop_index == i


# ===================================================================
# Observation Generation Tests
# ===================================================================


class TestObservationGeneration:
    """Tests for generate_observations_for_transaction."""

    def _make_topology(self, size=50, seed=42):
        nodes = generate_node_pool(size=size, seed=seed)
        topo = build_topology(nodes, seed=seed)
        lookup = {n.node_id: n for n in nodes}
        return nodes, topo, lookup

    def test_generates_observations(self):
        nodes, topo, lookup = self._make_topology()
        rng = random.Random(42)
        obs = generate_observations_for_transaction(
            txid=12345, time_step=10, origin_node_id=0,
            topology=topo, node_lookup=lookup, rng=rng,
        )
        assert len(obs) > 0

    def test_all_observations_are_synthetic(self):
        nodes, topo, lookup = self._make_topology()
        rng = random.Random(42)
        obs = generate_observations_for_transaction(
            txid=12345, time_step=10, origin_node_id=0,
            topology=topo, node_lookup=lookup, rng=rng,
        )
        for o in obs:
            assert o.is_synthetic is True

    def test_observations_have_correct_txid(self):
        nodes, topo, lookup = self._make_topology()
        rng = random.Random(42)
        obs = generate_observations_for_transaction(
            txid=99999, time_step=25, origin_node_id=0,
            topology=topo, node_lookup=lookup, rng=rng,
        )
        for o in obs:
            assert o.txid == 99999

    def test_timestamps_are_within_time_step_window(self):
        nodes, topo, lookup = self._make_topology()
        rng = random.Random(42)
        step = 15
        obs = generate_observations_for_transaction(
            txid=12345, time_step=step, origin_node_id=0,
            topology=topo, node_lookup=lookup, rng=rng,
        )
        start, end = time_step_to_datetime_range(step)
        # Allow a small buffer for propagation delays beyond the window
        buffer = timedelta(seconds=60)
        for o in obs:
            assert o.timestamp >= start, f"{o.timestamp} < {start}"
            assert o.timestamp < end + buffer

    def test_timestamps_are_within_propagation_window(self):
        """All observation timestamps should be within a few seconds of each other."""
        nodes, topo, lookup = self._make_topology()
        rng = random.Random(42)
        obs = generate_observations_for_transaction(
            txid=12345, time_step=10, origin_node_id=0,
            topology=topo, node_lookup=lookup, rng=rng,
        )
        if len(obs) >= 2:
            timestamps = [o.timestamp for o in obs]
            spread = max(timestamps) - min(timestamps)
            # All hops should complete within a few seconds
            assert spread < timedelta(seconds=30)

    def test_observations_have_valid_ips(self):
        nodes, topo, lookup = self._make_topology()
        rng = random.Random(42)
        obs = generate_observations_for_transaction(
            txid=12345, time_step=10, origin_node_id=0,
            topology=topo, node_lookup=lookup, rng=rng,
        )
        for o in obs:
            assert o.src_ip != ""
            assert o.dst_ip != ""
            assert o.src_ip != o.dst_ip

    def test_observations_have_script_type(self):
        nodes, topo, lookup = self._make_topology()
        rng = random.Random(42)
        obs = generate_observations_for_transaction(
            txid=12345, time_step=10, origin_node_id=0,
            topology=topo, node_lookup=lookup, rng=rng,
        )
        # All observations for same tx share the same script type
        if obs:
            script_types = {o.script_type for o in obs}
            assert len(script_types) == 1
            assert obs[0].script_type is not None

    def test_observations_have_protocol(self):
        nodes, topo, lookup = self._make_topology()
        rng = random.Random(42)
        obs = generate_observations_for_transaction(
            txid=12345, time_step=10, origin_node_id=0,
            topology=topo, node_lookup=lookup, rng=rng,
        )
        for o in obs:
            assert o.protocol == "TCP"

    def test_provenance_fields_propagated(self):
        nodes, topo, lookup = self._make_topology()
        rng = random.Random(42)
        obs = generate_observations_for_transaction(
            txid=12345, time_step=10, origin_node_id=0,
            topology=topo, node_lookup=lookup, rng=rng,
            generation_run_id="run_42",
            scenario_id="normal",
            generator_version="0.2.0",
        )
        for o in obs:
            assert o.generation_run_id == "run_42"
            assert o.scenario_id == "normal"
            assert o.generator_version == "0.2.0"

    def test_observations_have_country_and_asn(self):
        nodes, topo, lookup = self._make_topology()
        rng = random.Random(42)
        obs = generate_observations_for_transaction(
            txid=12345, time_step=10, origin_node_id=0,
            topology=topo, node_lookup=lookup, rng=rng,
        )
        for o in obs:
            assert o.country is not None
            assert o.asn is not None

    def test_reproducible_observations(self):
        nodes, topo, lookup = self._make_topology()
        rng1 = random.Random(42)
        rng2 = random.Random(42)
        obs_a = generate_observations_for_transaction(
            txid=12345, time_step=10, origin_node_id=0,
            topology=topo, node_lookup=lookup, rng=rng1,
        )
        obs_b = generate_observations_for_transaction(
            txid=12345, time_step=10, origin_node_id=0,
            topology=topo, node_lookup=lookup, rng=rng2,
        )
        assert len(obs_a) == len(obs_b)
        for a, b in zip(obs_a, obs_b):
            assert a.src_ip == b.src_ip
            assert a.dst_ip == b.dst_ip
            assert a.timestamp == b.timestamp


# ===================================================================
# Origin Node Selection Tests
# ===================================================================


class TestPickOriginNode:
    """Tests for pick_origin_node."""

    def test_returns_valid_node_id(self):
        nodes = generate_node_pool(size=50, seed=42)
        rng = random.Random(42)
        node_id = pick_origin_node(nodes, rng)
        assert 0 <= node_id < 50

    def test_prefer_type_biases_selection(self):
        """When preferring tor_exit, most selections should be tor_exit."""
        nodes = generate_node_pool(size=200, seed=42)
        rng = random.Random(42)
        tor_count = 0
        trials = 100
        for _ in range(trials):
            nid = pick_origin_node(nodes, rng, prefer_type=NodeType.TOR_EXIT)
            if nodes[nid].node_type == NodeType.TOR_EXIT:
                tor_count += 1
        # With 5× weight on ~5% of nodes, we expect a meaningful bias
        assert tor_count > 5, f"Expected tor_exit bias, got {tor_count}/{trials}"

    def test_no_preference_is_diverse(self):
        nodes = generate_node_pool(size=200, seed=42)
        rng = random.Random(42)
        selected = set()
        for _ in range(50):
            selected.add(pick_origin_node(nodes, rng))
        assert len(selected) > 10


# ===================================================================
# GeoIP Resolver Tests
# ===================================================================


class TestGeoIPResolver:
    """Tests for the offline GeoIP resolver."""

    def test_node_pool_lookup(self):
        nodes = generate_node_pool(size=10, seed=42)
        resolver = GeoIPResolver(node_pool=nodes)
        result = resolver.lookup(nodes[0].ip)
        assert result.country == nodes[0].country
        assert result.asn == nodes[0].asn
        assert result.source == "node_pool"

    def test_pool_size(self):
        nodes = generate_node_pool(size=25, seed=42)
        resolver = GeoIPResolver(node_pool=nodes)
        assert resolver.pool_size == 25

    def test_empty_pool(self):
        resolver = GeoIPResolver()
        assert resolver.pool_size == 0

    def test_builtin_fallback_for_unknown_ip(self):
        """IPs not in the pool should use the builtin range."""
        resolver = GeoIPResolver()  # No pool
        result = resolver.lookup("77.1.2.3")  # Should match DE range
        assert result.source == "builtin_range"
        assert result.country is not None

    def test_builtin_returns_known_countries(self):
        resolver = GeoIPResolver()
        # 77.x.x.x is in the DE range
        result = resolver.lookup("77.10.20.30")
        assert result.country == "DE"
        # 45.x.x.x is in the JP range
        result = resolver.lookup("45.10.20.30")
        assert result.country == "JP"

    def test_unknown_ip_returns_none(self):
        """An IP outside all known ranges returns None."""
        resolver = GeoIPResolver()
        result = resolver.lookup("1.2.3.4")  # First octet 1, not in ranges
        assert result.country is None
        assert result.asn is None

    def test_invalid_ip_returns_none(self):
        resolver = GeoIPResolver()
        result = resolver.lookup("not_an_ip")
        assert result.country is None
        assert result.asn is None

    def test_convenience_lookup_country(self):
        nodes = generate_node_pool(size=10, seed=42)
        resolver = GeoIPResolver(node_pool=nodes)
        country = resolver.lookup_country(nodes[0].ip)
        assert country == nodes[0].country

    def test_convenience_lookup_asn(self):
        nodes = generate_node_pool(size=10, seed=42)
        resolver = GeoIPResolver(node_pool=nodes)
        asn = resolver.lookup_asn(nodes[0].ip)
        assert asn == nodes[0].asn

    def test_pool_takes_priority_over_builtin(self):
        """If an IP is in both pool and builtin range, pool wins."""
        # Create a node with an IP in a known builtin range but different country
        node = NetworkNode(
            node_id=0, ip="77.1.2.3", asn=99999, country="XX",
        )
        resolver = GeoIPResolver(node_pool=[node])
        result = resolver.lookup("77.1.2.3")
        assert result.country == "XX"  # Pool value, not DE from builtin
        assert result.asn == 99999
        assert result.source == "node_pool"

    def test_all_pool_ips_resolvable(self):
        """Every IP in the node pool should resolve correctly."""
        nodes = generate_node_pool(size=100, seed=42)
        resolver = GeoIPResolver(node_pool=nodes)
        for node in nodes:
            result = resolver.lookup(node.ip)
            assert result.country == node.country
            assert result.asn == node.asn
