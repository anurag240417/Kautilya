"""Synthetic network data generator module.

Generates synthetic network observations (IP, port, ASN, country,
propagation timestamps) for Elliptic++ transactions. All generated
data carries ``is_synthetic=True``.

Must use reproducible random seeds and record generation configuration.
See CONTEXT.md §3, DATASET.md §9–10, and AGENTS.md §6–7.
"""

from backend.generator.anomalies import generate_scenario_observations
from backend.generator.export import (
    export_ground_truth_to_csv,
    export_observations_to_csv,
    observations_to_dataframe,
)
from backend.generator.geoip import GeoIPResolver, GeoIPResult
from backend.generator.ground_truth import GroundTruthRecord, GroundTruthTracker
from backend.generator.node_pool import NetworkNode, NodeType, generate_node_pool
from backend.generator.pipeline import (
    generate_synthetic_network_dataset,
    validate_network_generation_results,
)
from backend.generator.propagation import (
    PropagationHop,
    generate_observations_for_transaction,
    pick_origin_node,
    simulate_propagation,
)
from backend.generator.scenarios import (
    ScenarioParams,
    ScenarioType,
    get_all_scenario_types,
    select_scenario,
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

__all__ = [
    # Node pool & Topology
    "NetworkNode",
    "NodeType",
    "generate_node_pool",
    "build_topology",
    "get_neighbors",
    "get_topology_stats",
    # GeoIP
    "GeoIPResolver",
    "GeoIPResult",
    # Temporal mapping
    "time_step_to_datetime",
    "time_step_to_datetime_range",
    "get_all_step_mappings",
    # Scenarios & Anomalies
    "ScenarioType",
    "ScenarioParams",
    "select_scenario",
    "get_all_scenario_types",
    "generate_scenario_observations",
    # Propagation simulation
    "PropagationHop",
    "simulate_propagation",
    "generate_observations_for_transaction",
    "pick_origin_node",
    # Ground truth & Export
    "GroundTruthRecord",
    "GroundTruthTracker",
    "observations_to_dataframe",
    "export_observations_to_csv",
    "export_ground_truth_to_csv",
    # Pipeline & Validation
    "generate_synthetic_network_dataset",
    "validate_network_generation_results",
]
