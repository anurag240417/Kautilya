"""Correlation module.

Finds relationships between normalized evidence. Correlation
results must preserve evidence type and confidence.

Temporal correlation alone must NOT be treated as proof of
wallet/IP ownership.
"""

from .entity_resolution import (
    WalletIPCorrelation,
    resolve_ip_to_candidate_wallets,
    resolve_wallet_to_candidate_ips,
)
from .temporal import (
    DEFAULT_TEMPORAL_CONFIG,
    TemporalWindowConfig,
    compute_propagation_consistency,
    compute_temporal_proximity,
    correlate_temporally,
    get_temporal_window,
    is_in_temporal_window,
)
from .tx_ip import (
    DEFAULT_CORRELATION_CONFIG,
    CorrelationConfig,
    compute_confidence_metrics,
    compute_topology_consistency,
    correlate_ip_to_transactions,
    correlate_transaction_to_ips,
)

__all__ = [
    # Configs
    "TemporalWindowConfig",
    "DEFAULT_TEMPORAL_CONFIG",
    "CorrelationConfig",
    "DEFAULT_CORRELATION_CONFIG",
    # Temporal correlation
    "get_temporal_window",
    "is_in_temporal_window",
    "compute_temporal_proximity",
    "compute_propagation_consistency",
    "correlate_temporally",
    # Transaction-to-IP correlation
    "compute_topology_consistency",
    "compute_confidence_metrics",
    "correlate_transaction_to_ips",
    "correlate_ip_to_transactions",
    # Entity resolution
    "WalletIPCorrelation",
    "resolve_wallet_to_candidate_ips",
    "resolve_ip_to_candidate_wallets",
]
