"""ML feature registry — single source of truth for feature definitions.

Defines all feature sets, label policies, and the M1/M2/M3
experimental feature configurations referenced in CONTEXT.md.

Feature classifications follow DATASET.md §7:
    - Interpretable features may appear in human-facing explanations.
    - Anonymized features feed ML internally but MUST NOT appear
      directly in explanations.
"""

# ======================================================================
# Label Policy
# ======================================================================

LABEL_ILLICIT: int = 1
LABEL_LICIT: int = 2
LABEL_UNKNOWN: int = 3

# Only Illicit and Licit are used for supervised training.
# Unknown (3) is excluded from training but can be scored at inference.
TRAIN_LABELS: frozenset[int] = frozenset({LABEL_ILLICIT, LABEL_LICIT})

# Binary target mapping for the classifier:
#   Illicit → 1 (positive class)
#   Licit   → 0 (negative class)
BINARY_TARGET_MAP: dict[int, int] = {
    LABEL_ILLICIT: 1,
    LABEL_LICIT: 0,
}


# ======================================================================
# Transaction Features (from txs_features.csv, 184 columns total)
# ======================================================================

# 17 interpretable transaction features (columns 168–184).
# These may be used in human-facing explanations.
INTERPRETABLE_TX_FEATURES: list[str] = [
    "in_txs_degree",
    "out_txs_degree",
    "total_BTC",
    "fees",
    "size",
    "num_input_addresses",
    "num_output_addresses",
    "in_BTC_min",
    "in_BTC_max",
    "in_BTC_mean",
    "in_BTC_median",
    "in_BTC_total",
    "out_BTC_min",
    "out_BTC_max",
    "out_BTC_mean",
    "out_BTC_median",
    "out_BTC_total",
]

# 93 local features (anonymized, not interpretable).
ANONYMIZED_LOCAL_FEATURES: list[str] = [f"Local_feature_{i}" for i in range(1, 94)]

# 72 aggregate features (anonymized, not interpretable).
ANONYMIZED_AGGREGATE_FEATURES: list[str] = [f"Aggregate_feature_{i}" for i in range(1, 73)]

# Combined anonymized features (165 total).
ANONYMIZED_TX_FEATURES: list[str] = ANONYMIZED_LOCAL_FEATURES + ANONYMIZED_AGGREGATE_FEATURES

# All blockchain transaction features (182, excluding txId and time_step).
ALL_TX_FEATURES: list[str] = ANONYMIZED_TX_FEATURES + INTERPRETABLE_TX_FEATURES


# ======================================================================
# Wallet Features (from wallets_features.csv, 57 columns total)
# ======================================================================

# 10 scalar wallet features (interpretable).
WALLET_SCALAR_FEATURES: list[str] = [
    "num_txs_as_sender",
    "num_txs_as_receiver",
    "first_block_appeared_in",
    "last_block_appeared_in",
    "lifetime_in_blocks",
    "total_txs",
    "first_sent_block",
    "first_received_block",
    "num_timesteps_appeared_in",
    "num_addr_transacted_multiple",
]

# 9 grouped stats features × 5 summary values each = 45 features.
_STATS_GROUPS: list[str] = [
    "btc_transacted",
    "btc_sent",
    "btc_received",
    "fees",
    "fees_as_share",
    "blocks_btwn_txs",
    "blocks_btwn_input_txs",
    "blocks_btwn_output_txs",
    "transacted_w_address",
]
_STATS_SUFFIXES: list[str] = ["total", "min", "max", "mean", "median"]

WALLET_STATS_FEATURES: list[str] = [
    f"{group}_{suffix}" for group in _STATS_GROUPS for suffix in _STATS_SUFFIXES
]

# All wallet features (55 total, excluding address and time_step).
ALL_WALLET_FEATURES: list[str] = WALLET_SCALAR_FEATURES + WALLET_STATS_FEATURES


# ======================================================================
# Graph-Derived Features (from Phase 3 — graph/features.py)
# ======================================================================

GRAPH_FEATURES: list[str] = [
    "in_degree",
    "out_degree",
    "total_degree",
    "pagerank",
    "betweenness_centrality",
    "clustering_coefficient",
    "hub_score",
    "authority_score",
]


# ======================================================================
# Network-Derived Features (from Phase 4 — generator)
# ======================================================================

# These are aggregate statistics computed from synthetic network
# observations for each transaction. They will be populated in
# the dataset preparation step.
NETWORK_FEATURES: list[str] = [
    "num_observations",
    "num_unique_src_ips",
    "num_unique_dst_ips",
    "num_unique_countries",
    "num_unique_asns",
    "mean_propagation_delay_ms",
    "max_propagation_delay_ms",
    "uses_tor",
    "uses_vpn",
]


# ======================================================================
# Experimental Feature Sets (CONTEXT.md — Experimental Progression)
# ======================================================================

FEATURE_SETS: dict[str, list[str]] = {
    # M1: blockchain features only
    "M1": ALL_TX_FEATURES,
    # M2: blockchain + graph-derived features
    "M2": ALL_TX_FEATURES + GRAPH_FEATURES,
    # M3: blockchain + graph + network features
    "M3": ALL_TX_FEATURES + GRAPH_FEATURES + NETWORK_FEATURES,
}


# ======================================================================
# Forbidden columns — must NEVER appear in feature matrix X
# ======================================================================

FORBIDDEN_FEATURE_COLUMNS: frozenset[str] = frozenset(
    {
        "txId",
        "txid",
        "address",
        "class",
        "label",
        "time_step",
        "Time step",
    }
)
