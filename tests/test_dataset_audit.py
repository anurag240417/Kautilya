"""Dataset audit tests for Elliptic++ files.

Programmatic validation of the actual Elliptic++ dataset files:
schemas, row counts, class distributions, temporal structure,
and referential integrity between edgelists and feature files.

Tests are skipped if dataset files are not present (CI-safe).
"""

import pandas as pd
import pytest

from backend.config import get_settings

settings = get_settings()

# Expected file paths
DATASET_DIR = settings.elliptic_pp_dir

# Skip all tests if dataset directory doesn't exist
pytestmark = pytest.mark.skipif(
    not DATASET_DIR.exists(),
    reason=f"Elliptic++ dataset not found at {DATASET_DIR}",
)

# --- Expected schemas ---

EXPECTED_TXS_FEATURES_COLS = (
    ["txId", "Time step"]
    + [f"Local_feature_{i}" for i in range(1, 94)]
    + [f"Aggregate_feature_{i}" for i in range(1, 73)]
    + [
        "in_txs_degree", "out_txs_degree", "total_BTC", "fees", "size",
        "num_input_addresses", "num_output_addresses",
        "in_BTC_min", "in_BTC_max", "in_BTC_mean", "in_BTC_median", "in_BTC_total",
        "out_BTC_min", "out_BTC_max", "out_BTC_mean", "out_BTC_median", "out_BTC_total",
    ]
)

EXPECTED_TXS_CLASSES_COLS = ["txId", "class"]

EXPECTED_WALLETS_FEATURES_FIRST_COLS = [
    "address", "Time step", "num_txs_as_sender", "num_txs_as receiver",
]

EXPECTED_WALLETS_CLASSES_COLS = ["address", "class"]

EXPECTED_TXS_EDGELIST_COLS = ["txId1", "txId2"]
EXPECTED_ADDR_TX_COLS = ["input_address", "txId"]
EXPECTED_TX_ADDR_COLS = ["txId", "output_address"]
EXPECTED_ADDR_ADDR_COLS = ["input_address", "output_address"]


# --- Helpers ---


def _read_header(filename: str) -> list[str]:
    """Read only the header row of a CSV file."""
    df = pd.read_csv(DATASET_DIR / filename, nrows=0)
    return list(df.columns)


def _count_rows(filename: str) -> int:
    """Count data rows (excluding header) efficiently."""
    # Read only first column to count rows without loading all data
    df = pd.read_csv(DATASET_DIR / filename, usecols=[0])
    return len(df)


# --- File existence ---


class TestFileExistence:
    def test_txs_features_exists(self):
        assert (DATASET_DIR / "txs_features.csv").exists()

    def test_txs_classes_exists(self):
        assert (DATASET_DIR / "txs_classes.csv").exists()

    def test_wallets_features_exists(self):
        assert (DATASET_DIR / "wallets_features.csv").exists()

    def test_wallets_classes_exists(self):
        assert (DATASET_DIR / "wallets_classes.csv").exists()

    def test_txs_edgelist_exists(self):
        assert (DATASET_DIR / "txs_edgelist.csv").exists()

    def test_addr_tx_edgelist_exists(self):
        assert (DATASET_DIR / "AddrTx_edgelist.csv").exists()

    def test_tx_addr_edgelist_exists(self):
        assert (DATASET_DIR / "TxAddr_edgelist.csv").exists()

    def test_addr_addr_edgelist_exists(self):
        assert (DATASET_DIR / "AddrAddr_edgelist.csv").exists()


# --- Schema validation ---


class TestSchemas:
    def test_txs_features_schema(self):
        header = _read_header("txs_features.csv")
        assert header == EXPECTED_TXS_FEATURES_COLS, (
            f"Expected {len(EXPECTED_TXS_FEATURES_COLS)} columns, got {len(header)}. "
            f"Mismatch at: {set(header) ^ set(EXPECTED_TXS_FEATURES_COLS)}"
        )

    def test_txs_features_column_count(self):
        header = _read_header("txs_features.csv")
        assert len(header) == 184, f"Expected 184 columns, got {len(header)}"

    def test_txs_classes_schema(self):
        header = _read_header("txs_classes.csv")
        assert header == EXPECTED_TXS_CLASSES_COLS

    def test_wallets_features_starts_with_expected_cols(self):
        header = _read_header("wallets_features.csv")
        assert header[:4] == EXPECTED_WALLETS_FEATURES_FIRST_COLS, (
            f"First 4 columns: {header[:4]}"
        )

    def test_wallets_features_column_count(self):
        header = _read_header("wallets_features.csv")
        assert len(header) == 57, f"Expected 57 columns, got {len(header)}"

    def test_wallets_classes_schema(self):
        header = _read_header("wallets_classes.csv")
        assert header == EXPECTED_WALLETS_CLASSES_COLS

    def test_txs_edgelist_schema(self):
        header = _read_header("txs_edgelist.csv")
        assert header == EXPECTED_TXS_EDGELIST_COLS

    def test_addr_tx_edgelist_schema(self):
        header = _read_header("AddrTx_edgelist.csv")
        assert header == EXPECTED_ADDR_TX_COLS

    def test_tx_addr_edgelist_schema(self):
        header = _read_header("TxAddr_edgelist.csv")
        assert header == EXPECTED_TX_ADDR_COLS

    def test_addr_addr_edgelist_schema(self):
        header = _read_header("AddrAddr_edgelist.csv")
        assert header == EXPECTED_ADDR_ADDR_COLS


# --- Row counts ---


class TestRowCounts:
    def test_txs_features_row_count(self):
        count = _count_rows("txs_features.csv")
        assert count == 203769, f"Expected 203,769 rows, got {count:,}"

    def test_txs_classes_row_count(self):
        count = _count_rows("txs_classes.csv")
        assert count == 203769, f"Expected 203,769 rows, got {count:,}"

    def test_txs_edgelist_row_count(self):
        count = _count_rows("txs_edgelist.csv")
        assert count == 234355, f"Expected 234,355 rows, got {count:,}"

    def test_wallets_features_row_count(self):
        count = _count_rows("wallets_features.csv")
        assert count >= 1_200_000, f"Expected ~1.27M rows, got {count:,}"

    def test_wallets_classes_row_count(self):
        count = _count_rows("wallets_classes.csv")
        assert count >= 800_000, f"Expected ~822K rows, got {count:,}"


# --- Class values and distributions ---


class TestClassValues:
    def test_txs_class_values_valid(self):
        """All transaction class values must be in {1, 2, 3}."""
        df = pd.read_csv(DATASET_DIR / "txs_classes.csv")
        invalid = df[~df["class"].isin([1, 2, 3])]
        assert len(invalid) == 0, f"Found {len(invalid)} invalid class values"

    def test_wallets_class_values_valid(self):
        """All wallet class values must be in {1, 2, 3}."""
        df = pd.read_csv(DATASET_DIR / "wallets_classes.csv")
        invalid = df[~df["class"].isin([1, 2, 3])]
        assert len(invalid) == 0, f"Found {len(invalid)} invalid class values"

    def test_txs_class_distribution(self):
        """Report class distribution (informational — prints counts)."""
        df = pd.read_csv(DATASET_DIR / "txs_classes.csv")
        dist = df["class"].value_counts().sort_index()
        # Just verify all 3 classes are present
        assert set(dist.index) == {1, 2, 3}, f"Classes present: {set(dist.index)}"
        # Print distribution for audit record
        print("\nTransaction class distribution:")
        print(f"  Illicit (1): {dist.get(1, 0):,}")
        print(f"  Licit   (2): {dist.get(2, 0):,}")
        print(f"  Unknown (3): {dist.get(3, 0):,}")

    def test_wallets_class_distribution(self):
        """Report wallet class distribution."""
        df = pd.read_csv(DATASET_DIR / "wallets_classes.csv")
        dist = df["class"].value_counts().sort_index()
        assert set(dist.index) == {1, 2, 3}, f"Classes present: {set(dist.index)}"
        print("\nWallet class distribution:")
        print(f"  Illicit (1): {dist.get(1, 0):,}")
        print(f"  Licit   (2): {dist.get(2, 0):,}")
        print(f"  Unknown (3): {dist.get(3, 0):,}")


# --- Temporal structure ---


class TestTemporalStructure:
    def test_time_step_range(self):
        """Transaction time_step values must span 1–49."""
        df = pd.read_csv(DATASET_DIR / "txs_features.csv", usecols=["txId", "Time step"])
        assert df["Time step"].min() == 1, f"Min time_step: {df['Time step'].min()}"
        assert df["Time step"].max() == 49, f"Max time_step: {df['Time step'].max()}"

    def test_all_49_time_steps_present(self):
        """All 49 time steps must be present in the dataset."""
        df = pd.read_csv(DATASET_DIR / "txs_features.csv", usecols=["txId", "Time step"])
        unique_steps = sorted(df["Time step"].unique())
        assert unique_steps == list(range(1, 50)), (
            f"Missing time steps: {set(range(1, 50)) - set(unique_steps)}"
        )

    def test_time_step_distribution(self):
        """Report transactions per time step (informational)."""
        df = pd.read_csv(DATASET_DIR / "txs_features.csv", usecols=["txId", "Time step"])
        dist = df["Time step"].value_counts().sort_index()
        print("\nTransactions per time step (min/max/mean):")
        print(f"  Min: {dist.min():,} (step {dist.idxmin()})")
        print(f"  Max: {dist.max():,} (step {dist.idxmax()})")
        print(f"  Mean: {dist.mean():,.0f}")


# --- Primary key integrity ---


class TestPrimaryKeyIntegrity:
    def test_txs_features_no_null_ids(self):
        """No null txId values in features."""
        df = pd.read_csv(DATASET_DIR / "txs_features.csv", usecols=["txId"])
        assert df["txId"].isna().sum() == 0

    def test_wallets_features_no_null_addresses(self):
        """No null address values in wallet features."""
        df = pd.read_csv(DATASET_DIR / "wallets_features.csv", usecols=["address"])
        assert df["address"].isna().sum() == 0

    def test_txid_consistency_features_vs_classes(self):
        """txIds in classes file must match txIds in features file."""
        features_ids = set(
            pd.read_csv(DATASET_DIR / "txs_features.csv", usecols=["txId"])["txId"]
        )
        classes_ids = set(
            pd.read_csv(DATASET_DIR / "txs_classes.csv", usecols=["txId"])["txId"]
        )
        missing_in_features = classes_ids - features_ids
        missing_in_classes = features_ids - classes_ids
        assert len(missing_in_features) == 0, (
            f"{len(missing_in_features)} txIds in classes but not in features"
        )
        assert len(missing_in_classes) == 0, (
            f"{len(missing_in_classes)} txIds in features but not in classes"
        )

    def test_wallet_label_coverage(self):
        """Document wallet address label coverage.

        wallets_features.csv has ~1.27M rows but many duplicate addresses
        (same address at different time_steps). The unique address count
        should roughly match wallets_classes.csv.
        """
        features_df = pd.read_csv(DATASET_DIR / "wallets_features.csv", usecols=["address"])
        feature_addrs = set(features_df["address"])
        class_addrs = set(
            pd.read_csv(DATASET_DIR / "wallets_classes.csv", usecols=["address"])["address"]
        )
        unlabeled = feature_addrs - class_addrs
        labeled = feature_addrs & class_addrs
        print("\nWallet label coverage:")
        print(f"  Total rows in wallets_features: {len(features_df):,}")
        print(f"  Unique addresses with features: {len(feature_addrs):,}")
        print(f"  Addresses with labels: {len(labeled):,}")
        print(f"  Addresses without labels: {len(unlabeled):,}")
        print(f"  Label coverage: {len(labeled) / len(feature_addrs) * 100:.1f}%")
        # All unique addresses should have labels
        assert len(unlabeled) == 0, (
            f"{len(unlabeled):,} addresses have features but no label"
        )


# --- Edgelist referential integrity ---


class TestEdgelistIntegrity:
    def test_txs_edgelist_ids_exist_in_features(self):
        """txIds in the transaction edgelist should exist in txs_features."""
        feature_ids = set(
            pd.read_csv(DATASET_DIR / "txs_features.csv", usecols=["txId"])["txId"]
        )
        edgelist = pd.read_csv(DATASET_DIR / "txs_edgelist.csv")
        edge_ids = set(edgelist["txId1"]) | set(edgelist["txId2"])
        missing = edge_ids - feature_ids
        # Report but allow some missing (dataset may have edge-only nodes)
        if missing:
            print(f"\n  {len(missing):,} txIds in edgelist not in features (edge-only nodes)")
        # At least 95% of edge txIds should be in features
        coverage = 1 - len(missing) / len(edge_ids)
        assert coverage >= 0.95, (
            f"Only {coverage:.1%} of edgelist txIds found in features"
        )

    def test_addr_tx_edgelist_sample_addresses(self):
        """Sample of addresses from AddrTx edgelist should exist in wallets_features."""
        wallet_addrs = set(
            pd.read_csv(DATASET_DIR / "wallets_features.csv", usecols=["address"])["address"]
        )
        # Read a sample rather than the full edgelist for efficiency
        edgelist = pd.read_csv(DATASET_DIR / "AddrTx_edgelist.csv", nrows=10000)
        edge_addrs = set(edgelist["input_address"])
        found = edge_addrs & wallet_addrs
        coverage = len(found) / len(edge_addrs)
        print(f"\n  AddrTx sample address coverage: {coverage:.1%}")
        assert coverage >= 0.90, (
            f"Only {coverage:.1%} of sampled addresses found in wallet features"
        )

    def test_no_duplicate_txs_edgelist(self):
        """Check for duplicate edges in transaction edgelist."""
        edgelist = pd.read_csv(DATASET_DIR / "txs_edgelist.csv")
        dupes = edgelist.duplicated().sum()
        print(f"\n  Duplicate tx-tx edges: {dupes:,}")
        # Warning-level: report but don't fail unless >1% are dupes
        dupe_rate = dupes / len(edgelist) if len(edgelist) > 0 else 0
        assert dupe_rate < 0.01, f"Duplicate rate: {dupe_rate:.2%}"
