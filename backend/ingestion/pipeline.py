"""Unified Ingestion Pipeline for Elliptic++ dataset.

Orchestrates parsing and validation of all 8 CSV files.
"""

import logging
from pathlib import Path

import pandas as pd

from backend.ingestion.csv_parser import (
    parse_addr_addr_edgelist,
    parse_addr_tx_edgelist,
    parse_tx_addr_edgelist,
    parse_txs_classes,
    parse_txs_edgelist,
    parse_txs_features,
    parse_wallets_classes,
    parse_wallets_features,
)
from backend.ingestion.errors import IngestionError
from backend.ingestion.validator import (
    validate_edgelist,
    validate_txs_classes,
    validate_txs_features,
    validate_wallets_classes,
    validate_wallets_features,
)

logger = logging.getLogger(__name__)


def run_ingestion_pipeline(dataset_dir: Path) -> dict[str, pd.DataFrame]:
    """Run the complete ingestion pipeline on the dataset directory.

    Parses all 8 CSV files and validates them.
    If any file is missing or invalid, an IngestionError is raised.

    Args:
        dataset_dir: Path to the Elliptic++ dataset directory.

    Returns:
        Dictionary mapping dataset names to their parsed and validated DataFrames.
    """
    logger.info("Starting ingestion pipeline from %s", dataset_dir)

    # 1. Parse all files
    datasets = {
        "txs_features": parse_txs_features(dataset_dir / "txs_features.csv"),
        "txs_classes": parse_txs_classes(dataset_dir / "txs_classes.csv"),
        "wallets_features": parse_wallets_features(dataset_dir / "wallets_features.csv"),
        "wallets_classes": parse_wallets_classes(dataset_dir / "wallets_classes.csv"),
        "txs_edgelist": parse_txs_edgelist(dataset_dir / "txs_edgelist.csv"),
        "addr_tx_edgelist": parse_addr_tx_edgelist(dataset_dir / "AddrTx_edgelist.csv"),
        "tx_addr_edgelist": parse_tx_addr_edgelist(dataset_dir / "TxAddr_edgelist.csv"),
        "addr_addr_edgelist": parse_addr_addr_edgelist(dataset_dir / "AddrAddr_edgelist.csv"),
    }

    # 2. Validate all files
    validation_results = {
        "txs_features": validate_txs_features(datasets["txs_features"]),
        "txs_classes": validate_txs_classes(datasets["txs_classes"]),
        "wallets_features": validate_wallets_features(datasets["wallets_features"]),
        "wallets_classes": validate_wallets_classes(datasets["wallets_classes"]),
        "txs_edgelist": validate_edgelist(datasets["txs_edgelist"], "txs_edgelist"),
        "addr_tx_edgelist": validate_edgelist(datasets["addr_tx_edgelist"], "addr_tx_edgelist"),
        "tx_addr_edgelist": validate_edgelist(datasets["tx_addr_edgelist"], "tx_addr_edgelist"),
        "addr_addr_edgelist": validate_edgelist(
            datasets["addr_addr_edgelist"], "addr_addr_edgelist"
        ),
    }

    # 3. Check for errors
    all_errors = []
    for name, result in validation_results.items():
        if not result.is_valid:
            for err in result.errors:
                all_errors.append(f"{name}: {err}")

    if all_errors:
        raise IngestionError(
            f"Validation failed with {len(all_errors)} errors:\n" + "\n".join(all_errors)
        )

    logger.info("Ingestion pipeline completed successfully")
    return datasets
