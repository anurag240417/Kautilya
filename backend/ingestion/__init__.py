"""Data ingestion module.

Reads external data formats for the blockchain/wallet layer and any
explicitly supplied offline input files. Network-layer data is generated
by ``generator/``; it must not be conflated with this ingestion pipeline.
"""

from .csv_parser import (
    parse_addr_addr_edgelist,
    parse_addr_tx_edgelist,
    parse_tx_addr_edgelist,
    parse_txs_classes,
    parse_txs_edgelist,
    parse_txs_features,
    parse_wallets_classes,
    parse_wallets_features,
)
from .errors import IngestionError
from .json_parser import parse_json
from .pipeline import run_ingestion_pipeline
from .validator import (
    ValidationResult,
    validate_edgelist,
    validate_txs_classes,
    validate_txs_features,
    validate_wallets_classes,
    validate_wallets_features,
)
from .xml_parser import parse_xml

__all__ = [
    "IngestionError",
    "ValidationResult",
    "parse_addr_addr_edgelist",
    "parse_addr_tx_edgelist",
    "parse_json",
    "parse_tx_addr_edgelist",
    "parse_txs_classes",
    "parse_txs_edgelist",
    "parse_txs_features",
    "parse_wallets_classes",
    "parse_wallets_features",
    "parse_xml",
    "run_ingestion_pipeline",
    "validate_edgelist",
    "validate_txs_classes",
    "validate_txs_features",
    "validate_wallets_classes",
    "validate_wallets_features",
]
