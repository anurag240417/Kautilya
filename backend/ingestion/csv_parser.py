"""CSV parser for Elliptic++ dataset files.

Reads and validates CSV files from the Elliptic++ dataset,
producing pandas DataFrames with canonical column names for
downstream normalization.

Target files:
    - txs_features.csv      (184 columns)
    - txs_classes.csv        (2 columns)
    - wallets_features.csv   (57 columns)
    - wallets_classes.csv    (2 columns)
    - txs_edgelist.csv       (2 columns)
    - AddrTx_edgelist.csv    (2 columns)
    - TxAddr_edgelist.csv    (2 columns)
    - AddrAddr_edgelist.csv  (2 columns)
"""

import logging
from pathlib import Path

import pandas as pd

from backend.ingestion.errors import IngestionError

logger = logging.getLogger(__name__)

# --- Expected schemas ---

_TXS_FEATURES_FIRST_COLS = ["txId", "Time step"]
_TXS_FEATURES_NCOLS = 184

_TXS_CLASSES_COLS = ["txId", "class"]
_WALLETS_CLASSES_COLS = ["address", "class"]

_WALLETS_FEATURES_FIRST_COLS = ["address", "Time step"]
_WALLETS_FEATURES_NCOLS = 57

_TXS_EDGELIST_COLS = ["txId1", "txId2"]
_ADDR_TX_COLS = ["input_address", "txId"]
_TX_ADDR_COLS = ["txId", "output_address"]
_ADDR_ADDR_COLS = ["input_address", "output_address"]

# --- Column renames (quirky headers → canonical names) ---

_HEADER_RENAMES = {
    "Time step": "time_step",
    "num_txs_as receiver": "num_txs_as_receiver",
}


def _read_csv(path: Path, expected_cols: list[str] | None = None) -> pd.DataFrame:
    """Read a CSV file with basic schema validation.

    Args:
        path: Path to the CSV file.
        expected_cols: If provided, validates that the DataFrame
            has exactly these columns (in order).

    Returns:
        Parsed DataFrame.

    Raises:
        IngestionError: If the file is missing, unreadable, or has
            an unexpected schema.
    """
    if not path.exists():
        msg = f"File not found: {path}"
        raise IngestionError(msg)

    try:
        df = pd.read_csv(path)
    except Exception as exc:
        msg = f"Failed to read CSV file {path}: {exc}"
        raise IngestionError(msg) from exc

    if expected_cols is not None and list(df.columns) != expected_cols:
        msg = (
            f"Schema mismatch in {path.name}: "
            f"expected columns {expected_cols}, "
            f"got {list(df.columns)}"
        )
        raise IngestionError(msg)

    logger.info("Parsed %s: %d rows, %d columns", path.name, len(df), len(df.columns))
    return df


def _validate_col_count(df: pd.DataFrame, path: Path, expected: int) -> None:
    """Validate column count without checking exact names."""
    if len(df.columns) != expected:
        msg = f"Column count mismatch in {path.name}: expected {expected}, got {len(df.columns)}"
        raise IngestionError(msg)


def _validate_first_cols(df: pd.DataFrame, path: Path, expected: list[str]) -> None:
    """Validate that the first N columns match expected names."""
    actual = list(df.columns[: len(expected)])
    if actual != expected:
        msg = f"Leading columns mismatch in {path.name}: expected {expected}, got {actual}"
        raise IngestionError(msg)


def _rename_headers(df: pd.DataFrame) -> pd.DataFrame:
    """Rename quirky dataset headers to canonical names."""
    renames = {k: v for k, v in _HEADER_RENAMES.items() if k in df.columns}
    if renames:
        df = df.rename(columns=renames)
    return df


# --- Public parser functions ---


def parse_txs_features(path: Path) -> pd.DataFrame:
    """Parse txs_features.csv (203,769 rows × 184 columns).

    Renames ``Time step`` → ``time_step``.
    """
    df = _read_csv(path)
    _validate_col_count(df, path, _TXS_FEATURES_NCOLS)
    _validate_first_cols(df, path, _TXS_FEATURES_FIRST_COLS)
    return _rename_headers(df)


def parse_txs_classes(path: Path) -> pd.DataFrame:
    """Parse txs_classes.csv (203,769 rows × 2 columns)."""
    return _read_csv(path, expected_cols=_TXS_CLASSES_COLS)


def parse_wallets_features(path: Path) -> pd.DataFrame:
    """Parse wallets_features.csv (1,268,260 rows × 57 columns).

    Renames ``Time step`` → ``time_step`` and
    ``num_txs_as receiver`` → ``num_txs_as_receiver``.
    """
    df = _read_csv(path)
    _validate_col_count(df, path, _WALLETS_FEATURES_NCOLS)
    _validate_first_cols(df, path, _WALLETS_FEATURES_FIRST_COLS)
    return _rename_headers(df)


def parse_wallets_classes(path: Path) -> pd.DataFrame:
    """Parse wallets_classes.csv (822,942 rows × 2 columns)."""
    return _read_csv(path, expected_cols=_WALLETS_CLASSES_COLS)


def parse_txs_edgelist(path: Path) -> pd.DataFrame:
    """Parse txs_edgelist.csv (234,355 rows × 2 columns)."""
    return _read_csv(path, expected_cols=_TXS_EDGELIST_COLS)


def parse_addr_tx_edgelist(path: Path) -> pd.DataFrame:
    """Parse AddrTx_edgelist.csv."""
    return _read_csv(path, expected_cols=_ADDR_TX_COLS)


def parse_tx_addr_edgelist(path: Path) -> pd.DataFrame:
    """Parse TxAddr_edgelist.csv."""
    return _read_csv(path, expected_cols=_TX_ADDR_COLS)


def parse_addr_addr_edgelist(path: Path) -> pd.DataFrame:
    """Parse AddrAddr_edgelist.csv."""
    return _read_csv(path, expected_cols=_ADDR_ADDR_COLS)
