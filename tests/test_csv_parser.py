"""Tests for the CSV parser."""

import pandas as pd
import pytest

from backend.ingestion.csv_parser import (
    _TXS_CLASSES_COLS,
    _TXS_FEATURES_NCOLS,
    parse_txs_classes,
    parse_txs_features,
)
from backend.ingestion.errors import IngestionError


def test_parse_txs_features_valid(tmp_path):
    """Parse valid CSV -> correct DataFrame shape and types, headers renamed."""
    path = tmp_path / "txs_features.csv"

    # Create header
    header = (
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

    # Create one row of data
    row = [12345, 1] + [0.0] * 182

    df = pd.DataFrame([row], columns=header)
    df.to_csv(path, index=False)

    parsed = parse_txs_features(path)
    assert len(parsed) == 1
    assert len(parsed.columns) == _TXS_FEATURES_NCOLS

    # Check header renaming ("Time step" -> "time_step")
    assert "time_step" in parsed.columns
    assert "Time step" not in parsed.columns

    # Check types
    assert parsed["txId"].dtype == "int64"
    assert parsed["time_step"].dtype == "int64"


def test_parse_txs_features_missing_columns(tmp_path):
    """Parse CSV with missing columns -> raises IngestionError."""
    path = tmp_path / "txs_features.csv"

    # Create header with only 180 columns (missing 4)
    header = ["txId", "Time step"] + [f"Local_feature_{i}" for i in range(1, 179)]
    df = pd.DataFrame([[1, 1] + [0.0] * 178], columns=header)
    df.to_csv(path, index=False)

    with pytest.raises(IngestionError, match="Column count mismatch"):
        parse_txs_features(path)


def test_parse_txs_features_wrong_first_columns(tmp_path):
    """Parse CSV with wrong leading column names -> raises IngestionError."""
    path = tmp_path / "txs_features.csv"

    header = ["wrong_txId", "wrong_time_step"] + [f"Col_{i}" for i in range(1, 183)]
    df = pd.DataFrame([[1, 1] + [0.0] * 182], columns=header)
    df.to_csv(path, index=False)

    with pytest.raises(IngestionError, match="Leading columns mismatch"):
        parse_txs_features(path)


def test_parse_txs_classes_wrong_column_names(tmp_path):
    """Parse CSV with wrong column names (exact schema check) -> raises IngestionError."""
    path = tmp_path / "txs_classes.csv"

    df = pd.DataFrame({"id": [1], "label": [2]})
    df.to_csv(path, index=False)

    with pytest.raises(IngestionError, match="Schema mismatch"):
        parse_txs_classes(path)


def test_parse_empty_csv(tmp_path):
    """Parse empty CSV (header only) -> returns empty DataFrame."""
    path = tmp_path / "txs_classes.csv"

    # Empty dataframe, just headers
    df = pd.DataFrame(columns=_TXS_CLASSES_COLS)
    df.to_csv(path, index=False)

    parsed = parse_txs_classes(path)
    assert len(parsed) == 0
    assert list(parsed.columns) == _TXS_CLASSES_COLS


def test_file_not_found():
    """File not found -> raises IngestionError."""
    from pathlib import Path
    path = Path("/nonexistent/file.csv")

    with pytest.raises(IngestionError, match="File not found"):
        parse_txs_classes(path)
