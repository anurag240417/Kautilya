"""Tests for the data validator."""

import pandas as pd

from backend.ingestion.validator import (
    validate_txs_classes,
    validate_txs_features,
    validate_wallets_features,
)


def test_validate_valid_data():
    """Valid data -> is_valid=True, no errors."""
    df = pd.DataFrame({
        "txId": [1, 2, 3],
        "time_step": [1, 25, 49]
    })

    result = validate_txs_features(df)
    assert result.is_valid is True
    assert len(result.errors) == 0
    assert len(result.warnings) == 0

    # Check stats population
    assert result.stats["row_count"] == 3
    assert result.stats["time_step_min"] == 1
    assert result.stats["time_step_max"] == 49
    assert result.stats["null_txId_count"] == 0


def test_validate_null_primary_key():
    """Null primary key -> error in result."""
    df = pd.DataFrame({
        "txId": [1, None, 3],
        "class": [1, 2, 1]
    })

    result = validate_txs_classes(df)
    assert result.is_valid is False
    assert any("null values in primary key" in err for err in result.errors)


def test_validate_duplicate_primary_keys():
    """Duplicate primary keys -> warning in result."""
    df = pd.DataFrame({
        "txId": [1, 1, 2],
        "class": [1, 2, 1]
    })

    result = validate_txs_classes(df)
    # Check if duplicate pk is warning or error.
    # In implementation, _check_duplicate_primary_key adds a warning.
    assert result.is_valid is True
    assert len(result.warnings) == 1
    assert "duplicate values in 'txId'" in result.warnings[0]


def test_validate_out_of_range_time_step():
    """Out-of-range time_step (0 or 50) -> error."""
    df = pd.DataFrame({
        "txId": [1, 2],
        "time_step": [0, 50]
    })

    result = validate_txs_features(df)
    assert result.is_valid is False
    assert any("below 1" in err for err in result.errors)
    assert any("above 49" in err for err in result.errors)


def test_validate_invalid_class_value():
    """Invalid class value (4 or 0) -> error."""
    df = pd.DataFrame({
        "txId": [1, 2, 3],
        "class": [1, 0, 4]
    })

    result = validate_txs_classes(df)
    assert result.is_valid is False
    assert any("invalid values in 'class'" in err for err in result.errors)


def test_validate_missing_expected_column():
    """Missing expected column -> error."""
    df = pd.DataFrame({
        "txId": [1, 2],
        # "class" is missing
    })

    result = validate_txs_classes(df)
    assert result.is_valid is False
    assert any("Missing required columns" in err for err in result.errors)


def test_validate_wallets_features_duplicate_addresses():
    """Duplicate addresses in wallets_features are expected, result should be valid."""
    df = pd.DataFrame({
        "address": ["addr1", "addr1", "addr2"],
        "time_step": [1, 2, 1]
    })

    result = validate_wallets_features(df)
    assert result.is_valid is True
    assert len(result.errors) == 0
    assert result.stats["duplicate_address_rows"] == 1
    assert result.stats["unique_addresses"] == 2


def test_validate_stats_population():
    """Stats are populated correctly (row counts, class distribution)."""
    df = pd.DataFrame({
        "txId": [1, 2, 3, 4],
        "class": [1, 2, 2, 3]
    })

    result = validate_txs_classes(df)
    assert result.is_valid is True
    assert result.stats["row_count"] == 4

    dist = result.stats["class_distribution"]
    assert dist[1] == 1
    assert dist[2] == 2
    assert dist[3] == 1
