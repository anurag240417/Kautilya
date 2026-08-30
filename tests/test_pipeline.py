"""Tests for the unified ingestion pipeline."""

from unittest.mock import patch

import pandas as pd
import pytest

from backend.ingestion.errors import IngestionError
from backend.ingestion.pipeline import run_ingestion_pipeline
from backend.ingestion.validator import ValidationResult


@pytest.fixture
def mock_dataset_dir(tmp_path):
    """Create a mock dataset directory with all required files."""
    files = [
        "txs_features.csv",
        "txs_classes.csv",
        "wallets_features.csv",
        "wallets_classes.csv",
        "txs_edgelist.csv",
        "AddrTx_edgelist.csv",
        "TxAddr_edgelist.csv",
        "AddrAddr_edgelist.csv",
    ]
    for filename in files:
        (tmp_path / filename).touch()
    return tmp_path


@patch("backend.ingestion.pipeline.parse_addr_addr_edgelist")
@patch("backend.ingestion.pipeline.parse_tx_addr_edgelist")
@patch("backend.ingestion.pipeline.parse_addr_tx_edgelist")
@patch("backend.ingestion.pipeline.parse_txs_edgelist")
@patch("backend.ingestion.pipeline.parse_wallets_classes")
@patch("backend.ingestion.pipeline.parse_wallets_features")
@patch("backend.ingestion.pipeline.parse_txs_classes")
@patch("backend.ingestion.pipeline.parse_txs_features")
@patch("backend.ingestion.pipeline.validate_edgelist")
@patch("backend.ingestion.pipeline.validate_wallets_classes")
@patch("backend.ingestion.pipeline.validate_wallets_features")
@patch("backend.ingestion.pipeline.validate_txs_classes")
@patch("backend.ingestion.pipeline.validate_txs_features")
def test_run_ingestion_pipeline_success(
    mock_val_txs_feat,
    mock_val_txs_class,
    mock_val_wallets_feat,
    mock_val_wallets_class,
    mock_val_edge,
    mock_parse_txs_feat,
    mock_parse_txs_class,
    mock_parse_wallets_feat,
    mock_parse_wallets_class,
    mock_parse_txs_edge,
    mock_parse_addr_tx,
    mock_parse_tx_addr,
    mock_parse_addr_addr,
    mock_dataset_dir,
):
    """Test successful execution of the ingestion pipeline."""
    # Setup mocks
    mock_df = pd.DataFrame()
    for mock_parser in [
        mock_parse_txs_feat,
        mock_parse_txs_class,
        mock_parse_wallets_feat,
        mock_parse_wallets_class,
        mock_parse_txs_edge,
        mock_parse_addr_tx,
        mock_parse_tx_addr,
        mock_parse_addr_addr,
    ]:
        mock_parser.return_value = mock_df

    for mock_validator in [
        mock_val_txs_feat,
        mock_val_txs_class,
        mock_val_wallets_feat,
        mock_val_wallets_class,
        mock_val_edge,
    ]:
        mock_validator.return_value = ValidationResult(is_valid=True)

    result = run_ingestion_pipeline(mock_dataset_dir)

    assert isinstance(result, dict)
    assert len(result) == 8
    assert "txs_features" in result
    assert "addr_addr_edgelist" in result


@patch("backend.ingestion.pipeline.parse_txs_features")
@patch("backend.ingestion.pipeline.parse_txs_classes")
@patch("backend.ingestion.pipeline.parse_wallets_features")
@patch("backend.ingestion.pipeline.parse_wallets_classes")
@patch("backend.ingestion.pipeline.parse_txs_edgelist")
@patch("backend.ingestion.pipeline.parse_addr_tx_edgelist")
@patch("backend.ingestion.pipeline.parse_tx_addr_edgelist")
@patch("backend.ingestion.pipeline.parse_addr_addr_edgelist")
@patch("backend.ingestion.pipeline.validate_txs_features")
@patch("backend.ingestion.pipeline.validate_txs_classes")
@patch("backend.ingestion.pipeline.validate_wallets_features")
@patch("backend.ingestion.pipeline.validate_wallets_classes")
@patch("backend.ingestion.pipeline.validate_edgelist")
def test_run_ingestion_pipeline_validation_error(
    mock_val_edge,
    mock_val_wallets_class,
    mock_val_wallets_feat,
    mock_val_txs_class,
    mock_val_txs_feat,
    mock_parse_addr_addr,
    mock_parse_tx_addr,
    mock_parse_addr_tx,
    mock_parse_txs_edge,
    mock_parse_wallets_class,
    mock_parse_wallets_feat,
    mock_parse_txs_class,
    mock_parse_txs_feat,
    mock_dataset_dir,
):
    """Test that pipeline raises IngestionError if validation fails."""
    # Parsers return mock data
    for parser in [
        mock_parse_txs_feat,
        mock_parse_txs_class,
        mock_parse_wallets_feat,
        mock_parse_wallets_class,
        mock_parse_txs_edge,
        mock_parse_addr_tx,
        mock_parse_tx_addr,
        mock_parse_addr_addr,
    ]:
        parser.return_value = pd.DataFrame()

    # Validators return valid, except one
    mock_val_txs_feat.return_value = ValidationResult(is_valid=True)
    mock_val_txs_class.return_value = ValidationResult(is_valid=False, errors=["Mock error 1"])
    mock_val_wallets_feat.return_value = ValidationResult(is_valid=True)
    mock_val_wallets_class.return_value = ValidationResult(is_valid=False, errors=["Mock error 2"])
    mock_val_edge.return_value = ValidationResult(is_valid=True)

    with pytest.raises(IngestionError, match="Validation failed with 2 errors"):
        run_ingestion_pipeline(mock_dataset_dir)
