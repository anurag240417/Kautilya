"""Tests for normalization pipelines."""

from datetime import UTC, datetime

import pandas as pd

from backend.domain.graph import AddrAddrEdge, AddrTxEdge, TxAddrEdge, TxTxEdge
from backend.domain.network import NetworkObservation
from backend.domain.transaction import Transaction, TransactionFeatures
from backend.domain.wallet import Wallet
from backend.normalization.graph import (
    normalize_addr_addr_edges,
    normalize_addr_tx_edges,
    normalize_tx_addr_edges,
    normalize_tx_tx_edges,
)
from backend.normalization.network import normalize_network_observations
from backend.normalization.transaction import normalize_transactions
from backend.normalization.wallet import normalize_wallets


def test_normalize_transactions():
    """Test mapping of transaction DataFrames to Transaction and TransactionFeatures."""
    # Mock features dataframe
    row_features = {"txId": 123, "time_step": 1, "total_BTC": 50.0}
    for i in range(1, 94):
        row_features[f"Local_feature_{i}"] = i * 0.1
    for i in range(1, 73):
        row_features[f"Aggregate_feature_{i}"] = i * 0.2

    df_features = pd.DataFrame([row_features])

    # Mock classes dataframe
    df_classes = pd.DataFrame([{"txId": 123, "class": 2}])

    # Run normalizer
    gen = normalize_transactions(df_features, df_classes)
    tx, feats = next(gen)

    # Verify mapping
    assert isinstance(tx, Transaction)
    assert tx.txid == 123
    assert tx.time_step == 1
    assert tx.label == 2
    assert tx.total_btc == 50.0

    assert isinstance(feats, TransactionFeatures)
    assert feats.txid == 123
    assert len(feats.local_features) == 93
    assert len(feats.aggregate_features) == 72
    assert feats.local_features[0] == 0.1


def test_normalize_wallets():
    """Test mapping of wallet DataFrames to Wallet domain objects."""
    # Mock features dataframe
    df_features = pd.DataFrame([{
        "address": "1abc",
        "time_step": 2,
        "num_txs_as_sender": 5.0,
        "btc_transacted_total": 100.0,
        "btc_transacted_mean": 20.0
    }])

    # Mock classes dataframe
    df_classes = pd.DataFrame([{"address": "1abc", "class": 1}])

    # Run normalizer
    gen = normalize_wallets(df_features, df_classes)
    wallet = next(gen)

    # Verify mapping
    assert isinstance(wallet, Wallet)
    assert wallet.address == "1abc"
    assert wallet.time_step == 2
    assert wallet.label == 1
    assert wallet.num_txs_as_sender == 5.0

    # Verify StatsSummary mapping
    assert wallet.btc_transacted is not None
    assert wallet.btc_transacted.total == 100.0
    assert wallet.btc_transacted.mean == 20.0
    assert wallet.btc_transacted.min is None


def test_normalize_network_observations():
    """Test mapping of network DataFrames to NetworkObservation domain objects."""
    # Mock dataframe
    now = datetime.now(tz=UTC)
    df = pd.DataFrame([{
        "txid": 456,
        "src_ip": "192.168.1.1",
        "dst_ip": "10.0.0.1",
        "src_port": 8333,
        "dst_port": 8333,
        "timestamp": now,
        "country": "US"
    }])

    # Run normalizer
    gen = normalize_network_observations(df)
    obs = next(gen)

    # Verify mapping
    assert isinstance(obs, NetworkObservation)
    assert obs.txid == 456
    assert obs.src_ip == "192.168.1.1"
    assert obs.is_synthetic is True
    assert obs.country == "US"


def test_normalize_tx_tx_edges():
    """Test mapping of txs_edgelist DataFrames."""
    df = pd.DataFrame([{"txId1": 1, "txId2": 2}])
    edge = next(normalize_tx_tx_edges(df))
    assert isinstance(edge, TxTxEdge)
    assert edge.source_txid == 1
    assert edge.target_txid == 2
    assert edge.is_synthetic is False


def test_normalize_addr_tx_edges():
    """Test mapping of AddrTx_edgelist DataFrames."""
    df = pd.DataFrame([{"input_address": "addr1", "txId": 99}])
    edge = next(normalize_addr_tx_edges(df))
    assert isinstance(edge, AddrTxEdge)
    assert edge.input_address == "addr1"
    assert edge.txid == 99
    assert edge.is_synthetic is False


def test_normalize_tx_addr_edges():
    """Test mapping of TxAddr_edgelist DataFrames."""
    df = pd.DataFrame([{"txId": 100, "output_address": "addr2"}])
    edge = next(normalize_tx_addr_edges(df))
    assert isinstance(edge, TxAddrEdge)
    assert edge.txid == 100
    assert edge.output_address == "addr2"
    assert edge.is_synthetic is False


def test_normalize_addr_addr_edges():
    """Test mapping of AddrAddr_edgelist DataFrames."""
    df = pd.DataFrame([{"input_address": "addr1", "output_address": "addr2"}])
    edge = next(normalize_addr_addr_edges(df))
    assert isinstance(edge, AddrAddrEdge)
    assert edge.input_address == "addr1"
    assert edge.output_address == "addr2"
    assert edge.is_synthetic is False
