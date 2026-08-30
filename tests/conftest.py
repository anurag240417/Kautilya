"""Shared pytest fixtures for ChainTrace tests."""

from datetime import UTC, datetime

import pytest

from backend.domain import (
    AddrAddrEdge,
    AddrTxEdge,
    EntityClass,
    Evidence,
    EvidenceType,
    ExperimentConfig,
    GeneratorConfig,
    MLScore,
    NetworkObservation,
    RiskScore,
    ScriptType,
    StatsSummary,
    Transaction,
    TransactionFeatures,
    TxAddrEdge,
    TxTxEdge,
    Wallet,
)


@pytest.fixture
def sample_transaction():
    """A minimal valid Transaction instance."""
    return Transaction(
        txid=3321,
        time_step=1,
        label=EntityClass.UNKNOWN,
        total_btc=0.5,
        fees=0.0001,
    )


@pytest.fixture
def sample_transaction_features():
    """A minimal valid TransactionFeatures instance."""
    return TransactionFeatures(
        txid=3321,
        local_features=[0.0] * 93,
        aggregate_features=[0.0] * 72,
    )


@pytest.fixture
def sample_wallet():
    """A minimal valid Wallet instance."""
    return Wallet(
        address="14YRXHHof4BY1TVxN5FqYPcEdpmXiYT78a",
        time_step=25,
        label=EntityClass.LICIT,
        num_txs_as_sender=5.0,
        num_txs_as_receiver=10.0,
        btc_transacted=StatsSummary(
            total=1.5, min=0.01, max=0.8, mean=0.3, median=0.25,
        ),
    )


@pytest.fixture
def sample_network_observation():
    """A minimal valid NetworkObservation instance."""
    return NetworkObservation(
        txid=3321,
        src_ip="192.168.1.1",
        dst_ip="10.0.0.1",
        src_port=8333,
        dst_port=8333,
        protocol="TCP",
        timestamp=datetime(2024, 1, 15, 14, 32, 0, tzinfo=UTC),
        asn=15169,
        country="US",
        script_type=ScriptType.P2PKH,
        generation_run_id="run_001",
        scenario_id="normal_propagation",
        generator_version="0.1.0",
    )


@pytest.fixture
def sample_edges():
    """Sample edge instances for all four native edgelists."""
    return {
        "tx_tx": TxTxEdge(source_txid=230425980, target_txid=5530458),
        "addr_tx": AddrTxEdge(
            input_address="14YRXHHof4BY1TVxN5FqYPcEdpmXiYT78a",
            txid=230325127,
        ),
        "tx_addr": TxAddrEdge(
            txid=230325127,
            output_address="1GASxu5nMntiRKdVtTVRvEbP965G51bhHH",
        ),
        "addr_addr": AddrAddrEdge(
            input_address="14YRXHHof4BY1TVxN5FqYPcEdpmXiYT78a",
            output_address="1GASxu5nMntiRKdVtTVRvEbP965G51bhHH",
        ),
    }


@pytest.fixture
def sample_experiment_config():
    """A valid ExperimentConfig for temporal split."""
    return ExperimentConfig(
        experiment_id="exp_001",
        dataset_version="elliptic_pp_v1",
        feature_configuration="M1_blockchain_only",
        split_strategy="temporal",
        random_seed=42,
        train_time_steps=list(range(1, 35)),
        test_time_steps=list(range(35, 50)),
    )


@pytest.fixture
def sample_generator_config():
    """A valid GeneratorConfig."""
    return GeneratorConfig(
        random_seed=42,
        generator_version="0.1.0",
        scenario_configs={"normal": {"weight": 0.8}, "anomalous": {"weight": 0.2}},
        node_pool_size=1000,
    )


@pytest.fixture
def sample_ml_score():
    """A minimal valid MLScore."""
    return MLScore(
        entity_id="3321",
        entity_type="transaction",
        prediction=1,
        probability=0.87,
        model_version="rf_v1",
        feature_set="M1_blockchain_only",
    )


@pytest.fixture
def sample_risk_score():
    """A minimal valid RiskScore."""
    return RiskScore(
        entity_id="3321",
        entity_type="transaction",
        score=0.75,
        behavioral_signal=0.8,
        graph_signal=0.6,
        anomaly_signal=0.9,
        explanation="High transaction volume with unusual graph connectivity",
    )


@pytest.fixture
def sample_evidence():
    """A minimal valid Evidence record."""
    return Evidence(
        evidence_type=EvidenceType.OBSERVATION,
        source_entity_id="3321",
        source_entity_type="transaction",
        description="Transaction occurred at time_step 1",
        confidence=1.0,
        is_synthetic=False,
        provenance="elliptic_pp",
    )
