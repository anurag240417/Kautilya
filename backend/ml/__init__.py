"""Machine learning module.

Responsible for supervised classification, anomaly detection,
and clustering. Training and inference must remain separate.
Training must not execute during API requests.
"""

from .anomaly import AnomalyDetector
from .classifier import TransactionClassifier
from .dataset import load_transaction_dataset, prepare_ml_splits
from .evaluation import TemporalEvaluation, evaluate_temporal, summarize_evaluation
from .feature_registry import (
    ALL_TX_FEATURES,
    ALL_WALLET_FEATURES,
    ANONYMIZED_TX_FEATURES,
    BINARY_TARGET_MAP,
    FEATURE_SETS,
    FORBIDDEN_FEATURE_COLUMNS,
    GRAPH_FEATURES,
    INTERPRETABLE_TX_FEATURES,
    LABEL_ILLICIT,
    LABEL_LICIT,
    LABEL_UNKNOWN,
    NETWORK_FEATURES,
    TRAIN_LABELS,
    WALLET_SCALAR_FEATURES,
    WALLET_STATS_FEATURES,
)
from .inference import (
    MLInferencePipeline,
    find_latest_model,
    load_anomaly_detector,
    load_classifier,
)
from .leakage import LeakageAuditResult, audit_leakage
from .splitting import temporal_train_test_split

__all__ = [
    # Anomaly
    "AnomalyDetector",
    # Classifier
    "TransactionClassifier",
    # Inference
    "MLInferencePipeline",
    "find_latest_model",
    "load_classifier",
    "load_anomaly_detector",
    # Dataset
    "load_transaction_dataset",
    "prepare_ml_splits",
    # Evaluation
    "TemporalEvaluation",
    "evaluate_temporal",
    "summarize_evaluation",
    # Feature registry
    "ALL_TX_FEATURES",
    "ALL_WALLET_FEATURES",
    "ANONYMIZED_TX_FEATURES",
    "BINARY_TARGET_MAP",
    "FEATURE_SETS",
    "FORBIDDEN_FEATURE_COLUMNS",
    "GRAPH_FEATURES",
    "INTERPRETABLE_TX_FEATURES",
    "LABEL_ILLICIT",
    "LABEL_LICIT",
    "LABEL_UNKNOWN",
    "NETWORK_FEATURES",
    "TRAIN_LABELS",
    "WALLET_SCALAR_FEATURES",
    "WALLET_STATS_FEATURES",
    # Graph features
    "extract_graph_feature_df",
    "merge_graph_features",
    # Splitting
    "temporal_train_test_split",
    # Leakage
    "LeakageAuditResult",
    "audit_leakage",
]
