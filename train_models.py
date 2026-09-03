"""End-to-end model training runner.

Executes dataset cleaning, temporal train/test splitting, leakage auditing,
model training, temporal evaluation, and artifact serialization for:
    - M1: Blockchain features only (182 features)
    - M2: Blockchain + Graph features (190 features)
    - M3: Blockchain + Graph + Synthetic Network features (199 features)
    - Unsupervised Anomaly Detector (Isolation Forest on 165 anonymized features)

Saves all trained .joblib artifacts and _metadata.json files to backend/models/.
See ARCHITECTURE.md §ml/, CONTEXT.md §Experimental Progression, and DATASET.md.
"""

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from backend.config import configure_logging, get_settings
from backend.domain.graph import TxTxEdge
from backend.graph.builder import ChainTraceGraph
from backend.ml.anomaly import AnomalyDetector
from backend.ml.classifier import TransactionClassifier
from backend.ml.dataset import load_transaction_dataset, prepare_ml_splits
from backend.ml.evaluation import evaluate_temporal, summarize_evaluation
from backend.ml.feature_registry import (
    ALL_TX_FEATURES,
    ANONYMIZED_TX_FEATURES,
    FEATURE_SETS,
    GRAPH_FEATURES,
    NETWORK_FEATURES,
)
from backend.ml.graph_features import extract_graph_feature_df, merge_graph_features

logger = logging.getLogger("train_models")


def build_network_features_df(observations_path: Path) -> pd.DataFrame:
    """Aggregate per-txId network features from synthetic network observations.

    Args:
        observations_path: Path to network_observations.csv.

    Returns:
        DataFrame with txId and 9 network feature columns.
    """
    logger.info("Loading network observations from %s", observations_path)
    obs = pd.read_csv(observations_path)

    # Standardize column name
    if "txid" in obs.columns:
        obs = obs.rename(columns={"txid": "txId"})

    # Convert timestamps to numeric milliseconds relative to min timestamp per txId
    obs["dt"] = pd.to_datetime(obs["timestamp"], format="ISO8601", utc=True)

    grouped = obs.groupby("txId")
    rows = []

    for txid, group in grouped:
        t_min = group["dt"].min()
        delays_ms = (group["dt"] - t_min).dt.total_seconds() * 1000.0

        scenario_ids = set(group["scenario_id"].dropna().unique())
        uses_tor = 1 if "tor_origin" in scenario_ids else 0
        uses_vpn = 1 if "vpn_origin" in scenario_ids else 0

        rows.append({
            "txId": txid,
            "num_observations": len(group),
            "num_unique_src_ips": group["src_ip"].nunique(),
            "num_unique_dst_ips": group["dst_ip"].nunique(),
            "num_unique_countries": group["country"].nunique(),
            "num_unique_asns": group["asn"].nunique(),
            "mean_propagation_delay_ms": float(delays_ms.mean()),
            "max_propagation_delay_ms": float(delays_ms.max()),
            "uses_tor": uses_tor,
            "uses_vpn": uses_vpn,
        })

    net_df = pd.DataFrame(rows, columns=["txId"] + NETWORK_FEATURES)
    logger.info("Aggregated network features for %d transactions", len(net_df))
    return net_df


def merge_network_features(df: pd.DataFrame, net_df: pd.DataFrame) -> pd.DataFrame:
    """Merge network features onto transaction DataFrame, filling missing with 0.

    Args:
        df: Transaction DataFrame with txId.
        net_df: Network features DataFrame.

    Returns:
        DataFrame augmented with NETWORK_FEATURES.
    """
    merged = df.merge(net_df, on="txId", how="left")
    for col in NETWORK_FEATURES:
        merged[col] = merged[col].fillna(0.0)
    return merged


def compute_medians_dict(train_df: pd.DataFrame, feature_cols: list[str]) -> dict[str, float]:
    """Compute median values on training set for missing value imputation."""
    medians = train_df[feature_cols].median().to_dict()
    return {k: float(v) for k, v in medians.items()}


def print_comparison_table(results: list[dict]) -> None:
    """Print a clean CLI summary table comparing M1, M2, and M3."""
    header = (
        f"{'Model':<8} | {'Features':<8} | {'Accuracy':<9} | {'Precision':<9} | "
        f"{'Recall':<9} | {'F1-Score':<9} | {'ROC-AUC':<9}"
    )
    separator = "-" * len(header)
    print("\n" + separator)
    print("           MODEL PROGRESSION COMPARISON SUMMARY")
    print(separator)
    print(header)
    print(separator)

    for r in results:
        auc_str = f"{r['roc_auc']:.4f}" if r["roc_auc"] is not None else "N/A"
        print(
            f"{r['model']:<8} | {r['n_features']:<8} | {r['accuracy']:<9.4f} | "
            f"{r['precision']:<9.4f} | {r['recall']:<9.4f} | {r['f1']:<9.4f} | {auc_str:<9}"
        )
    print(separator + "\n")


def main() -> None:
    configure_logging("INFO")
    settings = get_settings()
    models_dir = settings.models_dir
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("       ChainTrace Machine Learning Model Training Pipeline")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. Ingestion & Pre-training Dataset Cleaning
    # ------------------------------------------------------------------
    logger.info("Stage 1: Loading and validating Elliptic++ transaction dataset...")
    t0 = time.time()
    base_df = load_transaction_dataset(settings)
    logger.info("Loaded %d transactions in %.2fs", len(base_df), time.time() - t0)

    # Check and log missing value audit
    supervised_mask = base_df["label"].isin([1, 2])
    nan_cols = base_df.loc[supervised_mask, ALL_TX_FEATURES].isna().sum()
    cols_with_nan = nan_cols[nan_cols > 0]
    logger.info("Dataset cleaning audit: %d columns with NaNs in supervised pool", len(cols_with_nan))
    if len(cols_with_nan) > 0:
        logger.info("Columns with NaNs: %s", list(cols_with_nan.index))

    results = []

    # ------------------------------------------------------------------
    # 2. Train Model M1 (Blockchain Only)
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    logger.info("Stage 2: Preparing and training Model M1 (Blockchain features only)...")
    m1_features = FEATURE_SETS["M1"]

    X_train_m1, y_train_m1, X_test_m1, y_test_m1, config_m1 = prepare_ml_splits(
        df=base_df,
        feature_cols=m1_features,
        train_cutoff=34,
        experiment_id="m1_blockchain_baseline",
        feature_configuration="M1",
        random_seed=42,
    )

    # Extract test time steps for temporal evaluation
    test_supervised_df = base_df[
        (base_df["time_step"] > 34) & (base_df["label"].isin([1, 2]))
    ]
    test_time_steps = test_supervised_df["time_step"].values

    # Pre-compute training medians for imputation persistence
    train_supervised_m1 = base_df[
        (base_df["time_step"] <= 34) & (base_df["label"].isin([1, 2]))
    ]
    medians_m1 = compute_medians_dict(train_supervised_m1, m1_features)

    clf_m1 = TransactionClassifier(
        feature_set_name="M1",
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        feature_medians=medians_m1,
    )
    clf_m1.train(X_train_m1, y_train_m1, feature_names=m1_features)

    # Temporal evaluation
    eval_m1 = evaluate_temporal(clf_m1, X_test_m1, y_test_m1, test_time_steps)
    test_metrics_m1 = eval_m1.overall_metrics
    clf_m1.save(models_dir, experiment_config=config_m1, test_metrics=test_metrics_m1)

    results.append({
        "model": "M1",
        "n_features": len(m1_features),
        "accuracy": test_metrics_m1["accuracy"],
        "precision": test_metrics_m1["precision"],
        "recall": test_metrics_m1["recall"],
        "f1": test_metrics_m1["f1"],
        "roc_auc": test_metrics_m1.get("roc_auc"),
    })
    logger.info("Model M1 trained & saved. Overall F1=%.4f, AUC=%s", test_metrics_m1["f1"], test_metrics_m1.get("roc_auc"))

    # ------------------------------------------------------------------
    # 3. Train Model M2 (Blockchain + Graph Features)
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    logger.info("Stage 3: Building graph and extracting features for Model M2...")

    # Load tx-to-tx edgelist
    edgelist_df = pd.read_csv(settings.txs_edgelist_path)
    logger.info("Loaded %d transaction edges from %s", len(edgelist_df), settings.txs_edgelist_path.name)

    graph = ChainTraceGraph()
    tx_edges = (
        TxTxEdge(source_txid=row[0], target_txid=row[1])
        for row in edgelist_df[["txId1", "txId2"]].itertuples(index=False)
    )
    graph.add_tx_tx_edges(tx_edges)
    logger.info("Built transaction graph: %d nodes, %d edges", graph.node_count, graph.edge_count)

    # Approximate betweenness centrality (k=500) for efficiency on large graph
    graph_feat_df = extract_graph_feature_df(graph, betweenness_k=500)
    m2_df = merge_graph_features(base_df, graph_feat_df)

    m2_features = FEATURE_SETS["M2"]
    X_train_m2, y_train_m2, X_test_m2, y_test_m2, config_m2 = prepare_ml_splits(
        df=m2_df,
        feature_cols=m2_features,
        train_cutoff=34,
        experiment_id="m2_blockchain_graph",
        feature_configuration="M2",
        random_seed=42,
    )

    train_supervised_m2 = m2_df[
        (m2_df["time_step"] <= 34) & (m2_df["label"].isin([1, 2]))
    ]
    medians_m2 = compute_medians_dict(train_supervised_m2, m2_features)

    clf_m2 = TransactionClassifier(
        feature_set_name="M2",
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
        feature_medians=medians_m2,
    )
    clf_m2.train(X_train_m2, y_train_m2, feature_names=m2_features)

    eval_m2 = evaluate_temporal(clf_m2, X_test_m2, y_test_m2, test_time_steps)
    test_metrics_m2 = eval_m2.overall_metrics
    clf_m2.save(models_dir, experiment_config=config_m2, test_metrics=test_metrics_m2)

    results.append({
        "model": "M2",
        "n_features": len(m2_features),
        "accuracy": test_metrics_m2["accuracy"],
        "precision": test_metrics_m2["precision"],
        "recall": test_metrics_m2["recall"],
        "f1": test_metrics_m2["f1"],
        "roc_auc": test_metrics_m2.get("roc_auc"),
    })
    logger.info("Model M2 trained & saved. Overall F1=%.4f, AUC=%s", test_metrics_m2["f1"], test_metrics_m2.get("roc_auc"))

    # ------------------------------------------------------------------
    # 4. Train Model M3 (Blockchain + Graph + Network Features)
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    logger.info("Stage 4: Integrating synthetic network features for Model M3...")

    net_obs_path = Path("generated_network_data/network_observations.csv")
    if net_obs_path.exists():
        net_df = build_network_features_df(net_obs_path)
        m3_df = merge_network_features(m2_df, net_df)

        m3_features = FEATURE_SETS["M3"]
        X_train_m3, y_train_m3, X_test_m3, y_test_m3, config_m3 = prepare_ml_splits(
            df=m3_df,
            feature_cols=m3_features,
            train_cutoff=34,
            experiment_id="m3_blockchain_graph_network",
            feature_configuration="M3",
            random_seed=42,
        )

        train_supervised_m3 = m3_df[
            (m3_df["time_step"] <= 34) & (m3_df["label"].isin([1, 2]))
        ]
        medians_m3 = compute_medians_dict(train_supervised_m3, m3_features)

        clf_m3 = TransactionClassifier(
            feature_set_name="M3",
            n_estimators=200,
            random_state=42,
            class_weight="balanced",
            feature_medians=medians_m3,
        )
        clf_m3.train(X_train_m3, y_train_m3, feature_names=m3_features)

        eval_m3 = evaluate_temporal(clf_m3, X_test_m3, y_test_m3, test_time_steps)
        test_metrics_m3 = eval_m3.overall_metrics
        clf_m3.save(models_dir, experiment_config=config_m3, test_metrics=test_metrics_m3)

        results.append({
            "model": "M3",
            "n_features": len(m3_features),
            "accuracy": test_metrics_m3["accuracy"],
            "precision": test_metrics_m3["precision"],
            "recall": test_metrics_m3["recall"],
            "f1": test_metrics_m3["f1"],
            "roc_auc": test_metrics_m3.get("roc_auc"),
        })
        logger.info("Model M3 trained & saved. Overall F1=%.4f, AUC=%s", test_metrics_m3["f1"], test_metrics_m3.get("roc_auc"))
    else:
        logger.warning("network_observations.csv not found at %s. Skipping M3.", net_obs_path)

    # ------------------------------------------------------------------
    # 5. Train Unsupervised Anomaly Detector (Isolation Forest)
    # ------------------------------------------------------------------
    print("\n" + "-" * 70)
    logger.info("Stage 5: Training Unsupervised Anomaly Detector (Isolation Forest)...")
    anom_features = ANONYMIZED_TX_FEATURES  # 165 columns

    # Fit exclusively on training partition (time_step <= 34)
    train_anom_df = base_df[base_df["time_step"] <= 34][anom_features].copy()
    medians_anom = compute_medians_dict(train_anom_df, anom_features)
    train_anom_df = train_anom_df.fillna(medians_anom)
    X_train_anom = train_anom_df.values.astype(np.float64)

    anom_detector = AnomalyDetector(
        contamination=0.10,
        n_estimators=200,
        random_state=42,
        feature_names=anom_features,
        feature_medians=medians_anom,
    )
    anom_detector.fit(X_train_anom)
    anom_detector.save(models_dir)
    logger.info("Anomaly Detector trained on %d samples and saved.", len(X_train_anom))

    # ------------------------------------------------------------------
    # 6. Final Summary Report
    # ------------------------------------------------------------------
    print_comparison_table(results)
    print("All models successfully trained and serialized to:", models_dir.resolve())


if __name__ == "__main__":
    main()
