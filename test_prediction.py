"""CLI tool for manually testing and inspecting trained ChainTrace ML models.

Usage:
    # Test a specific transaction by ID:
    ./.venv/bin/python test_prediction.py --txid 7134700

    # Pick a random transaction from the test set:
    ./.venv/bin/python test_prediction.py --random

    # Test with custom feature values:
    ./.venv/bin/python test_prediction.py --custom --total-btc 500.0 --fees 0.05
"""

import argparse
import random
import pandas as pd

from backend.config import get_settings
from backend.ml.anomaly import AnomalyDetector
from backend.ml.inference import (
    MLInferencePipeline,
    load_anomaly_detector,
    load_classifier,
)


def print_prediction_card(result, ground_truth=None, actual_features=None, anomaly_explanation=None):
    label_str = "ILLICIT (Suspicious)" if result.prediction == 1 else "LICIT (Normal)"
    color_prefix = "🚨 " if result.prediction == 1 else "✅ "

    print("\n" + "=" * 65)
    print(f"             TRANSACTION FORENSIC ML EVALUATION")
    print("=" * 65)
    print(f" Transaction ID   : {result.entity_id}")
    print(f" Active Model     : {result.model_version} ({result.feature_set})")
    print("-" * 65)
    print(f" ML Prediction    : {color_prefix}{label_str}")
    print(f" Model Confidence : {result.probability * 100:.2f}%")

    if ground_truth is not None:
        gt_str = "Illicit (1)" if ground_truth == 1 else "Licit (2)" if ground_truth == 2 else "Unknown (3)"
        match = "MATCH (Correct)" if (result.prediction == 1 and ground_truth == 1) or (result.prediction == 0 and ground_truth == 2) else "MISMATCH"
        print(f" Ground Truth     : {gt_str}  [{match}]")

    print("-" * 65)
    if result.anomaly_score is not None:
        print(f" Anomaly Score    : {result.anomaly_score:.4f} / 1.0000")
        if anomaly_explanation:
            print(f" Anomaly Context  : {anomaly_explanation}")

    if actual_features:
        print("-" * 65)
        print(" Key Interpretable Features:")
        for k, v in list(actual_features.items())[:8]:
            if isinstance(v, float):
                print(f"   • {k:<24}: {v:.6f}")
            else:
                print(f"   • {k:<24}: {v}")
    print("=" * 65 + "\n")


def main():
    parser = argparse.ArgumentParser(description="ChainTrace Model Testing Tool")
    parser.add_argument("--txid", type=int, help="Transaction ID from dataset to test")
    parser.add_argument("--model", choices=["M1", "M2", "M3"], default="M1", help="Model version to test (default M1)")
    parser.add_argument("--random", action="store_true", help="Pick a random transaction from the test set")
    parser.add_argument("--class-type", choices=["illicit", "licit"], help="Filter random selection by class")
    parser.add_argument("--custom", action="store_true", help="Test custom synthetic feature inputs")
    parser.add_argument("--total-btc", type=float, default=1.0, help="Custom total BTC")
    parser.add_argument("--fees", type=float, default=0.0005, help="Custom transaction fees")
    args = parser.parse_args()

    settings = get_settings()

    # 1. Load trained models
    print("Loading models from backend/models/ ...")
    clf = load_classifier(feature_set_name=args.model)
    anom = load_anomaly_detector()
    pipeline = MLInferencePipeline(classifier=clf, anomaly_detector=anom)

    # 2. Handle Custom Input
    if args.custom:
        custom_features = {
            "total_BTC": args.total_btc,
            "fees": args.fees,
            "in_txs_degree": 1.0,
            "out_txs_degree": 2.0,
        }
        res = pipeline.predict_transaction(custom_features, txid="custom_test_001")
        anom_desc = AnomalyDetector.explain_score(res.anomaly_score) if res.anomaly_score else None
        print_prediction_card(res, actual_features=custom_features, anomaly_explanation=anom_desc)
        return

    # 3. Load dataset rows for lookup
    print("Loading dataset for transaction lookup...")
    features_df = pd.read_csv(settings.txs_features_path)
    classes_df = pd.read_csv(settings.txs_classes_path)
    merged = features_df.merge(classes_df, on="txId")

    target_row = None
    if args.txid:
        matches = merged[merged["txId"] == args.txid]
        if len(matches) == 0:
            print(f"Transaction ID {args.txid} not found in dataset.")
            return
        target_row = matches.iloc[0]
    elif args.random:
        # Pick from test set (Time step >= 35)
        test_pool = merged[merged["Time step"] >= 35]
        if args.class_type == "illicit":
            test_pool = test_pool[test_pool["class"] == 1]
        elif args.class_type == "licit":
            test_pool = test_pool[test_pool["class"] == 2]

        target_row = test_pool.sample(1, random_state=random.randint(1, 999999)).iloc[0]
    else:
        # Default: sample an illicit transaction from test set to showcase detection
        test_illicit = merged[(merged["Time step"] >= 35) & (merged["class"] == 1)]
        target_row = test_illicit.sample(1, random_state=42).iloc[0]

    txid = int(target_row["txId"])
    ground_truth = int(target_row["class"])

    # Prepare feature dict
    feat_dict = target_row.to_dict()

    # Predict
    res = pipeline.predict_transaction(feat_dict, txid=txid)
    anom_desc = AnomalyDetector.explain_score(res.anomaly_score) if res.anomaly_score else None

    # Filter top interpretable features to show
    interp_keys = [
        "Time step", "total_BTC", "fees", "size", "in_txs_degree", "out_txs_degree",
        "num_input_addresses", "num_output_addresses", "in_BTC_total", "out_BTC_total"
    ]
    disp_feats = {k: feat_dict[k] for k in interp_keys if k in feat_dict}

    print_prediction_card(res, ground_truth=ground_truth, actual_features=disp_feats, anomaly_explanation=anom_desc)


if __name__ == "__main__":
    main()
