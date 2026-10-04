"""Run the detector benchmark and write reports/benchmark_results.json + BENCHMARK.md.

Real run (needs Elliptic++ under KAUTILYA_DATASET_DIR):

    python -m scripts.run_benchmark

Pipeline smoke test on random data (numbers are meaningless):

    python -m scripts.run_benchmark --smoke-test
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from backend.config import configure_logging, get_settings
from backend.ml.benchmark import robustness_sweep, rolling_origin_evaluation, run_benchmark
from backend.ml.benchmark_report import render_markdown
from backend.ml.dataset import load_transaction_dataset
from backend.ml.feature_registry import ALL_TX_FEATURES, INTERPRETABLE_TX_FEATURES
from backend.ml.splitting import temporal_train_test_split

logger = logging.getLogger("run_benchmark")


def _smoke_dataset(n_per_step: int = 120, seed: int = 0) -> pd.DataFrame:
    """Random stand-in with the Elliptic++ columns the benchmark touches."""
    rng = np.random.default_rng(seed)
    rows = []
    for step in range(1, 50):
        label = np.where(rng.random(n_per_step) < 0.1, 1, 2)
        df = pd.DataFrame(
            rng.normal(size=(n_per_step, len(ALL_TX_FEATURES))), columns=ALL_TX_FEATURES
        )
        df["in_txs_degree"] = rng.integers(0, 8, n_per_step)
        df["out_txs_degree"] = rng.integers(0, 8, n_per_step)
        df["num_output_addresses"] = rng.integers(1, 15, n_per_step)
        df["num_input_addresses"] = rng.integers(1, 15, n_per_step)
        df["total_BTC"] = rng.exponential(30, n_per_step)
        df["out_BTC_max"] = rng.exponential(10, n_per_step)
        df["out_BTC_total"] = df["out_BTC_max"] + rng.exponential(5, n_per_step)
        df.loc[label == 1, ALL_TX_FEATURES[:5]] += 1.0  # weak learnable signal
        df["txId"] = np.arange(n_per_step) + step * 10_000
        df["time_step"] = step
        df["label"] = label
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--smoke-test", action="store_true", help="use random data")
    ap.add_argument("--train-cutoff", type=int, default=34)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--n-estimators", type=int, default=200)
    ap.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = ap.parse_args()

    configure_logging("INFO")
    df = _smoke_dataset() if args.smoke_test else load_transaction_dataset(get_settings())
    feature_cols = list(ALL_TX_FEATURES)

    train_df, test_df, _ = temporal_train_test_split(df, train_cutoff=args.train_cutoff)
    logger.info("train=%d test=%d", len(train_df), len(test_df))

    bench, fitted = run_benchmark(
        train_df,
        test_df,
        feature_cols,
        n_boot=args.n_boot,
        n_estimators=args.n_estimators,
        seed=args.seed,
    )

    medians = train_df[feature_cols].median()
    X_train = train_df[feature_cols].fillna(medians).to_numpy(dtype=np.float64)
    X_test = test_df[feature_cols].fillna(medians).to_numpy(dtype=np.float64)
    robustness = robustness_sweep(
        fitted, test_df, X_test, test_df["target"].to_numpy(), X_train, seed=args.seed
    )

    all_labelled = pd.concat([train_df, test_df])
    rolling = rolling_origin_evaluation(all_labelled, feature_cols, seed=args.seed)

    results = {
        "meta": {
            "dataset": "random synthetic stand-in"
            if args.smoke_test
            else "Elliptic++ (txs_features/txs_classes)",
            "feature_set": "M1 (blockchain)",
            "n_features": len(feature_cols),
            "train_cutoff": args.train_cutoff,
            "seed": args.seed,
            "smoke_test": args.smoke_test,
            "interpretable_columns_used_by_rules": INTERPRETABLE_TX_FEATURES,
        },
        "benchmark": {
            "n_train": bench.n_train,
            "n_test": bench.n_test,
            "detectors": bench.detectors,
        },
        "rolling_origin": rolling,
        "robustness": robustness,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = "benchmark_smoke" if args.smoke_test else "benchmark"
    (args.out_dir / f"{stem}_results.json").write_text(
        json.dumps(results, indent=2, default=float), encoding="utf-8"
    )
    md_path = args.out_dir / f"{stem.upper()}.md"
    md_path.write_text(render_markdown(results), encoding="utf-8")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
