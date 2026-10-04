"""CLI runner for generating and inspecting synthetic network data.

Usage:
    python generate_network_data.py [--sample-size 500] [--seed 42] [--output-dir ./generated_output]

Generates synthetic network observations (IPs, ports, ASNs, countries,
propagation timestamps) and ground truth metadata, exports them to CSV,
and prints a summary preview.
"""

import argparse
import logging
from pathlib import Path

import pandas as pd

from backend.config import get_settings
from backend.domain.experiment import GeneratorConfig
from backend.generator.export import (
    export_ground_truth_to_csv,
    export_observations_to_csv,
)
from backend.generator.pipeline import generate_synthetic_network_dataset
from backend.ingestion.csv_parser import parse_txs_classes, parse_txs_features

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate_network_data")


def create_sample_transactions(n: int = 100) -> pd.DataFrame:
    """Create sample transaction data when dataset CSVs are not present."""
    import random

    rng = random.Random(42)
    rows = []
    for txid in range(1, n + 1):
        time_step = (txid % 49) + 1
        # 1=Illicit (15%), 2=Licit (70%), 3=Unknown (15%)
        rand_val = rng.random()
        if rand_val < 0.15:
            label = 1
        elif rand_val < 0.85:
            label = 2
        else:
            label = 3

        rows.append({"txid": txid, "time_step": time_step, "label": label})

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Kautilya Synthetic Network Data Generator"
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=500,
        help="Number of transactions to generate network data for (0 = all)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible generation",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./generated_network_data",
        help="Directory to save output CSV files",
    )
    args = parser.parse_args()

    settings = get_settings()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load transactions from dataset or create sample
    txs_df: pd.DataFrame
    if settings.txs_classes_path.exists() and settings.txs_features_path.exists():
        logger.info("Loading transaction data from %s...", settings.elliptic_pp_dir)
        df_classes = parse_txs_classes(settings.txs_classes_path)
        df_features = parse_txs_features(settings.txs_features_path)[["txId", "time_step"]]
        txs_df = pd.merge(df_features, df_classes, on="txId", how="left")
        if args.sample_size > 0 and args.sample_size < len(txs_df):
            txs_df = txs_df.head(args.sample_size)
    else:
        logger.info(
            "Elliptic++ dataset not found at %s. Using %d sample transactions.",
            settings.elliptic_pp_dir,
            args.sample_size,
        )
        txs_df = create_sample_transactions(args.sample_size)

    logger.info("Running network generation pipeline for %d transactions...", len(txs_df))

    # 2. Run generation pipeline
    config = GeneratorConfig(
        random_seed=args.seed,
        generator_version="0.1.0",
        node_pool_size=500,
    )

    obs_df, gt_df, stats = generate_synthetic_network_dataset(
        config=config,
        txs_df=txs_df,
        generation_run_id=f"cli_run_seed_{args.seed}",
    )

    # 3. Export to CSV
    obs_file = export_observations_to_csv(obs_df, out_dir / "network_observations.csv")
    gt_file = export_ground_truth_to_csv(gt_df, out_dir / "network_ground_truth.csv")

    # 4. Print summary & preview
    print("\n" + "=" * 70)
    print("               SYNTHETIC NETWORK DATA GENERATION COMPLETE          ")
    print("=" * 70)
    print(f" Total Transactions Processed : {stats['unique_txids']:,}")
    print(f" Total Network Observations   : {stats['total_observations']:,}")
    print(f" Unique Source IPs           : {stats['unique_src_ips']:,}")
    print(f" Unique Destination IPs      : {stats['unique_dst_ips']:,}")
    print(f" Unique Countries Represented: {stats['unique_countries']:,}")
    print(f" Provenance Verified         : {stats['synthetic_provenance_verified']} (100% is_synthetic=True)")
    print("-" * 70)
    print(" Script Type Distribution:")
    for stype, count in stats["script_type_distribution"].items():
        print(f"   - {stype:<10}: {count:,}")
    print("-" * 70)
    print(" Scenario Distribution:")
    for scenario, count in stats.get("scenario_distribution", {}).items():
        print(f"   - {scenario:<20}: {count:,}")
    print("=" * 70)

    print("\n--- SAMPLE NETWORK OBSERVATIONS (First 10 rows) ---")
    preview_cols = ["txid", "src_ip", "dst_ip", "src_port", "dst_port", "country", "script_type", "scenario_id", "is_synthetic"]
    print(obs_df[preview_cols].head(10).to_string(index=False))

    print(f"\nFiles saved successfully:")
    print(f"  - Observations CSV : {obs_file.resolve()}")
    print(f"  - Ground Truth CSV  : {gt_file.resolve()}\n")


if __name__ == "__main__":
    main()
