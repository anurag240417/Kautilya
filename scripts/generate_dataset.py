"""Write a synthetic dataset in the problem-statement schema (CSV, JSON and XML).

    python -m scripts.generate_dataset --n-tx 20000 --out data/synthetic

Also writes ``<out>_ground_truth.csv`` (address -> true entity, illicit flag, scenario)
so a run can be scored against the planted truth.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from backend.forensics.synth import SynthConfig, generate_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-tx", type=int, default=20_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument(
        "--out", type=Path, default=Path("data/synthetic"), help="output path without extension"
    )
    ap.add_argument("--formats", default="csv,json,xml")
    args = ap.parse_args()

    ds = generate_dataset(SynthConfig(n_tx=args.n_tx, seed=args.seed))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for fmt in args.formats.split(","):
        path = args.out.with_suffix(f".{fmt}")
        getattr(ds, f"write_{fmt}")(path)
        print(f"wrote {path}")

    ent = ds.truth.entities.set_index("entity_id")
    truth = pd.DataFrame(
        {
            "address": ds.addresses,
            "true_entity": ds.truth.addr_entity,
            "is_illicit": ent["is_illicit"].to_numpy()[ds.truth.addr_entity],
            "scenario": ent["scenario"].to_numpy()[ds.truth.addr_entity],
            "role": ent["role"].to_numpy()[ds.truth.addr_entity],
        }
    )
    truth_path = args.out.with_name(args.out.name + "_ground_truth.csv")
    truth.to_csv(truth_path, index=False)
    print(f"wrote {truth_path}")
    print(ds.summary())


if __name__ == "__main__":
    main()
