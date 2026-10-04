"""Analyse a raw transaction file (CSV/JSON/XML) in the problem-statement schema.

    python -m scripts.analyze_dataset path/to/dataset.csv [--top 20] [--amount-unit auto|btc|sat]

Prints the ingestion audit (what was accepted, repaired, dropped), heuristic counts
and the top-ranked investigative leads.  Use this first on any dataset you are given.
"""

from __future__ import annotations

import argparse
import json
import logging

from backend.forensics.evidence import build_entity_report
from backend.forensics.ingest import load_raw_transactions
from backend.forensics.pipeline import ForensicsConfig, run_forensics


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("path")
    ap.add_argument("--top", type=int, default=15)
    ap.add_argument("--amount-unit", default="auto", choices=["auto", "btc", "sat"])
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    ds = load_raw_transactions(args.path, amount_unit=args.amount_unit)
    print("== Ingestion audit ==")
    print(json.dumps(ds.report.to_dict(), indent=2))
    print("\n== Dataset ==")
    print(json.dumps(ds.summary(), indent=2))

    res = run_forensics(ds, ForensicsConfig())
    print("\n== Heuristic hits ==")
    print(json.dumps(res.h.counts(), indent=2))
    print(f"\nModel trained via: {res.label_source}")
    print("\n== Top leads ==")
    for eid in res.alert_table(args.top).index:
        rep = build_entity_report(res, int(eid))
        print(
            f"\n#{rep['score']['rank']} {rep['label']}  score={rep['score']['final']:.0f} ({rep['score']['tier']})"
        )
        print("  " + rep["summary"])


if __name__ == "__main__":
    main()
