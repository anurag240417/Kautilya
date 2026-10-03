"""Entity-level benchmark on synthetic ground truth -> reports/FORENSICS_BENCHMARK.md + JSON.

python -m scripts.run_forensic_benchmark                  # 60k transactions (~1 min)
python -m scripts.run_forensic_benchmark --n-tx 150000
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from backend.forensics.benchmark import render_markdown, run_forensic_benchmark
from backend.forensics.entities import clustering_quality
from backend.forensics.heuristics import evaluate_heuristics
from backend.forensics.pipeline import ForensicsConfig, prepare
from backend.forensics.synth import SynthConfig, generate_dataset


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-tx", type=int, default=60_000)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--n-estimators", type=int, default=150)
    ap.add_argument("--n-boot", type=int, default=200)
    ap.add_argument("--out-dir", type=Path, default=Path("reports"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    scale = max(1, args.n_tx // 5000)
    ds = generate_dataset(
        SynthConfig(
            n_tx=args.n_tx,
            seed=args.seed,
            n_ransomware=max(4, scale),
            n_darknet=max(3, scale // 2),
            n_layering=max(6, 2 * scale),
            n_dust=max(4, scale),
        )
    )
    cfg = ForensicsConfig(n_estimators=args.n_estimators, seed=args.seed)
    prep = prepare(ds, cfg)
    result = run_forensic_benchmark(ds, cfg, n_boot=args.n_boot, prepared=prep)
    result["heuristic_validation"] = evaluate_heuristics(ds, prep.h)
    result["clustering"] = clustering_quality(ds, prep.et.addr_cluster)
    meta = {"dataset": f"synthetic, seed {args.seed}", "transactions": f"{ds.n_tx:,}"}

    md = render_markdown(result, meta)
    fence = "`" * 3
    md += f"\n## 5. Heuristic validation against planted truth\n\n{fence}json\n"
    md += json.dumps(result["heuristic_validation"], indent=2, default=float) + f"\n{fence}\n"
    md += f"\n## 6. Entity resolution quality\n\n{fence}json\n"
    md += json.dumps(result["clustering"], indent=2, default=float) + f"\n{fence}\n"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {"meta": meta, **result}
    (args.out_dir / "forensics_benchmark.json").write_text(
        json.dumps(payload, indent=2, default=float), encoding="utf-8"
    )
    (args.out_dir / "FORENSICS_BENCHMARK.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
