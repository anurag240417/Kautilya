"""Scale test: generate N transactions, run the full pipeline, report timings and memory.

    python -m scripts.scale_test --n-tx 1000000

Writes reports/SCALE_TEST.md.  Uses the synthetic generator, so it measures
pipeline throughput, not detection accuracy on real data.
"""

from __future__ import annotations

import argparse
import platform
import time
from pathlib import Path

from backend.forensics.pipeline import ForensicsConfig, run_forensics
from backend.forensics.synth import SynthConfig, generate_dataset


def peak_memory_mb() -> float | None:
    """Peak resident memory of this process in MB (Linux/macOS/Windows)."""
    try:
        import resource

        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 1024 if platform.system() == "Linux" else r / 1024 / 1024
    except ImportError:
        pass
    try:
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (n, ctypes.c_size_t)
                for n in (
                    "PeakWorkingSetSize",
                    "WorkingSetSize",
                    "QuotaPeakPagedPoolUsage",
                    "QuotaPagedPoolUsage",
                    "QuotaPeakNonPagedPoolUsage",
                    "QuotaNonPagedPoolUsage",
                    "PagefileUsage",
                    "PeakPagefileUsage",
                )
            ]

        c = Counters()
        c.cb = ctypes.sizeof(Counters)
        kernel32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(Counters),
            wintypes.DWORD,
        ]
        if not psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return None
        return c.PeakWorkingSetSize / 1024 / 1024
    except Exception:
        return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-tx", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--n-estimators", type=int, default=100)
    ap.add_argument("--out", type=Path, default=Path("reports/SCALE_TEST.md"))
    args = ap.parse_args()

    t0 = time.perf_counter()
    ds = generate_dataset(SynthConfig(n_tx=args.n_tx, seed=args.seed))
    t_gen = time.perf_counter() - t0
    s = ds.summary()
    print(
        f"generated {s['transactions']:,} tx / {s['addresses']:,} addresses in {t_gen:.0f}s",
        flush=True,
    )

    t1 = time.perf_counter()
    res = run_forensics(
        ds, ForensicsConfig(n_estimators=args.n_estimators, seed=args.seed, n_folds=2)
    )
    t_run = time.perf_counter() - t1
    rank = res.metrics.get("ranking", {}).get("fused_score", {})
    mem = peak_memory_mb()

    lines = [
        "# ChainTrace Scale Test",
        "",
        f"- Machine: {platform.platform()}, Python {platform.python_version()}",
        f"- Transactions: {s['transactions']:,}; addresses: {s['addresses']:,}; network observations: {s['observations']:,}",
        f"- Entities resolved: {res.et.n_entities:,}; active (>=2 events): {res.n_ranked:,}; "
        f"alerts raised: {int((res.scores['rank'] > 0).sum()):,}",
        f"- Peak memory: {'n/a' if mem is None else f'{mem:.0f} MB'}",
        f"- Data generation: {t_gen:.0f} s (not part of the pipeline)",
        f"- **Pipeline total: {t_run:.0f} s** ({s['transactions'] / t_run:,.0f} tx/s)",
        "",
        "| Stage | Seconds |",
        "|---|---|",
        *[f"| {k.removesuffix('_s')} | {v:.1f} |" for k, v in res.timings.items()],
        "",
        f"Fused-score PR-AUC on this run: {rank.get('pr_auc', float('nan')):.3f} "
        "(synthetic ground truth, 2-fold cross-fit).",
        "",
        "Notes: model training caps negatives at 40,000 entities; "
        f"Random Forest trees: {args.n_estimators}; folds: 2. Fusion runs on candidate entities only.",
    ]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
