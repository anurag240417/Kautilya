"""Kautilya Core Component Benchmarks.

Measures latency and throughput of key investigation pipeline components:
- Service layer API operations (transaction, wallet, alert, graph lookups)
- Risk engine synthesis (single + batch)
- Graph operations (ego-graph, shortest path)
- Demo data seeding

Usage:
    python -m benchmarks.bench_core
"""

import statistics
import time
from typing import Any

from backend.api.demo_data import build_demo_dataset
from backend.api.service import InvestigationService
from backend.domain.types import PriorityTier
from backend.risk.scorer import SignalInput, synthesize_risk_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _timeit(func: Any, iterations: int = 100) -> dict[str, float]:
    """Run a callable N times, return timing statistics in milliseconds."""
    times: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        func()
        elapsed = (time.perf_counter() - start) * 1000.0  # ms
        times.append(elapsed)

    times.sort()
    return {
        "iterations": iterations,
        "p50_ms": round(statistics.median(times), 4),
        "p95_ms": round(times[int(len(times) * 0.95)], 4),
        "p99_ms": round(times[int(len(times) * 0.99)], 4),
        "mean_ms": round(statistics.mean(times), 4),
        "min_ms": round(min(times), 4),
        "max_ms": round(max(times), 4),
        "ops_per_sec": round(iterations / (sum(times) / 1000.0), 1),
    }


def _format_result(name: str, result: dict[str, float]) -> str:
    """Format benchmark result as a readable line."""
    return (
        f"  {name:<45} "
        f"p50={result['p50_ms']:>8.3f}ms  "
        f"p95={result['p95_ms']:>8.3f}ms  "
        f"p99={result['p99_ms']:>8.3f}ms  "
        f"ops/s={result['ops_per_sec']:>10.1f}"
    )


# ---------------------------------------------------------------------------
# Benchmark Suites
# ---------------------------------------------------------------------------


def bench_demo_seeding() -> dict[str, Any]:
    """Benchmark full demo data seeding."""
    def _seed():
        svc = InvestigationService()
        svc.seed_sample_data()

    return _timeit(_seed, iterations=50)


def bench_service_operations(svc: InvestigationService) -> dict[str, dict[str, float]]:
    """Benchmark service-layer API operations."""
    results = {}

    # Transaction lookup (existing)
    results["get_transaction(1001)"] = _timeit(lambda: svc.get_transaction(1001))

    # Transaction lookup (missing)
    results["get_transaction(9999) [miss]"] = _timeit(lambda: svc.get_transaction(9999))

    # Wallet lookup
    results["get_wallet('1DrK44np3gMKuvcGeFHv')"] = _timeit(
        lambda: svc.get_wallet("1DrK44np3gMKuvcGeFHv")
    )

    # Wallet lookup (missing)
    results["get_wallet('nonexistent') [miss]"] = _timeit(
        lambda: svc.get_wallet("nonexistent")
    )

    # Alert listing (all)
    results["list_alerts()"] = _timeit(lambda: svc.list_alerts())

    # Alert listing (filtered by tier)
    from backend.domain.alert import AlertFilter
    crit_filter = AlertFilter(min_tier=PriorityTier.CRITICAL)
    results["list_alerts(tier=CRITICAL)"] = _timeit(
        lambda: svc.list_alerts(filter_criteria=crit_filter)
    )

    # Statistics
    results["get_statistics()"] = _timeit(lambda: svc.get_statistics())

    return results


def bench_graph_operations(svc: InvestigationService) -> dict[str, dict[str, float]]:
    """Benchmark graph query operations."""
    results = {}

    # Ego graph depth=1
    results["get_subgraph(1001, depth=1)"] = _timeit(
        lambda: svc.get_subgraph(1001, depth=1)
    )

    # Ego graph depth=2
    results["get_subgraph(1001, depth=2)"] = _timeit(
        lambda: svc.get_subgraph(1001, depth=2)
    )

    # Ego graph from wallet
    results["get_subgraph('1MixServiceXjk8dqG2hP', d=1)"] = _timeit(
        lambda: svc.get_subgraph("1MixServiceXjk8dqG2hP", depth=1)
    )

    # Shortest path
    results["find_entity_path(1001 → 1004)"] = _timeit(
        lambda: svc.find_entity_path(1001, 1004)
    )

    # Shortest path (no path)
    results["find_entity_path(1005 → 1001) [no path]"] = _timeit(
        lambda: svc.find_entity_path(1005, 1001), iterations=50
    )

    return results


def bench_risk_engine() -> dict[str, dict[str, float]]:
    """Benchmark risk synthesis engine."""
    results = {}

    # Single signal synthesis
    single = SignalInput(
        entity_id="bench-single",
        entity_type="transaction",
        illicit_probability=0.85,
    )
    results["synthesize_risk_score (1 signal)"] = _timeit(
        lambda: synthesize_risk_score(single), iterations=500
    )

    # Multi-signal synthesis
    multi = SignalInput(
        entity_id="bench-multi",
        entity_type="transaction",
        illicit_probability=0.92,
        anomaly_score=0.78,
        graph_signal=0.65,
        correlation_confidence=0.80,
        known_indicator_signal=0.90,
        contains_synthetic_input=True,
    )
    results["synthesize_risk_score (5 signals)"] = _timeit(
        lambda: synthesize_risk_score(multi), iterations=500
    )

    # Multi-signal with custom signals
    custom = SignalInput(
        entity_id="bench-custom",
        entity_type="transaction",
        illicit_probability=0.60,
        anomaly_score=0.40,
        custom_signals={"velocity": 0.7, "structuring": 0.5},
    )
    results["synthesize_risk_score (2+2 custom)"] = _timeit(
        lambda: synthesize_risk_score(custom), iterations=500
    )

    # Batch: 100 syntheses
    batch_signals = [
        SignalInput(
            entity_id=f"batch-{i}",
            entity_type="transaction",
            illicit_probability=i / 100.0,
            anomaly_score=(100 - i) / 100.0,
        )
        for i in range(100)
    ]

    def _batch():
        for s in batch_signals:
            synthesize_risk_score(s)

    results["synthesize_batch (100 entities)"] = _timeit(_batch, iterations=50)

    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_benchmarks() -> str:
    """Execute all benchmarks and return formatted report."""
    lines: list[str] = []
    lines.append("=" * 100)
    lines.append("Kautilya Core Component Benchmarks")
    lines.append("=" * 100)
    lines.append("")

    # 1. Demo seeding
    lines.append("▸ Demo Data Seeding")
    lines.append("-" * 100)
    seed_result = bench_demo_seeding()
    lines.append(_format_result("seed_sample_data()", seed_result))
    lines.append("")

    # 2. Service operations
    svc = InvestigationService()
    svc.seed_sample_data()

    lines.append("▸ Service Layer Operations")
    lines.append("-" * 100)
    svc_results = bench_service_operations(svc)
    for name, result in svc_results.items():
        lines.append(_format_result(name, result))
    lines.append("")

    # 3. Graph operations
    lines.append("▸ Graph Operations")
    lines.append("-" * 100)
    graph_results = bench_graph_operations(svc)
    for name, result in graph_results.items():
        lines.append(_format_result(name, result))
    lines.append("")

    # 4. Risk engine
    lines.append("▸ Risk Engine Synthesis")
    lines.append("-" * 100)
    risk_results = bench_risk_engine()
    for name, result in risk_results.items():
        lines.append(_format_result(name, result))
    lines.append("")

    lines.append("=" * 100)
    lines.append("All benchmarks completed.")
    lines.append("=" * 100)

    return "\n".join(lines)


if __name__ == "__main__":
    report = run_benchmarks()
    print(report)
