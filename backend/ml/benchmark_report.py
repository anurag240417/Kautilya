"""Render benchmark results as a Markdown report."""

from __future__ import annotations

_LABELS = {
    "random": "Random (chance)",
    "rules_only": "Rules only",
    "logistic_regression": "Logistic regression",
    "anomaly_only": "Isolation Forest only",
    "ml_only_rf": "Random Forest (ML only)",
    "fused_rf_anomaly": "Fused: RF + anomaly",
    "fused_rf_anomaly_rules": "Fused: RF + anomaly + rules",
}


def _f(v: float | None, digits: int = 3) -> str:
    return "n/a" if v is None else f"{v:.{digits}f}"


def _ci(ci: list[float] | tuple[float, float] | None) -> str:
    return "n/a" if not ci else f"[{ci[0]:.3f}, {ci[1]:.3f}]"


def render_markdown(results: dict) -> str:
    """Build the report from the dict written by ``scripts/run_benchmark.py``."""
    meta = results["meta"]
    bench = results["benchmark"]
    lines: list[str] = ["# Kautilya Benchmark Report", ""]
    lines += [
        f"- Dataset: **{meta['dataset']}**",
        f"- Features: {meta['feature_set']} ({meta['n_features']} columns)",
        f"- Split: train steps 1-{meta['train_cutoff']}, test steps after (strictly out-of-time)",
        f"- Train rows: {bench['n_train']:,}; test rows: {bench['n_test']:,}",
        f"- Seed: {meta['seed']}",
    ]
    if meta.get("smoke_test"):
        lines += ["", "> **SMOKE TEST ON RANDOM SYNTHETIC DATA. Numbers are not results.**"]
    lines += ["", "## 1. Detector comparison (same test rows)", ""]

    first = next(iter(bench["detectors"].values()))
    lines += [
        f"Test positives: {first['n_positive']:,} of {first['n']:,} "
        f"(base rate {first['base_rate']:.3%}). PR-AUC of a random detector equals the base rate.",
        "",
        "| Detector | PR-AUC | PR-AUC 95% CI | ROC-AUC | P@100 | P@100 95% CI | Recall@500 | Best-F1 | FPR at best-F1 |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for key, m in bench["detectors"].items():
        lines.append(
            f"| {_LABELS.get(key, key)} | {_f(m['pr_auc'])} | {_ci(m.get('pr_auc_ci95'))} | "
            f"{_f(m['roc_auc'])} | {_f(m.get('precision@100'))} | {_ci(m.get('precision@100_ci95'))} | "
            f"{_f(m.get('recall@500'))} | {_f(m.get('best_f1'))} | {_f(m.get('best_f1_fpr'), 4)} |"
        )
    lines += [
        "",
        "Best-F1 picks its threshold on the test data, so read it as a ceiling. "
        "PR-AUC and precision@k are the headline numbers.",
        "",
        "### Precision@k and lift over chance",
        "",
        "| Detector | P@50 | P@100 | P@200 | P@500 | Lift@100 |",
        "|---|---|---|---|---|---|",
    ]
    for key, m in bench["detectors"].items():
        lines.append(
            f"| {_LABELS.get(key, key)} | {_f(m.get('precision@50'))} | {_f(m.get('precision@100'))} | "
            f"{_f(m.get('precision@200'))} | {_f(m.get('precision@500'))} | {_f(m.get('lift@100'), 1)}x |"
        )

    rolling = results.get("rolling_origin") or []
    if rolling:
        lines += [
            "",
            "## 2. Stability across time windows (rolling origin)",
            "",
            "Retrained at each cutoff, tested on the next unseen window.",
            "",
            "| Train steps | Test steps | Test n | Positives | Base rate | ML PR-AUC | Rules PR-AUC | ML P@100 |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for r in rolling:
            lines.append(
                f"| {r['train_steps']} | {r['test_steps']} | {r['n_test']:,} | {r['n_positive']} | "
                f"{r['base_rate']:.3%} | {_f(r['ml_pr_auc'])} | {_f(r['rules_pr_auc'])} | "
                f"{_f(r['ml_precision@100'])} |"
            )

    rob = results.get("robustness")
    if rob:
        lines += [
            "",
            "## 3. Robustness to noisy or missing features",
            "",
            "Deployed model, degraded test features (no retraining).",
            "",
            "**Gaussian noise (sd = level x training std)**",
            "",
            "| Noise level | RF PR-AUC | Fused PR-AUC | Anomaly PR-AUC |",
            "|---|---|---|---|",
        ]
        for r in rob["gaussian_noise"]:
            lines.append(
                f"| {r['noise_level']} | {_f(r['ml_only_rf_pr_auc'])} | "
                f"{_f(r['fused_pr_auc'])} | {_f(r['anomaly_only_pr_auc'])} |"
            )
        lines += [
            "",
            "**Feature dropout (cells replaced by training mean)**",
            "",
            "| Dropout rate | RF PR-AUC | Fused PR-AUC | Anomaly PR-AUC |",
            "|---|---|---|---|",
        ]
        for r in rob["feature_dropout"]:
            lines.append(
                f"| {r['dropout_rate']} | {_f(r['ml_only_rf_pr_auc'])} | "
                f"{_f(r['fused_pr_auc'])} | {_f(r['anomaly_only_pr_auc'])} |"
            )

    lines += [
        "",
        "## Notes and limits",
        "",
        "- Rules baseline uses five fixed heuristics (fan-out, fan-in, peel shape, pass-through hub, "
        "large value); thresholds were set before looking at test data.",
        "- The fused rows call the production `synthesize_risk_score`, so they measure what the dashboard ranks by.",
        "- Elliptic++ labels are a known, partial ground truth. Unlabelled (class 3) rows are excluded.",
        "- Held-out *scenario-type* generalisation is not covered here. Elliptic++ has no scenario labels; "
        "the rolling-origin test is the out-of-distribution check.",
        "",
    ]
    return "\n".join(lines)
