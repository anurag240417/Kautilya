"""Tests for the benchmark harness (metrics, baselines, robustness)."""

import numpy as np
import pandas as pd

from backend.ml.benchmark import (
    best_f1_operating_point,
    bootstrap_ci,
    per_scenario_breakdown,
    perturb_features,
    precision_recall_at_k,
    ranking_metrics,
    rule_based_score,
    run_benchmark,
)
from backend.ml.benchmark_report import render_markdown

FEATURES = ["f1", "f2", "f3"]


def test_precision_recall_at_k_perfect_and_worst():
    y = np.array([1, 1, 0, 0, 0])
    good = np.array([0.9, 0.8, 0.3, 0.2, 0.1])
    bad = -good
    assert precision_recall_at_k(y, good, 2) == (1.0, 1.0)
    assert precision_recall_at_k(y, bad, 2) == (0.0, 0.0)


def test_k_is_capped_at_n():
    y = np.array([1, 0])
    p, r = precision_recall_at_k(y, np.array([0.9, 0.1]), 50)
    assert p == 0.5 and r == 1.0


def test_ranking_metrics_perfect_scores():
    y = np.array([1, 1, 0, 0, 0, 0])
    m = ranking_metrics(y, np.array([0.9, 0.8, 0.4, 0.3, 0.2, 0.1]), ks=(2,))
    assert m["pr_auc"] == 1.0
    assert m["roc_auc"] == 1.0
    assert m["precision@2"] == 1.0
    assert m["best_f1"] == 1.0
    assert m["best_f1_fpr"] == 0.0
    assert m["lift@2"] == 3.0  # base rate 1/3


def test_ranking_metrics_single_class_returns_none():
    m = ranking_metrics(np.zeros(5, dtype=int), np.random.rand(5))
    assert m["pr_auc"] is None and m["roc_auc"] is None


def test_best_f1_handles_tied_scores():
    y = np.array([1, 0, 1, 0])
    m = best_f1_operating_point(y, np.array([0.5, 0.5, 0.5, 0.5]))
    # Only one cut is possible with all-tied scores: flag everything.
    assert m["best_f1_recall"] == 1.0
    assert m["best_f1_precision"] == 0.5
    assert m["best_f1_fpr"] == 1.0


def test_bootstrap_ci_brackets_point_estimate_and_is_reproducible():
    rng = np.random.default_rng(0)
    y = (rng.random(400) < 0.2).astype(int)
    s = y * 0.5 + rng.random(400)
    from sklearn.metrics import average_precision_score as ap

    lo, hi = bootstrap_ci(y, s, ap, n_boot=100, seed=1)
    assert lo < ap(y, s) < hi
    assert (lo, hi) == bootstrap_ci(y, s, ap, n_boot=100, seed=1)


def test_rule_score_flags_and_missing_columns():
    df = pd.DataFrame(
        {
            "num_output_addresses": [12, 1],
            "num_input_addresses": [1, 1],
            "out_BTC_max": [1.0, 1.0],
            "out_BTC_total": [10.0, 10.0],
            "in_txs_degree": [0, 0],
            "out_txs_degree": [0, 0],
            "total_BTC": [1.0, 1.0],
        }
    )
    s = rule_based_score(df)
    assert s[0] == 0.2 and s[1] == 0.0
    # Missing columns must not crash: rules just do not fire.
    assert (rule_based_score(pd.DataFrame(index=range(3))) == 0).all()


def test_perturb_features_identity_and_noise():
    X = np.ones((50, 4))
    mean, std = np.ones(4), np.full(4, 2.0)
    assert np.array_equal(perturb_features(X, mean, std), X)
    noisy = perturb_features(X, mean, std, noise_level=0.5)
    assert not np.array_equal(noisy, X)
    dropped = perturb_features(X * 5, mean, std, dropout_rate=1.0)
    assert np.allclose(dropped, 1.0)


def test_per_scenario_breakdown():
    y = np.array([1, 1, 1, 0])
    s = np.array([0.9, 0.1, 0.8, 0.2])
    out = per_scenario_breakdown(y, s, ["tor", "tor", "vpn", "vpn"], threshold=0.5)
    assert out["tor"]["recall"] == 0.5
    assert out["vpn"]["recall"] == 1.0


def _toy_frames(seed: int = 0):
    rng = np.random.default_rng(seed)

    def make(n):
        y = (rng.random(n) < 0.15).astype(int)
        df = pd.DataFrame(rng.normal(size=(n, 3)), columns=FEATURES)
        df.loc[y == 1, "f1"] += 2.5
        df["target"] = y
        return df

    return make(600), make(400)


def test_run_benchmark_ml_beats_chance_and_report_renders():
    train, test = _toy_frames()
    result, _ = run_benchmark(train, test, FEATURES, ks=(50, 100), n_boot=10, n_estimators=25)
    det = result.detectors
    base = det["random"]["base_rate"]
    assert det["ml_only_rf"]["pr_auc"] > base + 0.3
    assert det["ml_only_rf"]["pr_auc"] > det["random"]["pr_auc"]
    assert {"rules_only", "logistic_regression", "anomaly_only", "fused_rf_anomaly"} <= set(det)

    md = render_markdown(
        {
            "meta": {
                "dataset": "toy",
                "feature_set": "t",
                "n_features": 3,
                "train_cutoff": 34,
                "seed": 42,
                "smoke_test": True,
            },
            "benchmark": {
                "n_train": result.n_train,
                "n_test": result.n_test,
                "detectors": det,
            },
        }
    )
    assert "Detector comparison" in md and "SMOKE TEST" in md
