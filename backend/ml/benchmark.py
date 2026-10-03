"""Benchmark harness: does the ML actually beat rules and simple baselines?

Answers four questions an evaluator will ask:

1. How good is each detector under class imbalance (PR-AUC, precision@k,
   recall@k, false-positive rate) and how certain are those numbers
   (bootstrap confidence intervals)?
2. Does ML beat a transparent rules-only detector, a linear model, and
   chance?  (``run_benchmark``)
3. Does performance hold up across different future windows, not just the
   one split we happened to pick?  (``rolling_origin_evaluation``)
4. How gracefully does it degrade when features are noisy or missing, as
   they would be against an evasive adversary?  (``robustness_sweep``)

All splits are temporal (train strictly before test); nothing here shuffles
across time steps.  Detectors are scored on the same test rows so the
numbers are directly comparable.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from backend.domain.risk import SignalInput
from backend.ml.anomaly import AnomalyDetector
from backend.ml.classifier import TransactionClassifier
from backend.risk.scorer import synthesize_risk_score

logger = logging.getLogger(__name__)

DEFAULT_KS: tuple[int, ...] = (50, 100, 200, 500)
DEFAULT_NOISE_LEVELS: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 1.0)
DEFAULT_DROPOUT_RATES: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5)

# Rule thresholds are fixed a priori (never tuned on test data).  They encode
# well-known laundering shapes using the interpretable Elliptic++ columns.
RULE_NAMES: tuple[str, ...] = (
    "fan_out_10plus",
    "fan_in_10plus",
    "peel_chain_shape",
    "pass_through_hub",
    "large_value_100btc",
)


# ======================================================================
# Metrics
# ======================================================================


def precision_recall_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> tuple[float, float]:
    """Precision and recall among the ``k`` highest-scored rows.

    Ties are broken by original order (stable sort) so results are
    deterministic.  ``k`` is capped at the number of rows.
    """
    k = max(1, min(int(k), len(y_true)))
    order = np.argsort(-scores, kind="stable")[:k]
    hits = float(y_true[order].sum())
    total_pos = float(y_true.sum())
    return hits / k, (hits / total_pos if total_pos > 0 else 0.0)


def ranking_metrics(
    y_true: np.ndarray,
    scores: np.ndarray,
    ks: Sequence[int] = DEFAULT_KS,
) -> dict:
    """Threshold-free ranking metrics plus top-k triage metrics.

    Also reports the best-F1 operating point and the false-positive rate at
    that point, which is what an analyst feels as alert fatigue.
    """
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=float)
    n_pos = int(y_true.sum())
    n = len(y_true)

    out: dict = {
        "n": n,
        "n_positive": n_pos,
        "base_rate": n_pos / n if n else 0.0,
        "pr_auc": None,
        "roc_auc": None,
    }
    if n_pos == 0 or n_pos == n:
        return out

    out["pr_auc"] = float(average_precision_score(y_true, scores))
    out["roc_auc"] = float(roc_auc_score(y_true, scores))

    for k in ks:
        kk = min(k, n)
        p, r = precision_recall_at_k(y_true, scores, kk)
        out[f"precision@{k}"] = p
        out[f"recall@{k}"] = r
        out[f"lift@{k}"] = p / out["base_rate"]

    best = best_f1_operating_point(y_true, scores)
    out.update(best)
    return out


def best_f1_operating_point(y_true: np.ndarray, scores: np.ndarray) -> dict:
    """Operating point maximising F1 over all score thresholds.

    Note: this picks the threshold on the evaluated data, so it is an
    optimistic *ceiling* for threshold-dependent metrics.  PR-AUC and
    precision@k are the honest headline numbers.
    """
    order = np.argsort(-scores, kind="stable")
    y_sorted = y_true[order]
    s_sorted = scores[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos

    # Only evaluate cut points where the score actually changes (ties).
    distinct = np.r_[np.where(np.diff(s_sorted))[0], len(s_sorted) - 1]
    tp, fp = tp[distinct], fp[distinct]

    precision = tp / (tp + fp)
    recall = tp / n_pos
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros_like(denom), where=denom > 0)
    i = int(np.argmax(f1))
    return {
        "best_f1": float(f1[i]),
        "best_f1_precision": float(precision[i]),
        "best_f1_recall": float(recall[i]),
        "best_f1_fpr": float(fp[i] / n_neg) if n_neg else 0.0,
        "best_f1_threshold": float(s_sorted[distinct[i]]),
    }


def bootstrap_ci(
    y_true: np.ndarray,
    scores: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    n_boot: int = 300,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float]:
    """Percentile bootstrap CI for ``metric(y_true, scores)``.

    Resamples rows with replacement; resamples containing a single class
    are skipped.  Returns (nan, nan) if no valid resample was drawn.
    """
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    scores = np.asarray(scores)
    n = len(y_true)
    vals: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        yb = y_true[idx]
        if yb.min() == yb.max():
            continue
        vals.append(float(metric(yb, scores[idx])))
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def _pr_auc(y: np.ndarray, s: np.ndarray) -> float:
    return float(average_precision_score(y, s))


def _p_at_100(y: np.ndarray, s: np.ndarray) -> float:
    return precision_recall_at_k(y, s, 100)[0]


# ======================================================================
# Detectors
# ======================================================================


def rule_based_score(df: pd.DataFrame) -> np.ndarray:
    """Transparent rules-only baseline in [0, 1].

    Fraction of five fixed heuristics that fire.  Missing columns count as
    "rule did not fire", so the baseline degrades instead of crashing.
    """

    def col(name: str) -> pd.Series:
        if name in df.columns:
            return df[name].astype(float).fillna(0.0)
        return pd.Series(0.0, index=df.index)

    n_out = col("num_output_addresses")
    n_in = col("num_input_addresses")
    out_max = col("out_BTC_max")
    out_total = col("out_BTC_total").replace(0.0, np.nan)

    flags = [
        n_out >= 10,
        n_in >= 10,
        (n_out == 2) & ((out_max / out_total).fillna(0.0) > 0.9),
        (col("in_txs_degree") >= 5) & (col("out_txs_degree") >= 5),
        col("total_BTC") >= 100,
    ]
    return np.mean([f.astype(float).to_numpy() for f in flags], axis=0)


def fused_score(
    rf_prob: np.ndarray,
    anomaly: np.ndarray,
    rules: np.ndarray | None = None,
) -> np.ndarray:
    """Score rows with the production risk synthesizer (0-100 scale).

    Uses ``synthesize_risk_score`` itself, not a re-implementation, so the
    "fused" row in the report measures what the dashboard actually ranks by.
    Rule score, if given, enters as the structural (graph) signal slot.
    """
    out = np.empty(len(rf_prob))
    for i in range(len(rf_prob)):
        signals = SignalInput(
            entity_id=str(i),
            entity_type="transaction",
            illicit_probability=float(np.clip(rf_prob[i], 0.0, 1.0)),
            anomaly_score=float(np.clip(anomaly[i], 0.0, 1.0)),
            graph_signal=None if rules is None else float(np.clip(rules[i], 0.0, 1.0)),
        )
        out[i] = synthesize_risk_score(signals).score
    return out


@dataclass
class FittedDetectors:
    """All detectors fitted on one training window."""

    rf: TransactionClassifier
    iforest: AnomalyDetector
    logreg: LogisticRegression
    scaler: StandardScaler
    feature_cols: list[str]
    seed: int = 42

    def score_all(self, test_df: pd.DataFrame, X_test: np.ndarray) -> dict[str, np.ndarray]:
        """Score one test matrix with every detector (higher = more suspicious)."""
        rng = np.random.default_rng(self.seed)
        rf_prob = self.rf.predict_proba(X_test)[:, 1]
        anomaly = self.iforest.score(X_test)
        rules = rule_based_score(test_df)
        return {
            "random": rng.random(len(X_test)),
            "rules_only": rules,
            "logistic_regression": self.logreg.predict_proba(self.scaler.transform(X_test))[:, 1],
            "anomaly_only": anomaly,
            "ml_only_rf": rf_prob,
            "fused_rf_anomaly": fused_score(rf_prob, anomaly),
            "fused_rf_anomaly_rules": fused_score(rf_prob, anomaly, rules),
        }


def fit_detectors(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_cols: list[str],
    n_estimators: int = 200,
    seed: int = 42,
) -> FittedDetectors:
    """Fit RF, Isolation Forest, and logistic regression on one train window."""
    rf = TransactionClassifier(
        feature_set_name="benchmark",
        n_estimators=n_estimators,
        random_state=seed,
    )
    rf.train(X_train, y_train, feature_names=feature_cols)

    iforest = AnomalyDetector()
    iforest.fit(X_train)

    scaler = StandardScaler().fit(X_train)
    logreg = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=seed)
    logreg.fit(scaler.transform(X_train), y_train)

    return FittedDetectors(rf, iforest, logreg, scaler, list(feature_cols), seed)


# ======================================================================
# Experiments
# ======================================================================


@dataclass
class BenchmarkResult:
    """Detector comparison on one train/test split."""

    n_train: int
    n_test: int
    detectors: dict[str, dict] = field(default_factory=dict)


def run_benchmark(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    ks: Sequence[int] = DEFAULT_KS,
    n_boot: int = 300,
    n_estimators: int = 200,
    seed: int = 42,
) -> tuple[BenchmarkResult, FittedDetectors]:
    """Compare every detector on the same temporal split.

    ``train_df`` / ``test_df`` need ``feature_cols`` and ``target``; the
    rules baseline additionally reads the interpretable columns when
    present.  Feature NaNs are median-imputed from the *training* window.
    """
    medians = train_df[feature_cols].median()
    X_train = train_df[feature_cols].fillna(medians).to_numpy(dtype=np.float64)
    X_test = test_df[feature_cols].fillna(medians).to_numpy(dtype=np.float64)
    y_train = train_df["target"].to_numpy()
    y_test = test_df["target"].to_numpy()

    fitted = fit_detectors(X_train, y_train, feature_cols, n_estimators, seed)
    scores = fitted.score_all(test_df, X_test)

    result = BenchmarkResult(n_train=len(train_df), n_test=len(test_df))
    for name, s in scores.items():
        m = ranking_metrics(y_test, s, ks)
        if m["pr_auc"] is not None and n_boot > 0:
            m["pr_auc_ci95"] = bootstrap_ci(y_test, s, _pr_auc, n_boot, seed=seed)
            m["precision@100_ci95"] = bootstrap_ci(y_test, s, _p_at_100, n_boot, seed=seed)
        result.detectors[name] = m
        logger.info("benchmark %-24s PR-AUC=%s", name, m["pr_auc"])
    return result, fitted


def rolling_origin_evaluation(
    df: pd.DataFrame,
    feature_cols: list[str],
    cutoffs: Sequence[int] = (20, 25, 30, 34, 39, 44),
    window: int = 5,
    n_estimators: int = 100,
    seed: int = 42,
    time_col: str = "time_step",
) -> list[dict]:
    """Retrain at several cutoffs and test on the next ``window`` steps.

    A single split can be lucky.  If PR-AUC is stable across origins the
    result is not an artefact of where we cut.  Each row reports the ML
    detector and the rules baseline on the same unseen window.
    """
    rows: list[dict] = []
    for cutoff in cutoffs:
        train = df[df[time_col] <= cutoff]
        test = df[(df[time_col] > cutoff) & (df[time_col] <= cutoff + window)]
        if len(train) == 0 or len(test) == 0 or test["target"].nunique() < 2:
            continue
        if train["target"].nunique() < 2:
            continue
        medians = train[feature_cols].median()
        X_tr = train[feature_cols].fillna(medians).to_numpy(dtype=np.float64)
        X_te = test[feature_cols].fillna(medians).to_numpy(dtype=np.float64)
        rf = TransactionClassifier(
            feature_set_name="rolling", n_estimators=n_estimators, random_state=seed
        )
        rf.train(X_tr, train["target"].to_numpy(), feature_names=feature_cols)
        y = test["target"].to_numpy()
        ml = ranking_metrics(y, rf.predict_proba(X_te)[:, 1])
        rules = ranking_metrics(y, rule_based_score(test))
        rows.append(
            {
                "train_steps": f"1-{cutoff}",
                "test_steps": f"{cutoff + 1}-{cutoff + window}",
                "n_test": len(test),
                "n_positive": int(y.sum()),
                "ml_pr_auc": ml["pr_auc"],
                "ml_precision@100": ml.get("precision@100"),
                "rules_pr_auc": rules["pr_auc"],
                "base_rate": ml["base_rate"],
            }
        )
    return rows


def perturb_features(
    X: np.ndarray,
    train_mean: np.ndarray,
    train_std: np.ndarray,
    noise_level: float = 0.0,
    dropout_rate: float = 0.0,
    seed: int = 42,
) -> np.ndarray:
    """Simulate evasive or degraded telemetry on a test matrix.

    ``noise_level``: Gaussian noise with sd = noise_level * train std per
    feature (an attacker obfuscating amounts/structure).
    ``dropout_rate``: fraction of cells replaced by the train mean (features
    that could not be observed).
    """
    rng = np.random.default_rng(seed)
    out = X.astype(np.float64).copy()
    if noise_level > 0:
        out += rng.normal(0.0, 1.0, out.shape) * (noise_level * train_std)
    if dropout_rate > 0:
        mask = rng.random(out.shape) < dropout_rate
        out = np.where(mask, train_mean, out)
    return out


def robustness_sweep(
    fitted: FittedDetectors,
    test_df: pd.DataFrame,
    X_test: np.ndarray,
    y_test: np.ndarray,
    X_train: np.ndarray,
    noise_levels: Sequence[float] = DEFAULT_NOISE_LEVELS,
    dropout_rates: Sequence[float] = DEFAULT_DROPOUT_RATES,
    seed: int = 42,
) -> dict[str, list[dict]]:
    """PR-AUC of ML-only and fused detectors as test features are degraded.

    The model is *not* retrained: this measures how a deployed model copes
    with telemetry that is noisier than it saw in training.  Rule scores are
    computed from the clean table, so the fused+rules row is not included.
    """
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    def evaluate(Xp: np.ndarray) -> dict:
        rf_prob = fitted.rf.predict_proba(Xp)[:, 1]
        anomaly = fitted.iforest.score(Xp)
        return {
            "ml_only_rf_pr_auc": ranking_metrics(y_test, rf_prob)["pr_auc"],
            "fused_pr_auc": ranking_metrics(y_test, fused_score(rf_prob, anomaly))["pr_auc"],
            "anomaly_only_pr_auc": ranking_metrics(y_test, anomaly)["pr_auc"],
        }

    noise_rows = [
        {
            "noise_level": nl,
            **evaluate(perturb_features(X_test, mean, std, noise_level=nl, seed=seed)),
        }
        for nl in noise_levels
    ]
    dropout_rows = [
        {
            "dropout_rate": dr,
            **evaluate(perturb_features(X_test, mean, std, dropout_rate=dr, seed=seed)),
        }
        for dr in dropout_rates
    ]
    return {"gaussian_noise": noise_rows, "feature_dropout": dropout_rows}


def per_scenario_breakdown(
    y_true: np.ndarray,
    scores: np.ndarray,
    groups: Sequence[str],
    threshold: float,
) -> dict[str, dict]:
    """Recall of positives per group (e.g. per synthetic scenario type).

    Useful once ground-truth scenario labels are available: shows which
    laundering patterns the detector misses instead of one blended number.
    """
    groups_arr = np.asarray(groups)
    out: dict[str, dict] = {}
    for g in sorted(set(groups_arr.tolist())):
        m = groups_arr == g
        pos = int(y_true[m].sum())
        caught = int(((scores[m] >= threshold) & (y_true[m] == 1)).sum())
        out[g] = {
            "n": int(m.sum()),
            "n_positive": pos,
            "recall": caught / pos if pos else None,
        }
    return out
