"""Benchmark the entity detector: ablations, baselines, and unseen scenarios.

Questions answered, all at *entity* level with out-of-fold predictions:

1. **Ablation** - what does each feature family add? behavioural only ->
   + network -> + structural heuristics -> + graph embeddings/propagation.
2. **Baselines** - do we beat rules-only, anomaly-only, logistic regression,
   and chance?
3. **Unseen scenarios (leave-one-scenario-out)** - train with one laundering
   typology completely removed, then test on it.  This is the honest
   out-of-distribution test; in-distribution numbers on synthetic data are
   easy to make look good.
4. **Robustness** - degrade test features with noise and dropout.

All folds are grouped by campaign so a campaign's entities never appear in
both train and test.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from backend.forensics.dataset import RawDataset
from backend.forensics.graphml import EMB_PREFIX, PROP_COLUMNS
from backend.forensics.seedrisk import propagate_seed_risk, seed_boost
from backend.forensics.pipeline import (
    ForensicsConfig,
    Prepared,
    _train_subset,
    assign_folds,
    entity_campaigns,
    fuse,
    prepare,
    structural_score,
    temporal_burst_score,
)
from backend.ml.benchmark import bootstrap_ci, perturb_features, ranking_metrics

logger = logging.getLogger(__name__)

HEURISTIC_COLS = [
    "coinjoin_txs",
    "dust_spray_txs",
    "peel_chain_txs",
    "layering_chain_txs",
    "consolidation_txs",
    "batch_payout_txs",
    "max_peel_chain_len",
    "max_layering_chain_len",
]
NETWORK_COLS = [
    "tor_share",
    "vpn_share",
    "n_origin_ips",
    "n_origin_countries",
    "top_ip_share",
    "rapid_geo_hops",
    "network_obfuscation",
]
KS = (10, 25, 50, 100)


def feature_families(X: pd.DataFrame) -> dict[str, list[str]]:
    """Column groups available in the entity matrix."""
    emb = [c for c in X.columns if c.startswith(EMB_PREFIX)]
    prop = [c for c in PROP_COLUMNS if c in X.columns]
    special = set(HEURISTIC_COLS) | set(NETWORK_COLS) | set(prop) | set(emb)
    behav = [c for c in X.columns if c not in special]
    return {
        "behavioural": behav,
        "network": [c for c in NETWORK_COLS if c in X.columns],
        "heuristics": [c for c in HEURISTIC_COLS if c in X.columns],
        "graph": prop,
        "embeddings": emb,
    }


def _fit(X, y, train_idx, cols, cfg, emb_cols=()):
    """RF restricted to ``cols`` (+ ``emb_cols``)."""
    from backend.forensics.model import EntityModel

    sub = X[list(cols) + list(emb_cols)]
    return EntityModel(
        base_cols=list(cols),
        emb_cols=list(emb_cols),
        use_embeddings=bool(emb_cols),
        n_estimators=cfg.n_estimators,
        seed=cfg.seed,
    ).fit(sub.iloc[train_idx], y[train_idx])


def _fit_predict(X, y, train_idx, test_idx, cols, cfg, emb_cols=()):
    m = _fit(X, y, train_idx, cols, cfg, emb_cols)
    return m, m.predict_proba(X[list(cols) + list(emb_cols)].iloc[test_idx])


def _oof(X, y, active, folds, cols, cfg, emb_cols=()):
    """Out-of-fold probabilities for every active labelled entity."""
    n_folds = int(folds.max()) + 1
    p = np.full(len(y), np.nan)
    rng = np.random.default_rng(cfg.seed)
    for f in range(n_folds):
        tr = _train_subset(
            np.flatnonzero(active & (folds != f) & ~np.isnan(y)), y, cfg.max_train_negatives, rng
        )
        te = np.flatnonzero(active & (folds == f) & ~np.isnan(y))
        if len(te) and y[tr].sum() >= 2:
            _, p[te] = _fit_predict(X, y, tr, te, cols, cfg, emb_cols)
    return p


def _metrics(y, s, n_boot, seed):
    m = ranking_metrics(y.astype(int), s, KS)
    if m["pr_auc"] is not None and n_boot:
        from sklearn.metrics import average_precision_score as ap

        m["pr_auc_ci95"] = bootstrap_ci(y.astype(int), s, ap, n_boot, seed=seed)
    return m


def run_forensic_benchmark(
    ds: RawDataset,
    cfg: ForensicsConfig | None = None,
    n_boot: int = 200,
    prepared: Prepared | None = None,
) -> dict:
    """Run ablation, baselines, leave-one-scenario-out and robustness on ``ds``."""
    if ds.truth is None:
        msg = "Benchmark needs ground truth (use synthetic data)"
        raise ValueError(msg)
    cfg = cfg or ForensicsConfig()
    prep = prepared or prepare(ds, cfg)
    X, active = prep.X, prep.active
    lab = prep.et.labels(ds)
    y = lab["is_illicit"].to_numpy(dtype=float)
    scen = lab["scenario"].to_numpy()
    fam = feature_families(X)
    groups = entity_campaigns(ds, prep.et.addr_cluster, prep.et.n_entities)
    folds = assign_folds(groups, cfg.n_folds, cfg.seed)
    ev = active & ~np.isnan(y)
    yy = y[ev].astype(int)

    out: dict = {
        "n_entities_evaluated": int(ev.sum()),
        "n_illicit": int(yy.sum()),
        "scenarios": {s: int(((scen == s) & ev).sum()) for s in np.unique(scen) if s != "normal"},
        "feature_families": {k: len(v) for k, v in fam.items()},
    }

    # ---------------- 1. ablation ---------------------------------------
    configs = {
        "behavioural_only": (fam["behavioural"], []),
        "+network": (fam["behavioural"] + fam["network"], []),
        "+heuristics": (fam["behavioural"] + fam["network"] + fam["heuristics"], []),
        "+graph_propagation": (
            fam["behavioural"] + fam["network"] + fam["heuristics"] + fam["graph"],
            [],
        ),
        "+graph_embeddings (full)": (
            fam["behavioural"] + fam["network"] + fam["heuristics"] + fam["graph"],
            fam["embeddings"],
        ),
    }
    oof: dict[str, np.ndarray] = {}
    ablation = {}
    for name, (cols, emb) in configs.items():
        p = _oof(X, y, active, folds, cols, cfg, emb)
        oof[name] = p
        ablation[name] = _metrics(y[ev], p[ev], n_boot, cfg.seed)
    out["ablation"] = ablation

    # ---------------- 2. baselines --------------------------------------
    F = prep.et.features
    heur = structural_score(F, prep.graph.propagation)
    network = F["network_obfuscation"].to_numpy()
    burst = temporal_burst_score(F)
    full = oof["+graph_embeddings (full)"]

    # anomaly-only: Isolation Forest fitted per fold on the fold's training entities
    anomaly = np.full(len(y), np.nan)
    iso_cols = fam["behavioural"] + fam["network"]
    Xi = X[iso_cols].fillna(X[iso_cols].median()).to_numpy(dtype=float)
    for f in range(cfg.n_folds):
        tr = np.flatnonzero(active & (folds != f))
        te = np.flatnonzero(active & (folds == f))
        iso = IsolationForest(n_estimators=200, random_state=cfg.seed, n_jobs=-1).fit(Xi[tr])
        ref = np.sort(-iso.score_samples(Xi[tr]))
        anomaly[te] = np.searchsorted(ref, -iso.score_samples(Xi[te]), side="right") / len(ref)

    # logistic regression on all non-embedding features
    lr_cols = fam["behavioural"] + fam["network"] + fam["heuristics"] + fam["graph"]
    lr = np.full(len(y), np.nan)
    med = X[lr_cols].median()
    Xl = np.log1p(np.abs(X[lr_cols].fillna(med).to_numpy(dtype=float))) * np.sign(
        X[lr_cols].fillna(med).to_numpy(dtype=float)
    )
    rng = np.random.default_rng(cfg.seed)
    for f in range(cfg.n_folds):
        tr = _train_subset(np.flatnonzero(ev & (folds != f)), y, cfg.max_train_negatives, rng)
        te = np.flatnonzero(ev & (folds == f))
        sc = StandardScaler().fit(Xl[tr])
        clf = LogisticRegression(
            class_weight="balanced", max_iter=2000, random_state=cfg.seed
        ).fit(sc.transform(Xl[tr]), y[tr].astype(int))
        lr[te] = clf.predict_proba(sc.transform(Xl[te]))[:, 1]

    ci = np.flatnonzero(ev)
    fused_s = np.zeros(len(y))
    fs, _, _ = fuse(
        np.nan_to_num(full[ci]), np.nan_to_num(anomaly[ci]), heur[ci], network[ci], burst[ci]
    )
    fused_s[ci] = fs
    rnd = np.random.default_rng(cfg.seed).random(len(y))
    baselines = {
        "random": rnd,
        "rules_only (structural heuristics)": heur,
        "anomaly_only (IsolationForest)": anomaly,
        "logistic_regression": lr,
        "RF_behavioural_only": oof["behavioural_only"],
        "RF_full (ours, model only)": full,
        "FUSED (ours)": fused_s,
    }
    out["baselines"] = {
        k: _metrics(y[ev], np.nan_to_num(v[ev]), n_boot, cfg.seed) for k, v in baselines.items()
    }

    # ---------------- 2b. seed-based risk propagation ---------------------
    # Reveal a fraction of the illicit entities as known seeds, spread risk from them, and score
    # the *remaining* entities. Seeds are excluded from every number reported here.
    absorbing = (
        ((F["in_degree"] >= 30) & (F["out_degree"] >= 8)) | (F["n_addresses"] >= 50)
    ).to_numpy()
    positives = np.flatnonzero(ev & (y == 1))
    seed_rows: dict = {}
    for frac in (0.05, 0.10, 0.20):
        rng_s = np.random.default_rng(cfg.seed + int(frac * 1000))
        picked = rng_s.choice(positives, max(2, int(len(positives) * frac)), replace=False)
        sr = propagate_seed_risk(
            prep.graph.adjacency, picked, absorbing=absorbing, max_hops=cfg.seed_max_hops
        )
        keep = ev & ~sr.is_seed
        yk = y[keep].astype(int)
        boosted = seed_boost(fused_s, sr.risk, cfg.seed_boost_weight)
        near = (sr.hops[keep] >= 1) & (sr.hops[keep] <= 2)
        seed_rows[f"{int(frac * 100)}%"] = {
            "n_seeds": int(sr.n_seeds),
            "seed_propagation_only": _metrics(yk, sr.risk[keep], n_boot, cfg.seed),
            "fused_without_seeds": _metrics(yk, fused_s[keep], n_boot, cfg.seed),
            "fused_with_seed_boost": _metrics(yk, boosted[keep], n_boot, cfg.seed),
            "illicit_share_within_2_hops": float(yk[near].mean()) if near.any() else None,
            "illicit_share_overall": float(yk.mean()),
            "illicit_found_within_2_hops": float(near[yk == 1].mean())
            if (yk == 1).any()
            else None,
        }
    out["seed_propagation"] = seed_rows

    # ---------------- 3. leave-one-scenario-out --------------------------
    loso: dict = {}
    normal_test = np.random.default_rng(cfg.seed + 5).random(len(y)) < 0.3
    all_cols = fam["behavioural"] + fam["network"] + fam["heuristics"] + fam["graph"]
    beh_cols = fam["behavioural"] + fam["network"]
    for s in [k for k in out["scenarios"] if out["scenarios"][k] >= 4]:
        test_mask = ev & ((scen == s) | ((scen == "normal") & normal_test))
        train_mask = ev & (scen != s) & ~((scen == "normal") & normal_test)
        tr = _train_subset(
            np.flatnonzero(train_mask), y, cfg.max_train_negatives, np.random.default_rng(cfg.seed)
        )
        te = np.flatnonzero(test_mask)
        row = {"n_test_positive": int(y[te].sum()), "n_test_negative": int((1 - y[te]).sum())}
        _, p_full = _fit_predict(X, y, tr, te, all_cols, cfg, fam["embeddings"])
        _, p_beh = _fit_predict(X, y, tr, te, beh_cols, cfg)
        fused_te, _, _ = fuse(p_full, np.nan_to_num(anomaly[te]), heur[te], network[te], burst[te])
        for name, sc_ in (
            ("rules_only", heur[te]),
            ("RF_behavioural_only", p_beh),
            ("RF_full", p_full),
            ("FUSED", fused_te),
        ):
            m = ranking_metrics(y[te].astype(int), sc_, ks=(10, 25, 50))
            row[name] = {
                "pr_auc": m["pr_auc"],
                "roc_auc": m["roc_auc"],
                "precision@10": m.get("precision@10"),
                "base_rate": m["base_rate"],
            }
        loso[s] = row
    out["leave_one_scenario_out"] = loso

    # ---------------- 4. robustness (full RF, noisy test features) -------
    rob = []
    perturb_cols = fam["behavioural"] + fam["network"]
    mean = X[perturb_cols].mean().to_numpy()
    std = X[perturb_cols].std().to_numpy()
    fold_models = []
    for f in range(cfg.n_folds):
        tr = _train_subset(
            np.flatnonzero(ev & (folds != f)),
            y,
            cfg.max_train_negatives,
            np.random.default_rng(cfg.seed),
        )
        fold_models.append(_fit(X, y, tr, all_cols, cfg, fam["embeddings"]))
    for kind, levels in (("noise", (0.0, 0.25, 0.5, 1.0)), ("dropout", (0.0, 0.25, 0.5))):
        for lv in levels:
            scores = np.full(len(y), np.nan)
            for f, m in enumerate(fold_models):
                te = np.flatnonzero(ev & (folds == f))
                Xt = X.iloc[te].copy()
                Xt[perturb_cols] = perturb_features(
                    Xt[perturb_cols].to_numpy(dtype=float),
                    mean,
                    std,
                    noise_level=lv if kind == "noise" else 0.0,
                    dropout_rate=lv if kind == "dropout" else 0.0,
                    seed=cfg.seed,
                )
                scores[te] = m.predict_proba(Xt)
            mm = ranking_metrics(y[ev].astype(int), scores[ev], ks=(25,))
            rob.append(
                {
                    "perturbation": kind,
                    "level": lv,
                    "pr_auc": mm["pr_auc"],
                    "precision@25": mm.get("precision@25"),
                }
            )
    out["robustness"] = rob
    return out


def _f(v, d=3):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{d}f}"


def render_markdown(r: dict, meta: dict | None = None) -> str:
    meta = meta or {}
    L = ["# Kautilya Forensics Benchmark (entity level, synthetic ground truth)", ""]
    if meta:
        L += [
            f"- Dataset: {meta.get('dataset', '')}",
            f"- Transactions: {meta.get('transactions', '')}",
        ]
    L += [
        f"- Entities evaluated: {r['n_entities_evaluated']:,} ({r['n_illicit']} illicit)",
        f"- Illicit entities per scenario: {r['scenarios']}",
        "- All numbers are out-of-fold; folds are grouped by campaign so no campaign is in both train and test.",
        "- Data is synthetic. Treat results as evidence the method works on the modelled typologies, not as real-world accuracy.",
        "",
        "## 1. Ablation: what each feature family adds",
        "",
        "| Feature set | PR-AUC | 95% CI | P@10 | P@50 | P@100 |",
        "|---|---|---|---|---|---|",
    ]
    for k, m in r["ablation"].items():
        L.append(
            f"| {k} | {_f(m['pr_auc'])} | {_ci(m)} | {_f(m.get('precision@10'))} | {_f(m.get('precision@50'))} | {_f(m.get('precision@100'))} |"
        )
    L += [
        "",
        "## 2. Baselines",
        "",
        "| Detector | PR-AUC | 95% CI | ROC-AUC | P@10 | P@50 | P@100 |",
        "|---|---|---|---|---|---|---|",
    ]
    for k, m in r["baselines"].items():
        L.append(
            f"| {k} | {_f(m['pr_auc'])} | {_ci(m)} | {_f(m['roc_auc'])} | {_f(m.get('precision@10'))} | {_f(m.get('precision@50'))} | {_f(m.get('precision@100'))} |"
        )
    if r.get("seed_propagation"):
        L += [
            "",
            "## 2b. Seed-based risk propagation (known-illicit seeds)",
            "",
            "A fraction of illicit entities is revealed as seeds; risk is spread from them (personalised PageRank, "
            "exchange-style hubs absorb). Everything below is scored on the remaining, non-seed entities.",
            "",
            "| Seeds revealed | Seeds | Seed propagation only (PR-AUC) | Fused, no seeds | Fused + seed boost | "
            "Illicit share within 2 hops (overall) | Illicit found within 2 hops |",
            "|---|---|---|---|---|---|---|",
        ]
        for frac, row in r["seed_propagation"].items():
            L.append(
                f"| {frac} | {row['n_seeds']} | {_f(row['seed_propagation_only']['pr_auc'])} | "
                f"{_f(row['fused_without_seeds']['pr_auc'])} | {_f(row['fused_with_seed_boost']['pr_auc'])} | "
                f"{_f(row['illicit_share_within_2_hops'])} ({_f(row['illicit_share_overall'])}) | "
                f"{_f(row['illicit_found_within_2_hops'])} |"
            )
    L += [
        "",
        "## 3. Unseen scenarios (leave-one-scenario-out)",
        "",
        "Each row trains with that scenario removed entirely, then tests on it (plus held-out normal entities).",
        "",
        "| Held-out scenario | Test pos / neg | Rules only | RF behavioural | RF full | FUSED |",
        "|---|---|---|---|---|---|",
    ]
    for s, row in r["leave_one_scenario_out"].items():
        L.append(
            f"| {s} | {row['n_test_positive']} / {row['n_test_negative']} | {_f(row['rules_only']['pr_auc'])} | "
            f"{_f(row['RF_behavioural_only']['pr_auc'])} | {_f(row['RF_full']['pr_auc'])} | {_f(row['FUSED']['pr_auc'])} |"
        )
    L += [
        "",
        "(PR-AUC; base rate of positives in each test set is low, so chance is near 0.)",
        "",
        "## 4. Robustness (RF full, noisy or missing test features)",
        "",
        "| Perturbation | Level | PR-AUC | P@25 |",
        "|---|---|---|---|",
    ]
    for row in r["robustness"]:
        L.append(
            f"| {row['perturbation']} | {row['level']} | {_f(row['pr_auc'])} | {_f(row['precision@25'])} |"
        )
    return "\n".join(L) + "\n"


def _ci(m):
    c = m.get("pr_auc_ci95")
    return "n/a" if not c else f"[{c[0]:.3f}, {c[1]:.3f}]"
