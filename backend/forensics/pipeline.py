"""End-to-end forensics pipeline: dataset -> ranked, explainable alerts.

Stages (each timed): heuristics -> entity resolution + features -> graph
signals -> cross-fitted model -> multi-signal fusion -> ranking.

*Cross-fitting*: when labels exist, entities are split into folds by
campaign and every entity is scored by a model that never saw it (or its
campaign) in training.  Dashboard scores are therefore honest, not in-sample.

*Fusion* reuses ``backend.risk.scorer.synthesize_risk_score`` so these alerts
share tiers, weights and safety caps with the rest of ChainTrace.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backend.domain.risk import SignalInput
from backend.forensics.dataset import RawDataset
from backend.forensics.entities import (
    EntityTable,
    build_entities,
    clustering_quality,
    first_observation,
)
from backend.forensics.graphml import (
    EMB_PREFIX,
    GraphSignals,
    PeerGroups,
    compute_graph_signals,
    peer_groups,
)
from backend.forensics.heuristics import (
    HeuristicConfig,
    TxHeuristics,
    detect_heuristics,
    evaluate_heuristics,
)
from backend.forensics.model import EntityModel
from backend.risk.scorer import SynthesisConfig, synthesize_risk_score

logger = logging.getLogger(__name__)

BASE_EXCLUDE = {"wallet_ip_link_lo", "wallet_ip_link_hi"}


@dataclass
class ForensicsConfig:
    embedding_dim: int = 16
    n_folds: int = 4
    n_estimators: int = 300
    top_alerts: int = 300
    use_embeddings: bool = True
    min_events: int = 2
    seed: int = 42
    max_train_negatives: int = 40_000
    n_peer_groups: int = 8
    train_synth_n_tx: int = 30_000  # when the input has no labels


@dataclass
class Prepared:
    h: TxHeuristics
    et: EntityTable
    graph: GraphSignals
    X: pd.DataFrame
    active: np.ndarray
    first_obs: pd.DataFrame


@dataclass
class ForensicsResult:
    ds: RawDataset
    cfg: ForensicsConfig
    h: TxHeuristics
    et: EntityTable
    graph: GraphSignals
    X: pd.DataFrame
    first_obs: pd.DataFrame
    scores: pd.DataFrame
    peer: PeerGroups
    models: list[EntityModel]
    entity_fold: np.ndarray
    labels: pd.DataFrame | None
    label_source: str
    metrics: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    n_ranked: int = 0
    model_version: str = ""
    alerts: list[dict] = field(default_factory=list)

    def model_for(self, eid: int) -> EntityModel:
        return self.models[int(self.entity_fold[eid])]

    def alert_table(self, limit: int | None = None) -> pd.DataFrame:
        t = self.scores[self.scores["rank"] > 0].sort_values("rank")
        return t.head(limit) if limit else t


# ======================================================================
# Scoring components
# ======================================================================


def structural_score(F: pd.DataFrame, prop: pd.DataFrame) -> np.ndarray:
    """Noisy-OR of structural laundering indicators, in [0, 1]."""
    terms = [
        0.55
        * (F["peel_chain_txs"] > 0)
        * np.minimum(1.0, F["max_peel_chain_len"] / 8.0).clip(lower=0.5),
        0.60 * (F["layering_chain_txs"] > 0),
        0.45 * (F["dust_spray_txs"] > 0),
        0.12 * (F["coinjoin_txs"] > 0),
        0.08 * (F["consolidation_txs"] >= 2),
        0.25 * prop["nbr_heuristic_exposure"].clip(0, 1),
    ]
    out = np.ones(len(F))
    for t in terms:
        out *= 1.0 - np.asarray(t, dtype=float)
    return 1.0 - out


def temporal_burst_score(F: pd.DataFrame) -> np.ndarray:
    return np.clip(
        0.6 * np.minimum(1.0, F["peak_hour_events"] / 10.0) * (F["n_events"] >= 5)
        + 0.4 * np.maximum(0.0, F["burstiness"]),
        0.0,
        1.0,
    ).to_numpy()


FUSION = SynthesisConfig(custom_weights={"network_obfuscation": 0.15, "temporal_burst": 0.10})


def fuse(
    p: np.ndarray,
    anomaly: np.ndarray,
    heuristic: np.ndarray,
    network: np.ndarray,
    burst: np.ndarray,
):
    """Run each row through the production risk synthesizer."""
    scores = np.zeros(len(p))
    tiers, active = [], []
    for i in range(len(p)):
        rs = synthesize_risk_score(
            SignalInput(
                entity_id=str(i),
                entity_type="wallet",
                illicit_probability=float(np.clip(p[i], 0, 1)),
                anomaly_score=float(np.clip(anomaly[i], 0, 1)),
                graph_signal=float(np.clip(heuristic[i], 0, 1)),
                custom_signals={
                    "network_obfuscation": float(np.clip(network[i], 0, 1)),
                    "temporal_burst": float(np.clip(burst[i], 0, 1)),
                },
            ),
            FUSION,
        )
        scores[i] = rs.score
        tiers.append(rs.priority_tier.value)
        active.append(",".join(rs.active_signals))
    return scores, tiers, active


# ======================================================================
# Preparation (shared by training data and target data)
# ======================================================================


def prepare(
    ds: RawDataset,
    cfg: ForensicsConfig,
    hcfg: HeuristicConfig | None = None,
    timings: dict | None = None,
) -> Prepared:
    timings = timings if timings is not None else {}
    t = time.perf_counter()
    h = detect_heuristics(ds, hcfg)
    timings["heuristics_s"] = time.perf_counter() - t
    t = time.perf_counter()
    et = build_entities(ds, h)
    timings["entities_s"] = time.perf_counter() - t
    t = time.perf_counter()
    graph = compute_graph_signals(et.flows, et.features, dim=cfg.embedding_dim, seed=cfg.seed)
    timings["graph_signals_s"] = time.perf_counter() - t
    X = (
        et.features.drop(columns=[c for c in BASE_EXCLUDE if c in et.features.columns])
        .join(graph.propagation)
        .join(graph.embeddings)
    )
    active = (et.features["n_events"] >= cfg.min_events).to_numpy()
    return Prepared(h, et, graph, X, active, first_observation(ds))


def _base_cols(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if not c.startswith(EMB_PREFIX)]


def _emb_cols(X: pd.DataFrame) -> list[str]:
    return [c for c in X.columns if c.startswith(EMB_PREFIX)]


def entity_campaigns(ds: RawDataset, addr_cluster: np.ndarray, n_ent: int) -> np.ndarray:
    """Campaign id per entity (-1 for none); synthetic truth only."""
    camp = ds.truth.entities["campaign"].to_numpy()[ds.truth.addr_entity]
    s = pd.Series(camp).groupby(addr_cluster).max()
    return s.reindex(range(n_ent), fill_value=-1).to_numpy()


def assign_folds(groups: np.ndarray, n_folds: int, seed: int) -> np.ndarray:
    """Fold per entity; entities sharing a campaign share a fold."""
    rng = np.random.default_rng(seed)
    fold = rng.integers(0, n_folds, len(groups))
    for g in np.unique(groups[groups >= 0]):
        fold[groups == g] = rng.integers(0, n_folds)
    return fold


def _train_subset(
    idx: np.ndarray, y: np.ndarray, cap_neg: int, rng: np.random.Generator
) -> np.ndarray:
    pos = idx[y[idx] == 1]
    neg = idx[y[idx] == 0]
    if len(neg) > cap_neg:
        neg = rng.choice(neg, cap_neg, replace=False)
    return np.concatenate([pos, neg])


def fit_model(
    X: pd.DataFrame,
    y: np.ndarray,
    train_idx: np.ndarray,
    cfg: ForensicsConfig,
    version: str,
    use_embeddings: bool | None = None,
) -> EntityModel:
    m = EntityModel(
        base_cols=_base_cols(X),
        emb_cols=_emb_cols(X),
        use_embeddings=cfg.use_embeddings if use_embeddings is None else use_embeddings,
        n_estimators=cfg.n_estimators,
        seed=cfg.seed,
        version=version,
    )
    return m.fit(X.iloc[train_idx], y[train_idx])


# ======================================================================
# Main entry point
# ======================================================================


def run_forensics(
    ds: RawDataset,
    cfg: ForensicsConfig | None = None,
    labels: pd.Series | None = None,
    hcfg: HeuristicConfig | None = None,
) -> ForensicsResult:
    """Score every entity in ``ds`` and rank investigative leads.

    ``labels``: optional Series (index = entity id, values 1/0, NaN unknown) that
    overrides ground truth, e.g. analyst feedback.  Without labels and without
    ground truth the model is trained on an internally generated synthetic
    dataset and transferred (flagged in ``label_source``).
    """
    cfg = cfg or ForensicsConfig()
    timings: dict = {}
    t_all = time.perf_counter()
    prep = prepare(ds, cfg, hcfg, timings)
    et, X, active = prep.et, prep.X, prep.active
    n_ent = et.n_entities
    rng = np.random.default_rng(cfg.seed)
    version = f"entity_rf_{cfg.seed}_{n_ent}"

    truth_labels = et.labels(ds)
    y = np.full(n_ent, np.nan)
    if labels is not None:
        y[labels.index.to_numpy()] = labels.to_numpy(dtype=float)
        label_source = "analyst_labels"
    elif truth_labels is not None:
        y = truth_labels["is_illicit"].to_numpy(dtype=float)
        label_source = "ground_truth_synthetic"
    else:
        label_source = "transfer_from_synthetic"

    t = time.perf_counter()
    models: list[EntityModel] = []
    fold_of = np.zeros(n_ent, dtype=np.int64)
    p = np.zeros(n_ent)
    anomaly = np.zeros(n_ent)

    if label_source == "transfer_from_synthetic":
        from backend.forensics.synth import SynthConfig, generate_dataset

        tds = generate_dataset(SynthConfig(n_tx=cfg.train_synth_n_tx, seed=cfg.seed + 1))
        tp = prepare(tds, cfg)
        ty = tp.et.labels(tds)["is_illicit"].to_numpy(dtype=float)
        idx = np.flatnonzero(tp.active & ~np.isnan(ty))
        sel = _train_subset(idx, ty, cfg.max_train_negatives, rng)
        m = fit_model(tp.X, ty, sel, cfg, version)
        models = [m]
        ai = np.flatnonzero(active)
        p[ai] = m.predict_proba(X.iloc[ai])
        anomaly[ai] = m.anomaly_score(X.iloc[ai])
    else:
        labelled = active & ~np.isnan(y)
        if ds.truth is not None:
            groups = entity_campaigns(ds, et.addr_cluster, n_ent)
        else:
            groups = np.full(n_ent, -1)
        n_folds = cfg.n_folds if labelled.sum() >= 200 else 2
        folds = assign_folds(groups, n_folds, cfg.seed)
        fold_of = folds
        for f in range(n_folds):
            tr = np.flatnonzero(labelled & (folds != f))
            sel = _train_subset(tr, y, cfg.max_train_negatives, rng)
            m = fit_model(X, y, sel, cfg, f"{version}_f{f}")
            models.append(m)
            te = np.flatnonzero(active & (folds == f))
            if len(te):
                p[te] = m.predict_proba(X.iloc[te])
                anomaly[te] = m.anomaly_score(X.iloc[te])
    timings["model_s"] = time.perf_counter() - t

    # ---- fusion ------------------------------------------------------------
    t = time.perf_counter()
    F = et.features
    heur = structural_score(F, prep.graph.propagation)
    network = F["network_obfuscation"].to_numpy()
    burst = temporal_burst_score(F)
    cand = active & ((p >= 0.05) | (heur > 0) | (anomaly >= 0.98) | (network >= 0.5))
    ci = np.flatnonzero(cand)
    score = np.zeros(n_ent)
    tier = np.array(["low"] * n_ent, dtype=object)
    act = np.array([""] * n_ent, dtype=object)
    if len(ci):
        s, tr_, ac = fuse(p[ci], anomaly[ci], heur[ci], network[ci], burst[ci])
        score[ci], tier[ci], act[ci] = s, tr_, ac
    timings["fusion_s"] = time.perf_counter() - t

    order = np.argsort(-score, kind="stable")
    rank = np.zeros(n_ent, dtype=np.int64)
    ranked = order[score[order] > 0]
    rank[ranked] = np.arange(1, len(ranked) + 1)

    t = time.perf_counter()
    pg = peer_groups(X[[c for c in X.columns]], active, cfg.n_peer_groups, cfg.seed)
    timings["peer_groups_s"] = time.perf_counter() - t

    is_service = ((F["in_degree"] >= 30) & (F["out_degree"] >= 8)) | (F["n_addresses"] >= 50)
    scores = pd.DataFrame(
        {
            "p": p,
            "anomaly": anomaly,
            "heuristic": heur,
            "network": network,
            "burst": burst,
            "score": score,
            "tier": tier,
            "active_signals": act,
            "rank": rank,
            "peer_group": pg.labels,
            "is_service": is_service.to_numpy(),
            "active": active,
        }
    )

    res = ForensicsResult(
        ds=ds,
        cfg=cfg,
        h=prep.h,
        et=et,
        graph=prep.graph,
        X=X,
        first_obs=prep.first_obs,
        scores=scores,
        peer=pg,
        models=models,
        entity_fold=fold_of,
        labels=truth_labels,
        label_source=label_source,
        timings=timings,
        n_ranked=int(active.sum()),
        model_version=version,
    )
    ytruth = truth_labels["is_illicit"].to_numpy(dtype=float) if truth_labels is not None else None
    res.metrics = compute_metrics(
        res, ytruth, labelled_mask=~np.isnan(y) if label_source == "analyst_labels" else None
    )
    timings["total_s"] = time.perf_counter() - t_all
    logger.info(
        "Forensics pipeline done in %.1fs: %s",
        timings["total_s"],
        {k: round(v, 2) for k, v in timings.items()},
    )
    return res


# ======================================================================
# Metrics
# ======================================================================


def compute_metrics(
    res: ForensicsResult, y: np.ndarray | None, labelled_mask: np.ndarray | None = None
) -> dict:
    """Dataset-level summary: heuristics vs truth, clustering quality, ranking quality.

    ``y`` is ground truth (synthetic only).  If ``labelled_mask`` is given the
    ranking is scored only on entities *outside* it (the ones the model was
    not told about), reported as ``ranking_unlabelled``.
    """
    from backend.ml.benchmark import ranking_metrics

    out: dict = {
        "dataset": res.ds.summary(),
        "heuristic_counts": res.h.counts(),
        "label_source": res.label_source,
    }
    if res.ds.truth is None or y is None:
        return out
    out["heuristic_validation"] = evaluate_heuristics(res.ds, res.h)
    out["clustering"] = clustering_quality(res.ds, res.et.addr_cluster)
    m = res.scores["active"].to_numpy() & ~np.isnan(y)
    key = "ranking"
    if labelled_mask is not None:
        m &= ~labelled_mask
        key = "ranking_unlabelled"
    if y[m].sum() >= 1 and (1 - y[m]).sum() >= 1:
        yy = y[m].astype(int)
        S = res.scores[m]
        ks = (10, 25, 50, 100)
        out[key] = {
            "fused_score": ranking_metrics(yy, S["score"].to_numpy(), ks=ks),
            "model_probability": ranking_metrics(yy, S["p"].to_numpy(), ks=ks),
            "structural_heuristics_only": ranking_metrics(yy, S["heuristic"].to_numpy(), ks=ks),
            "anomaly_only": ranking_metrics(yy, S["anomaly"].to_numpy(), ks=ks),
        }
    return out
