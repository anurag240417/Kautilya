"""Stateful service behind the forensics API: load data, score, query, learn.

Holds the current ``ForensicsResult``; all public methods are thread-safe
(the WSGI server may handle requests concurrently).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

from backend.forensics.cases import (
    CaseStore,
    dataset_fingerprint,
    retrain_with_feedback,
    simulate_seed_labels,
    uncertain_entities,
)
from backend.forensics.evidence import build_entity_report
from backend.forensics.graphview import entity_subgraph
from backend.forensics.ingest import load_raw_transactions
from backend.forensics.pipeline import ForensicsConfig, ForensicsResult, run_forensics
from backend.forensics.report import render_report_html
from backend.forensics.synth import SynthConfig, generate_dataset

logger = logging.getLogger(__name__)

def _default_tx() -> int:
    """Default synthetic dataset size; lower it with CHAINTRACE_DEFAULT_TX on small hosts."""
    try:
        return min(max(int(os.environ.get("CHAINTRACE_DEFAULT_TX", "20000")), 500), 400_000)
    except ValueError:
        return 20_000


DEFAULT_SYNTH_TX = _default_tx()


def data_dir() -> Path:
    return Path(os.environ.get("CHAINTRACE_DATA_DIR", str(Path.cwd() / "data"))).resolve()


def safe_data_path(rel: str) -> Path:
    """Resolve ``rel`` inside the data directory; reject anything outside it."""
    base = data_dir()
    p = (base / rel).resolve()
    if base != p and base not in p.parents:
        msg = "path must be inside the data directory"
        raise ValueError(msg)
    if not p.is_file():
        msg = f"file not found in data directory: {rel}"
        raise ValueError(msg)
    return p


def list_data_files() -> list[dict]:
    """Datasets the analyst can load: supported files directly inside the data directory."""
    base = data_dir()
    if not base.is_dir():
        return []
    out = []
    for p in sorted(base.iterdir()):
        if p.is_file() and p.suffix.lower() in (
            ".csv",
            ".tsv",
            ".json",
            ".jsonl",
            ".ndjson",
            ".xml",
        ):
            out.append({"path": p.name, "size_mb": round(p.stat().st_size / 1e6, 2)})
    return out


def _reasons(res: ForensicsResult, eid: int) -> list[str]:
    f = res.et.features.loc[eid]
    r = []
    if f["peel_chain_txs"] > 0:
        r.append("peel chain")
    if f["layering_chain_txs"] > 0:
        r.append("rapid-hop layering")
    if f["dust_spray_txs"] > 0:
        r.append("dust spray")
    if f["coinjoin_txs"] > 0:
        r.append("CoinJoin")
    if f["tor_share"] >= 0.25:
        r.append("Tor origin")
    if f["vpn_share"] >= 0.25:
        r.append("hosting/VPN origin")
    if f["pass_through_ratio"] >= 0.95 and f["n_sent_txs"] and f["n_recv_txs"]:
        r.append("pass-through")
    if f["recv_amount_cv"] >= 0 and f["recv_amount_cv"] < 0.15:
        r.append("uniform inbound amounts")
    if f["max_n_in_sent"] >= 10:
        r.append("consolidation")
    return r


class ForensicsService:
    def __init__(self, store: CaseStore | None = None) -> None:
        self._lock = threading.RLock()
        self._store = store
        self.res: ForensicsResult | None = None
        self.fingerprint = ""
        self.base_res: ForensicsResult | None = None  # result before any retrain
        self.last_retrain: dict | None = None
        self.loaded_at = 0.0
        self.loading = False

    @property
    def store(self) -> CaseStore:
        with self._lock:
            if self._store is None:
                self._store = CaseStore()
            return self._store

    # ------------------------------------------------------------ loading
    def _install(self, res: ForensicsResult) -> None:
        self.res = self.base_res = res
        self.fingerprint = dataset_fingerprint(res.ds)
        self.loaded_at = time.time()
        self.last_retrain = None

    def load_synthetic(
        self, n_tx: int = DEFAULT_SYNTH_TX, seed: int = 7, n_estimators: int = 200
    ) -> dict:
        if not 500 <= n_tx <= 400_000:
            msg = "n_tx must be between 500 and 400000"
            raise ValueError(msg)
        with self._lock:
            self.loading = True
            try:
                ds = generate_dataset(SynthConfig(n_tx=n_tx, seed=seed))
                self._install(
                    run_forensics(ds, ForensicsConfig(n_estimators=n_estimators, seed=seed))
                )
            finally:
                self.loading = False
        return self.status()

    def load_file(self, rel_path: str, n_estimators: int = 200) -> dict:
        path = safe_data_path(rel_path)
        with self._lock:
            self.loading = True
            try:
                ds = load_raw_transactions(path)
                self._install(run_forensics(ds, ForensicsConfig(n_estimators=n_estimators)))
            finally:
                self.loading = False
        return self.status()

    def ensure_loaded(self) -> ForensicsResult:
        with self._lock:
            if self.res is None:
                self.load_synthetic()
            return self.res  # type: ignore[return-value]

    # ------------------------------------------------------------ queries
    def status(self) -> dict:
        with self._lock:
            if self.res is None:
                return {"loaded": False, "loading": self.loading}
            r = self.res
            return {
                "loaded": True,
                "loading": self.loading,
                "dataset": r.ds.summary(),
                "ingest": r.ds.report.to_dict() if r.ds.report else None,
                "fingerprint": self.fingerprint,
                "label_source": r.label_source,
                "model_version": r.model_version,
                "timings_s": {k: round(v, 3) for k, v in r.timings.items()},
                "metrics": _jsonable(r.metrics),
                "n_entities": int(r.et.n_entities),
                "n_active_entities": int(r.n_ranked),
                "n_alerts": int((r.scores["rank"] > 0).sum()),
                "default_n_tx": DEFAULT_SYNTH_TX,
                "peer_groups": _jsonable(r.peer.descriptions),
                "last_retrain": self.last_retrain,
                "n_feedback": len(self.store.list_feedback(self.fingerprint)),
                "geoip_source": _geo_source(),
            }

    def alerts(
        self,
        limit: int = 50,
        offset: int = 0,
        min_score: float = 0.0,
        tier: str | None = None,
        search: str | None = None,
    ) -> dict:
        r = self.ensure_loaded()
        t = r.scores[(r.scores["rank"] > 0) & (r.scores["score"] >= min_score)]
        if tier:
            t = t[t["tier"] == tier]
        if search:
            s = search.strip().lower().removeprefix("e-")
            if s.isdigit():
                t = t[t.index == int(s)]
            else:
                # A TXID (or an 8+ character prefix of one): show every entity that sent
                # or received in that transaction.
                txm = np.flatnonzero(
                    r.ds.tx["txid"].astype(str).str.lower().str.startswith(s).to_numpy()
                ) if len(s) >= 8 else np.array([], dtype=np.int64)
                ents: set[int] = set()
                if len(txm):
                    cl = r.et.addr_cluster
                    for frame in (r.ds.inputs, r.ds.outputs):
                        ents |= set(cl[frame["addr"].to_numpy()[np.isin(frame["tx_idx"].to_numpy(), txm)]])
                ids = np.flatnonzero(
                    pd.Series(r.ds.addresses).str.lower().str.contains(s, regex=False).to_numpy()
                )
                ents |= set(np.unique(r.et.addr_cluster[ids]).tolist())
                t = t[t.index.isin(list(ents))]
        total = len(t)
        t = t.sort_values("rank").iloc[offset : offset + limit]
        fb = {
            f["entity_id"]: f["verdict"]
            for f in reversed(self.store.list_feedback(self.fingerprint))
        }
        rows = []
        for eid, s in t.iterrows():
            eid = int(eid)
            row = {
                "entity_id": eid,
                "label": f"E-{eid}",
                "rank": int(s["rank"]),
                "score": float(s["score"]),
                "tier": str(s["tier"]),
                "illicit_probability": round(float(s["p"]), 4),
                "anomaly_percentile": round(float(s["anomaly"]), 4),
                "structural_score": round(float(s["heuristic"]), 4),
                "network_obfuscation": round(float(s["network"]), 4),
                "n_addresses": int(r.et.features.at[eid, "n_addresses"]),
                "reasons": _reasons(r, eid),
                "verdict": fb.get(eid),
            }
            if r.labels is not None:
                row["truth_illicit"] = bool(r.labels.at[eid, "is_illicit"])
                row["truth_scenario"] = str(r.labels.at[eid, "scenario"])
            rows.append(row)
        return {"total": int(total), "offset": offset, "limit": limit, "alerts": rows}

    def entity(self, eid: int) -> dict:
        r = self.ensure_loaded()
        self._check(r, eid)
        rep = build_entity_report(r, eid)
        rep["feedback"] = [
            f for f in self.store.list_feedback(self.fingerprint) if f["entity_id"] == eid
        ]
        return _jsonable(rep)

    def graph(self, eid: int, radius: int = 2, max_nodes: int = 60) -> dict:
        r = self.ensure_loaded()
        self._check(r, eid)
        return _jsonable(
            entity_subgraph(r, eid, min(max(radius, 1), 3), min(max(max_nodes, 5), 150))
        )

    @staticmethod
    def _check(r: ForensicsResult, eid: int) -> None:
        if not 0 <= eid < r.et.n_entities:
            msg = f"unknown entity {eid}"
            raise KeyError(msg)

    # ------------------------------------------------------------ learning
    def add_feedback(self, eid: int, verdict: str, analyst: str = "", note: str = "") -> dict:
        r = self.ensure_loaded()
        self._check(r, eid)
        return self.store.add_feedback(self.fingerprint, eid, verdict, analyst, note)

    def feedback(self) -> list[dict]:
        self.ensure_loaded()
        return self.store.list_feedback(self.fingerprint)

    def uncertain(self, k: int = 10) -> list[dict]:
        r = self.ensure_loaded()
        done = set(self.store.feedback_labels(self.fingerprint).index)
        u = uncertain_entities(r, done, k)
        return [
            {
                "entity_id": int(i),
                "label": f"E-{i}",
                "score": float(row["score"]),
                "illicit_probability": round(float(row["p"]), 4),
                "uncertainty": round(float(row["uncertainty"]), 4),
            }
            for i, row in u.iterrows()
        ]

    def retrain(self, seed_fraction: float = 0.1) -> dict:
        """Retrain on seed labels (+ analyst feedback); report before/after on unseen entities."""
        r = self.ensure_loaded()
        with self._lock:
            seed = None
            if r.labels is not None:
                seed = simulate_seed_labels(
                    r.labels, r.scores["active"].to_numpy(), seed_fraction, r.cfg.seed
                )
            before_labels = seed if seed is not None else pd.Series(dtype=float)
            fb = self.store.feedback_labels(self.fingerprint)
            if len(before_labels) >= 4:
                before = run_forensics(r.ds, r.cfg, labels=before_labels).metrics.get(
                    "ranking_unlabelled", {}
                )
            else:
                before = {}
            new, info = retrain_with_feedback(r, self.store, self.fingerprint, seed)
            self.res = new
            summary = {
                "seed_labels": int(len(before_labels)),
                "feedback_labels": int(len(fb)),
                "before_feedback": _pick(before),
                "after_feedback": _pick(info["metrics"]),
                "note": "Metrics are measured on entities that were NOT in the training labels.",
            }
            self.last_retrain = summary
            return summary

    def reset_model(self) -> dict:
        with self._lock:
            if self.base_res is not None:
                self.res = self.base_res
                self.last_retrain = None
        return self.status()

    # ------------------------------------------------------------ reports & cases
    def report_html(self, entity_ids: list[int], title: str, analyst: str, notes: str = "") -> str:
        r = self.ensure_loaded()
        for e in entity_ids:
            self._check(r, e)
        if not 1 <= len(entity_ids) <= 20:
            msg = "choose between 1 and 20 entities"
            raise ValueError(msg)
        return render_report_html(r, entity_ids, title, analyst, notes)

    def create_case(
        self, title: str, entity_ids: list[int], analyst: str = "", notes: str = ""
    ) -> dict:
        self.ensure_loaded()
        return self.store.create_case(self.fingerprint, title, entity_ids, analyst, notes)

    def cases(self) -> list[dict]:
        self.ensure_loaded()
        return self.store.list_cases(self.fingerprint)


def _pick(m: dict) -> dict:
    out = {}
    for k in ("fused_score", "model_probability", "structural_heuristics_only"):
        if k in m:
            out[k] = {
                "pr_auc": m[k].get("pr_auc"),
                "precision@50": m[k].get("precision@50"),
                "precision@10": m[k].get("precision@10"),
                "n_positive": m[k].get("n_positive"),
            }
    return out


def _geo_source() -> str:
    from backend.forensics.geo import default_geoip

    return default_geoip().source


def _jsonable(o):
    """Recursively convert numpy/pandas scalars so ``json.dumps`` accepts them."""
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        f = float(o)
        return None if np.isnan(f) else f
    if isinstance(o, float) and np.isnan(o):
        return None
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return [_jsonable(v) for v in o.tolist()]
    return o


_service: ForensicsService | None = None
_service_lock = threading.Lock()


def get_forensics_service() -> ForensicsService:
    global _service
    with _service_lock:
        if _service is None:
            _service = ForensicsService()
        return _service


def set_forensics_service(s: ForensicsService | None) -> None:
    global _service
    with _service_lock:
        _service = s
