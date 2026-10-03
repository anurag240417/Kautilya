"""Case management and analyst feedback (SQLite, fully offline).

Analysts confirm or reject alerts; those verdicts become training labels and
the model is retrained (``retrain_with_feedback``).  Feedback is keyed by a
*dataset fingerprint*, so verdicts recorded against one dataset are never
silently applied to another.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd

from backend.forensics.dataset import RawDataset

logger = logging.getLogger(__name__)

VERDICTS = ("confirmed", "false_positive", "unsure")
VERDICT_LABEL = {"confirmed": 1.0, "false_positive": 0.0}


def default_db_path() -> Path:
    return Path(os.environ.get("CHAINTRACE_CASE_DB", str(Path.cwd() / ".chaintrace" / "cases.db")))


def dataset_fingerprint(ds: RawDataset) -> str:
    """SHA-256 over the transaction ids and amounts, stable across runs."""
    h = hashlib.sha256()
    h.update(str(ds.n_tx).encode())
    h.update("".join(map(str, ds.tx["txid"].to_numpy())).encode())
    h.update(np.round(ds.outputs["amount"].to_numpy(), 8).tobytes())
    return h.hexdigest()


class CaseStore:
    """Thread-safe SQLite store for feedback and cases."""

    def __init__(self, path: Path | str | None = None) -> None:
        self.path = Path(path) if path else default_db_path()
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fingerprint TEXT NOT NULL, entity_id INTEGER NOT NULL,
                verdict TEXT NOT NULL, analyst TEXT, note TEXT, created REAL NOT NULL
            );
            CREATE INDEX IF NOT EXISTS ix_feedback ON feedback (fingerprint, entity_id);
            CREATE TABLE IF NOT EXISTS cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                fingerprint TEXT NOT NULL, title TEXT NOT NULL, entity_ids TEXT NOT NULL,
                status TEXT NOT NULL DEFAULT 'open', analyst TEXT, notes TEXT,
                created REAL NOT NULL, updated REAL NOT NULL
            );
            """
        )

    # ---------------- feedback ----------------
    def add_feedback(
        self, fingerprint: str, entity_id: int, verdict: str, analyst: str = "", note: str = ""
    ) -> dict:
        if verdict not in VERDICTS:
            msg = f"verdict must be one of {VERDICTS}"
            raise ValueError(msg)
        now = time.time()
        with self._lock, self._db:
            cur = self._db.execute(
                "INSERT INTO feedback (fingerprint, entity_id, verdict, analyst, note, created) VALUES (?,?,?,?,?,?)",
                (fingerprint, int(entity_id), verdict, analyst, note, now),
            )
        return {
            "id": cur.lastrowid,
            "entity_id": int(entity_id),
            "verdict": verdict,
            "analyst": analyst,
            "note": note,
            "created": now,
        }

    def list_feedback(self, fingerprint: str) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM feedback WHERE fingerprint=? ORDER BY id DESC", (fingerprint,)
            ).fetchall()
        return [dict(r) for r in rows]

    def feedback_labels(self, fingerprint: str) -> pd.Series:
        """Latest decisive verdict per entity as 1.0 / 0.0 (``unsure`` ignored)."""
        with self._lock:
            rows = self._db.execute(
                "SELECT entity_id, verdict FROM feedback WHERE fingerprint=? ORDER BY id",
                (fingerprint,),
            ).fetchall()
        latest: dict[int, float] = {}
        for r in rows:
            if r["verdict"] in VERDICT_LABEL:
                latest[r["entity_id"]] = VERDICT_LABEL[r["verdict"]]
            else:
                latest.pop(r["entity_id"], None)
        return pd.Series(latest, dtype=float)

    def clear_feedback(self, fingerprint: str) -> None:
        with self._lock, self._db:
            self._db.execute("DELETE FROM feedback WHERE fingerprint=?", (fingerprint,))

    # ---------------- cases ----------------
    def create_case(
        self,
        fingerprint: str,
        title: str,
        entity_ids: list[int],
        analyst: str = "",
        notes: str = "",
    ) -> dict:
        now = time.time()
        with self._lock, self._db:
            cur = self._db.execute(
                "INSERT INTO cases (fingerprint, title, entity_ids, analyst, notes, created, updated) VALUES (?,?,?,?,?,?,?)",
                (
                    fingerprint,
                    title,
                    json.dumps([int(e) for e in entity_ids]),
                    analyst,
                    notes,
                    now,
                    now,
                ),
            )
        return self.get_case(cur.lastrowid)

    def get_case(self, case_id: int) -> dict | None:
        with self._lock:
            r = self._db.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone()
        return _case_row(r) if r else None

    def list_cases(self, fingerprint: str) -> list[dict]:
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM cases WHERE fingerprint=? ORDER BY id DESC", (fingerprint,)
            ).fetchall()
        return [_case_row(r) for r in rows]

    def update_case(
        self, case_id: int, status: str | None = None, notes: str | None = None
    ) -> dict | None:
        sets, vals = ["updated=?"], [time.time()]
        if status:
            sets.append("status=?")
            vals.append(status)
        if notes is not None:
            sets.append("notes=?")
            vals.append(notes)
        with self._lock, self._db:
            self._db.execute(f"UPDATE cases SET {', '.join(sets)} WHERE id=?", (*vals, case_id))
        return self.get_case(case_id)

    def close(self) -> None:
        self._db.close()


def _case_row(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["entity_ids"] = json.loads(d["entity_ids"])
    return d


# ======================================================================
# Active learning + retraining
# ======================================================================


def simulate_seed_labels(
    truth: pd.DataFrame, active: np.ndarray, fraction: float = 0.1, seed: int = 0
) -> pd.Series:
    """Pretend a fraction of entities are already known (e.g. from a watch list).

    Only meaningful on synthetic data where ground truth exists.
    """
    rng = np.random.default_rng(seed)
    idx = np.flatnonzero(active & truth["is_illicit"].notna().to_numpy())
    pos = idx[truth["is_illicit"].to_numpy()[idx]]
    neg = idx[~truth["is_illicit"].to_numpy()[idx]]
    take_p = rng.choice(pos, max(4, int(len(pos) * fraction)), replace=False) if len(pos) else pos
    take_n = rng.choice(neg, max(40, int(len(neg) * fraction)), replace=False)
    sel = np.concatenate([take_p, take_n])
    return pd.Series(truth["is_illicit"].to_numpy()[sel].astype(float), index=sel)


def uncertain_entities(res, labelled: set[int], k: int = 10) -> pd.DataFrame:
    """Active-learning queue: alerts whose probability is closest to 0.5.

    Labelling these teaches the model the most.  Restricted to entities the
    pipeline already considers worth a look (non-zero fused score).
    """
    s = res.scores
    cand = s[(s["score"] > 0) & s["active"] & ~s.index.isin(list(labelled))].copy()
    cand["uncertainty"] = 1.0 - (cand["p"] - 0.5).abs() * 2.0
    return cand.sort_values(["uncertainty", "score"], ascending=[False, False]).head(k)


def retrain_with_feedback(
    res, store: CaseStore, fingerprint: str, seed_labels: pd.Series | None = None
):
    """Retrain on seed labels + analyst feedback; returns (new_result, comparison)."""
    from backend.forensics.pipeline import run_forensics

    fb = store.feedback_labels(fingerprint)
    labels = pd.concat([seed_labels if seed_labels is not None else pd.Series(dtype=float), fb])
    labels = labels[~labels.index.duplicated(keep="last")]  # feedback overrides seed
    if labels.sum() < 2 or (1 - labels).sum() < 2:
        msg = (
            "Need at least 2 confirmed and 2 rejected entities (seed labels + feedback) to retrain"
        )
        raise ValueError(msg)
    new = run_forensics(res.ds, res.cfg, labels=labels)
    return new, {
        "n_labels": int(len(labels)),
        "n_feedback": int(len(fb)),
        "n_positive": int(labels.sum()),
        "metrics": new.metrics.get("ranking_unlabelled", {}),
    }
