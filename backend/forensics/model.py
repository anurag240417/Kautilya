"""Entity-level detection model: classifier + anomaly detector + explanations.

* **Classifier** - Random Forest on entity features (+ graph embeddings and
  propagation features).  Probabilities are calibrated with isotonic
  regression fitted on the forest's *out-of-bag* predictions, so a score of
  0.8 means roughly "80% of similar training entities were illicit".
* **Uncertainty** - the spread of the individual trees gives a 90% interval
  on each probability, reported alongside the point estimate.
* **Anomaly detector** - Isolation Forest (unsupervised), reported as a
  percentile rank.  Novelty is not guilt; it is fused with a safety cap.
* **Explanations** - model-agnostic *occlusion*: replace one feature (or the
  whole embedding block) with its training median and record how the
  calibrated probability changes.  Positive = pushes toward illicit.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.isotonic import IsotonicRegression

from backend.forensics.entities import FEATURE_DESCRIPTIONS
from backend.forensics.graphml import EMB_PREFIX, PROP_COLUMNS

logger = logging.getLogger(__name__)

EMB_GROUP = "graph_embedding"
Z90 = 1.645

EXTRA_DESCRIPTIONS = {
    EMB_GROUP: "position in the money-flow graph (who it transacts with)",
    "nbr_heuristic_exposure": "share of direct counterparties that show laundering patterns",
    "nbr2_heuristic_exposure": "share of counterparties-of-counterparties that show laundering patterns",
    "nbr_mean_pass_through": "how much its counterparties merely forward funds",
    "nbr_tor_share": "Tor usage among its counterparties",
}


def describe_feature(name: str) -> str:
    return FEATURE_DESCRIPTIONS.get(name) or EXTRA_DESCRIPTIONS.get(name, name.replace("_", " "))


@dataclass
class EntityModel:
    """Fit on labelled entities, score any entity table with the same columns."""

    base_cols: list[str]
    emb_cols: list[str] = field(default_factory=list)
    use_embeddings: bool = True
    n_estimators: int = 300
    seed: int = 42
    version: str = ""

    def __post_init__(self) -> None:
        self.rf: RandomForestClassifier | None = None
        self.iso: IsolationForest | None = None
        self.calibrator: IsotonicRegression | None = None
        self.medians: pd.Series | None = None
        self.iso_ref: np.ndarray | None = None
        self.n_train = 0
        self.n_pos = 0

    # ------------------------------------------------------------------
    @property
    def cols(self) -> list[str]:
        prop = [c for c in self.base_cols if c in PROP_COLUMNS]
        rest = [c for c in self.base_cols if c not in PROP_COLUMNS]
        return rest + prop + (self.emb_cols if self.use_embeddings else [])

    def _matrix(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.cols].fillna(self.medians[self.cols]).to_numpy(dtype=np.float64)

    def fit(
        self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray | None = None
    ) -> EntityModel:
        y = np.asarray(y).astype(int)
        if y.sum() < 2 or (1 - y).sum() < 2:
            msg = (
                f"Need at least 2 examples of each class (got {int(y.sum())} positive of {len(y)})"
            )
            raise ValueError(msg)
        self.medians = X[self.cols].median()
        A = self._matrix(X)
        self.rf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            min_samples_leaf=2,
            class_weight="balanced_subsample",
            oob_score=True,
            bootstrap=True,
            n_jobs=-1,
            random_state=self.seed,
        ).fit(A, y, sample_weight=sample_weight)
        oob = self.rf.oob_decision_function_[:, 1]
        ok = ~np.isnan(oob)
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0).fit(
            oob[ok], y[ok]
        )

        iso_cols = [c for c in self.cols if not c.startswith(EMB_PREFIX)]
        self.iso = IsolationForest(n_estimators=200, random_state=self.seed, n_jobs=-1).fit(
            X[iso_cols].fillna(self.medians[iso_cols]).to_numpy(dtype=np.float64)
        )
        self._iso_cols = iso_cols
        self.iso_ref = np.sort(
            -self.iso.score_samples(
                X[iso_cols].fillna(self.medians[iso_cols]).to_numpy(dtype=np.float64)
            )
        )
        self.n_train, self.n_pos = len(y), int(y.sum())
        logger.info(
            "EntityModel fit: n=%d positives=%d oob_acc=%.3f",
            self.n_train,
            self.n_pos,
            self.rf.oob_score_,
        )
        return self

    # ------------------------------------------------------------------
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Calibrated illicit probability."""
        raw = self.rf.predict_proba(self._matrix(X))[:, 1]
        return self.calibrator.predict(raw)

    def predict_interval(
        self, X: pd.DataFrame, z: float = Z90
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(p, p_low, p_high): calibrated mean and a tree-spread interval."""
        A = self._matrix(X)
        per_tree = np.stack([t.predict_proba(A)[:, 1] for t in self.rf.estimators_])
        mean, sd = per_tree.mean(axis=0), per_tree.std(axis=0)
        cal = self.calibrator.predict
        return cal(mean), cal(np.clip(mean - z * sd, 0, 1)), cal(np.clip(mean + z * sd, 0, 1))

    def anomaly_score(self, X: pd.DataFrame) -> np.ndarray:
        """Percentile rank of isolation score vs training population, in [0, 1]."""
        raw = -self.iso.score_samples(
            X[self._iso_cols].fillna(self.medians[self._iso_cols]).to_numpy(dtype=np.float64)
        )
        return np.searchsorted(self.iso_ref, raw, side="right") / len(self.iso_ref)

    def feature_importances(self) -> pd.Series:
        imp = pd.Series(self.rf.feature_importances_, index=self.cols)
        emb = imp[[c for c in imp.index if c.startswith(EMB_PREFIX)]].sum()
        imp = imp[[c for c in imp.index if not c.startswith(EMB_PREFIX)]]
        if self.use_embeddings and self.emb_cols:
            imp[EMB_GROUP] = emb
        return imp.sort_values(ascending=False)

    # ------------------------------------------------------------------
    def explain(self, X: pd.DataFrame, top_k: int = 6) -> list[list[dict]]:
        """Occlusion contributions for each row of ``X`` (usually a few alerts)."""
        if len(X) == 0:
            return []
        base_cols = [c for c in self.cols if not c.startswith(EMB_PREFIX)]
        groups: dict[str, list[str]] = {c: [c] for c in base_cols}
        if self.use_embeddings and self.emb_cols:
            groups[EMB_GROUP] = list(self.emb_cols)
        A = self._matrix(X)
        col_idx = {c: i for i, c in enumerate(self.cols)}
        base_p = self.calibrator.predict(self.rf.predict_proba(A)[:, 1])
        med = self.medians[self.cols].to_numpy()
        contrib = np.zeros((len(X), len(groups)))
        names = list(groups)
        for j, (name, cols) in enumerate(groups.items()):
            idx = [col_idx[c] for c in cols]
            B = A.copy()
            B[:, idx] = med[idx]
            contrib[:, j] = base_p - self.calibrator.predict(self.rf.predict_proba(B)[:, 1])
        out = []
        for i in range(len(X)):
            order = np.argsort(-np.abs(contrib[i]))[:top_k]
            row = X.iloc[i]
            items = []
            for j in order:
                name = names[j]
                if abs(contrib[i, j]) < 0.005:
                    continue
                items.append(
                    {
                        "feature": name,
                        "description": describe_feature(name),
                        "value": None if name == EMB_GROUP else _clean(row.get(name)),
                        "typical_value": None
                        if name == EMB_GROUP
                        else _clean(self.medians.get(name)),
                        "contribution": float(contrib[i, j]),
                        "direction": "raises risk" if contrib[i, j] > 0 else "lowers risk",
                    }
                )
            out.append(items)
        return out


def _clean(v):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else round(f, 6)
