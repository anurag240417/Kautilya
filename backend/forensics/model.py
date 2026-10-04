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


# Feature families for group-level explanations. Many signals overlap (a layering chain also means
# fast pass-through and Tor use), so removing one feature rarely changes the score; removing or
# adding a whole family does.
FAMILY_PREFIX = "group:"
FAMILIES: dict[str, tuple[str, list[str]]] = {
    "structural": (
        "laundering patterns (peel chains, rapid-hop layering, dust, CoinJoin)",
        [
            "coinjoin_txs",
            "dust_spray_txs",
            "peel_chain_txs",
            "layering_chain_txs",
            "consolidation_txs",
            "batch_payout_txs",
            "max_peel_chain_len",
            "max_layering_chain_len",
        ],
    ),
    "network": (
        "network origin (Tor, hosting/VPN, IP and country behaviour)",
        [
            "tor_share",
            "vpn_share",
            "n_origin_ips",
            "n_origin_countries",
            "top_ip_share",
            "rapid_geo_hops",
            "network_obfuscation",
        ],
    ),
    "timing": (
        "timing (how fast funds move, bursts of activity)",
        [
            "active_hours",
            "burstiness",
            "median_gap_s",
            "peak_hour_events",
            "peak_hour_share",
            "median_dwell_s",
            "rapid_spend_share",
        ],
    ),
    "flow": (
        "money flow (pass-through, volumes, counterparties)",
        [
            "pass_through_ratio",
            "recv_to_sent_txs",
            "total_sent_btc",
            "total_recv_btc",
            "mean_sent_btc",
            "std_sent_btc",
            "mean_recv_btc",
            "flow_out_btc",
            "max_single_out_btc",
            "in_degree",
            "out_degree",
            "n_sent_txs",
            "n_recv_txs",
            "n_events",
            "mean_n_out_sent",
            "max_n_out_sent",
            "mean_n_in_sent",
            "max_n_in_sent",
            "round_payment_share",
            "dust_received",
            "recv_amount_cv",
            "n_addresses",
        ],
    ),
    "graph": (
        "position in the money-flow graph (who it transacts with)",
        [*PROP_COLUMNS],
    ),
}


def describe_feature(name: str) -> str:
    if name.startswith(FAMILY_PREFIX):
        return FAMILIES[name.removeprefix(FAMILY_PREFIX)][0]
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
        # Family-level contributions: average of (a) removing the family from this entity and
        # (b) adding the family to a typical entity. Stays informative when signals overlap.
        fam_names, fam_contrib = [], []
        base_all = np.tile(med, (len(X), 1))
        p_typical = self.calibrator.predict(self.rf.predict_proba(base_all)[:, 1])
        for fname, (_, cols) in FAMILIES.items():
            members = list(cols) + (
                list(self.emb_cols) if fname == "graph" and self.use_embeddings else []
            )
            idx = [col_idx[c] for c in members if c in col_idx]
            if not idx:
                continue
            removed = A.copy()
            removed[:, idx] = med[idx]
            added = base_all.copy()
            added[:, idx] = A[:, idx]
            d_out = base_p - self.calibrator.predict(self.rf.predict_proba(removed)[:, 1])
            d_in = self.calibrator.predict(self.rf.predict_proba(added)[:, 1]) - p_typical
            fam_names.append(FAMILY_PREFIX + fname)
            fam_contrib.append((d_out + d_in) / 2.0)
        fam_contrib = np.array(fam_contrib).T if fam_contrib else np.zeros((len(X), 0))

        out = []
        for i in range(len(X)):
            row = X.iloc[i]
            items = []
            for j in range(len(names)):
                if abs(contrib[i, j]) >= 0.005:
                    items.append((names[j], contrib[i, j]))
            for j in range(len(fam_names)):
                if abs(fam_contrib[i, j]) >= 0.005:
                    items.append((fam_names[j], fam_contrib[i, j]))
            items.sort(key=lambda t: -abs(t[1]))
            rows = []
            for name, c in items[:top_k]:
                is_group = name == EMB_GROUP or name.startswith(FAMILY_PREFIX)
                rows.append(
                    {
                        "feature": name,
                        "description": describe_feature(name),
                        "value": None if is_group else _clean(row.get(name)),
                        "typical_value": None if is_group else _clean(self.medians.get(name)),
                        "contribution": float(c),
                        "direction": "raises risk" if c > 0 else "lowers risk",
                    }
                )
            out.append(rows)
        return out


def _clean(v):
    if v is None:
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if np.isnan(f) else round(f, 6)
