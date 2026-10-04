"""Risk propagation from seed illicit wallets.

Given a set of *seed* entities that are already known to be illicit (a watch list, or leads an
analyst has confirmed), spread risk outward through the money-flow graph and report, for every
other entity, how close it is to a seed.

* **Algorithm:** personalised PageRank (random walk with restart at the seeds), computed by
  power iteration on the sparse transition matrix.  Mass flows along edges in both directions
  (funds and counterparties), weighted by transaction value, and fades with distance.
* **Hub handling:** service-like entities (exchange-style hubs) are *absorbing*: they do not pass
  risk on, and are not themselves given risk.  Otherwise one exchange would taint every one of its
  customers, and the exchange itself would be flagged for receiving a criminal's cash-out.
* **Hops:** unweighted shortest distance to the nearest seed, so each alert can say
  "2 hops from known-illicit E-1234".
* **Boost, not a penalty:** ``seed_boost`` only ever raises a fused score.  Entities far from every
  seed are unaffected, so campaigns without a seed are not pushed down.

Proximity to a seed is a lead, not guilt: innocent counterparties of illicit entities (victims,
exchanges) are close to seeds too.  It is reported as correlation with the hop count shown.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import dijkstra

from backend.forensics.graphml import _row_normalise

logger = logging.getLogger(__name__)


@dataclass
class SeedRisk:
    risk: np.ndarray  # [0, 1]; 1.0 = at least as risky as a typical seed
    hops: np.ndarray  # distance to the nearest seed; -1 = not within max_hops
    nearest: np.ndarray  # entity id of the nearest seed; -1 = none
    is_seed: np.ndarray  # bool
    n_seeds: int


def propagate_seed_risk(
    adj: sparse.csr_matrix,
    seeds: np.ndarray,
    absorbing: np.ndarray | None = None,
    alpha: float = 0.15,
    iters: int = 40,
    max_hops: int = 4,
) -> SeedRisk:
    """Spread risk from ``seeds`` over the symmetric weighted adjacency ``adj``.

    ``absorbing``: boolean mask of hub entities that neither pass risk on nor are given any (seeds excepted).
    ``alpha``: restart probability (higher = risk stays closer to the seeds).
    Entities more than ``max_hops`` from every seed get zero risk.
    """
    n = adj.shape[0]
    seeds = np.unique(np.asarray(seeds, dtype=np.int64))
    seeds = seeds[(seeds >= 0) & (seeds < n)]
    risk = np.zeros(n)
    hops = np.full(n, -1, dtype=np.int64)
    nearest = np.full(n, -1, dtype=np.int64)
    is_seed = np.zeros(n, dtype=bool)
    if len(seeds) == 0:
        return SeedRisk(risk, hops, nearest, is_seed, 0)

    is_seed[seeds] = True
    if absorbing is not None:
        absorbing = (
            np.asarray(absorbing, dtype=bool) & ~is_seed
        )  # a known-illicit seed always spreads
    hops[seeds], nearest[seeds], risk[seeds] = 0, seeds, 1.0
    if adj.nnz == 0:
        return SeedRisk(risk, hops, nearest, is_seed, int(len(seeds)))

    a = adj.tocsr().astype(float)
    if absorbing is not None and absorbing.any():
        a = sparse.diags((~absorbing).astype(float)) @ a  # absorbing nodes: no outgoing risk
    transition_t = _row_normalise(a).T.tocsr()

    restart = np.zeros(n)
    restart[seeds] = 1.0 / len(seeds)
    r = restart.copy()
    for _ in range(iters):
        r = alpha * restart + (1.0 - alpha) * (transition_t @ r)

    reference = float(np.median(r[seeds]))
    if reference > 0:
        scaled = np.clip(r / reference, 0.0, 1.0)
    else:
        scaled = np.zeros(n)

    binary = a.copy()
    binary.data[:] = 1.0
    dist, _, source = dijkstra(
        binary,
        directed=True,
        indices=seeds,
        unweighted=True,
        limit=max_hops,
        min_only=True,
        return_predecessors=True,
    )
    reach = np.isfinite(dist)
    hops[reach] = dist[reach].astype(np.int64)
    nearest[reach] = source[reach]

    near = hops >= 0
    risk[near & ~is_seed] = scaled[near & ~is_seed]
    if absorbing is not None:
        # A hub (exchange-style service) is where funds end up, not a suspect: it keeps its hop
        # distance for display but receives no risk of its own.
        risk[absorbing & ~is_seed] = 0.0
    logger.info(
        "Seed risk: %d seeds, %d entities within %d hops",
        len(seeds),
        int(near.sum()) - len(seeds),
        max_hops,
    )
    return SeedRisk(risk, hops, nearest, is_seed, int(len(seeds)))


def seed_boost(score: np.ndarray, risk: np.ndarray, weight: float = 0.5) -> np.ndarray:
    """Raise ``score`` (0-100) toward 100 in proportion to seed proximity. Never lowers a score."""
    score = np.asarray(score, dtype=float)
    return np.clip(score + (100.0 - score) * weight * np.clip(risk, 0.0, 1.0), 0.0, 100.0)
