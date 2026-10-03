"""Graph machine learning on the entity value-flow graph (NumPy/SciPy only).

Three label-free graph signals, all sparse and scalable:

* **Node embeddings** - PPMI of the 1+2-hop transition matrix factorised with
  truncated SVD.  This is the closed form that DeepWalk/node2vec implicitly
  approximate, with no random-walk sampling and no GPU.
* **Neighbourhood propagation** - one and two rounds of mean-aggregation of
  label-free indicators (e.g. "touches a peel chain") over neighbours, the
  message-passing step of a GraphSAGE layer with fixed weights.
* **Peer groups** - MiniBatchKMeans over standardised features + embeddings,
  each group summarised by the features that deviate most from the population.

None of these use ground-truth labels, so computing them on the full graph
(including future test entities) is transductive but not label leakage.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.sparse.linalg import svds
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)

EMB_PREFIX = "emb_"
PROP_COLUMNS = (
    "nbr_heuristic_exposure",
    "nbr2_heuristic_exposure",
    "nbr_mean_pass_through",
    "nbr_tor_share",
)


def build_adjacency(flows: pd.DataFrame, n_ent: int, degree_cap: int = 50) -> sparse.csr_matrix:
    """Symmetric weighted adjacency; each node keeps its ``degree_cap`` heaviest edges.

    Capping hubs (exchanges) keeps the 2-hop product sparse and stops a single
    hub from dominating every embedding.
    """
    if len(flows) == 0:
        return sparse.csr_matrix((n_ent, n_ent))
    e = flows.groupby(["src", "dst"], sort=False)["amount"].sum().reset_index()
    e["w"] = 1.0 + np.log1p(e["amount"].to_numpy())
    e = e.sort_values(["src", "w"], ascending=[True, False], kind="stable")
    e = e[e.groupby("src").cumcount() < degree_cap]
    a = sparse.coo_matrix(
        (e["w"].to_numpy(), (e["src"].to_numpy(), e["dst"].to_numpy())), shape=(n_ent, n_ent)
    ).tocsr()
    a = a + a.T
    # cap again on the symmetrised matrix (in-hubs)
    a = _cap_rows(a, degree_cap * 2)
    a.setdiag(0)
    a.eliminate_zeros()
    return a.tocsr()


def _cap_rows(a: sparse.csr_matrix, cap: int) -> sparse.csr_matrix:
    counts = np.diff(a.indptr)
    if counts.max(initial=0) <= cap:
        return a
    coo = a.tocoo()
    df = pd.DataFrame({"r": coo.row, "c": coo.col, "v": coo.data}).sort_values(
        ["r", "v"], ascending=[True, False], kind="stable"
    )
    df = df[df.groupby("r").cumcount() < cap]
    return sparse.coo_matrix((df["v"], (df["r"], df["c"])), shape=a.shape).tocsr()


def _row_normalise(a: sparse.csr_matrix) -> sparse.csr_matrix:
    deg = np.asarray(a.sum(axis=1)).ravel()
    inv = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)
    return sparse.diags(inv) @ a


def node_embeddings(adj: sparse.csr_matrix, dim: int = 16, seed: int = 42) -> np.ndarray:
    """PPMI-SVD embeddings of the 1+2-hop random-walk co-occurrence matrix."""
    n = adj.shape[0]
    emb = np.zeros((n, dim))
    active = np.flatnonzero(np.diff(adj.indptr) > 0)
    if len(active) <= dim + 1:
        return emb
    sub = adj[active][:, active]
    p = _row_normalise(sub)
    m = (p + p @ p).tocoo()
    row_sum = np.asarray(m.tocsr().sum(axis=1)).ravel()
    col_sum = np.asarray(m.tocsr().sum(axis=0)).ravel()
    z = m.data.sum()
    pmi = np.log((m.data * z) / (row_sum[m.row] * col_sum[m.col] + 1e-12) + 1e-12)
    keep = pmi > 0
    ppmi = sparse.coo_matrix((pmi[keep], (m.row[keep], m.col[keep])), shape=m.shape).tocsr()
    k = min(dim, ppmi.shape[0] - 2)
    rng = np.random.default_rng(seed)
    u, s, _ = svds(ppmi, k=k, v0=rng.normal(size=ppmi.shape[0]))
    order = np.argsort(-s)
    vecs = u[:, order] * np.sqrt(s[order])
    emb[np.ix_(active, np.arange(k))] = vecs
    return emb


def propagation_features(
    adj: sparse.csr_matrix, features: pd.DataFrame, indicator: np.ndarray
) -> pd.DataFrame:
    """Mean-aggregate label-free neighbour signals over 1 and 2 hops."""
    p = _row_normalise(adj)
    ind = indicator.astype(float)
    n1 = p @ ind
    n2 = p @ n1
    out = pd.DataFrame(index=features.index)
    out["nbr_heuristic_exposure"] = n1
    out["nbr2_heuristic_exposure"] = n2
    out["nbr_mean_pass_through"] = p @ features["pass_through_ratio"].to_numpy()
    out["nbr_tor_share"] = p @ features["tor_share"].to_numpy()
    return out


def heuristic_indicator(features: pd.DataFrame) -> np.ndarray:
    """1.0 where an entity itself took part in a structural laundering pattern."""
    f = features
    return (
        ((f["peel_chain_txs"] > 0) | (f["layering_chain_txs"] > 0) | (f["dust_spray_txs"] > 0))
        .to_numpy()
        .astype(float)
    )


@dataclass
class GraphSignals:
    adjacency: sparse.csr_matrix
    embeddings: pd.DataFrame  # columns emb_0..emb_{d-1}
    propagation: pd.DataFrame


def compute_graph_signals(
    flows: pd.DataFrame, features: pd.DataFrame, dim: int = 16, seed: int = 42
) -> GraphSignals:
    n = len(features)
    adj = build_adjacency(flows, n)
    emb = node_embeddings(adj, dim=dim, seed=seed)
    emb_df = pd.DataFrame(
        emb, index=features.index, columns=[f"{EMB_PREFIX}{i}" for i in range(emb.shape[1])]
    )
    prop = propagation_features(adj, features, heuristic_indicator(features))
    logger.info("Graph signals: adjacency nnz=%d, emb dim=%d", adj.nnz, emb.shape[1])
    return GraphSignals(adj, emb_df, prop)


# ----------------------------------------------------------------------
# Unsupervised peer groups
# ----------------------------------------------------------------------


@dataclass
class PeerGroups:
    labels: np.ndarray  # per entity; -1 = too little activity to group
    descriptions: dict[int, dict]


def peer_groups(
    matrix: pd.DataFrame,
    active: np.ndarray,
    n_groups: int = 8,
    seed: int = 42,
    describe_top: int = 3,
) -> PeerGroups:
    """Cluster active entities into behavioural peer groups and describe them."""
    labels = np.full(len(matrix), -1, dtype=np.int64)
    idx = np.flatnonzero(active)
    if len(idx) < n_groups * 5:
        return PeerGroups(labels, {})
    X = matrix.iloc[idx].to_numpy(dtype=float)
    Xs = StandardScaler().fit_transform(
        np.clip(X, np.nanpercentile(X, 0.5, axis=0), np.nanpercentile(X, 99.5, axis=0))
    )
    km = MiniBatchKMeans(n_clusters=n_groups, random_state=seed, n_init=3, batch_size=2048)
    lab = km.fit_predict(Xs)
    labels[idx] = lab
    desc: dict[int, dict] = {}
    cols = list(matrix.columns)
    for g in range(n_groups):
        mask = lab == g
        centre = Xs[mask].mean(axis=0)
        order = np.argsort(-np.abs(centre))[:describe_top]
        desc[g] = {
            "size": int(mask.sum()),
            "distinguishing": [
                {
                    "feature": cols[i],
                    "z": float(centre[i]),
                    "direction": "high" if centre[i] > 0 else "low",
                }
                for i in order
            ],
        }
    return PeerGroups(labels, desc)
