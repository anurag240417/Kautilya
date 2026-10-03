"""Blockchain forensics heuristics, vectorised over the whole dataset.

Each heuristic is a published, explainable technique, not a learned model:

* **CoinJoin** - many equal-valued outputs from many independent inputs.
* **Dust spray** - many outputs at or below the dust limit.
* **Change-address detection** - in a 2-output tx, the output that is a
  never-seen address and not a round amount is probably change.
* **Peel chain** - a run of 2-output txs where each spends the previous
  change output and peels a smaller amount off.
* **Rapid hop (layering)** - single-input/single-output sweeps of coins that
  arrived minutes earlier, chained.
* **Consolidation / batch payout** - many-in-one-out and one-in-many-out.

Heuristics are *probabilistic evidence*, never proof: each exposes the
threshold it used so the report can quote it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from backend.forensics.dataset import NS_PER_S, RawDataset

logger = logging.getLogger(__name__)

SAT = 100_000_000


@dataclass(frozen=True)
class HeuristicConfig:
    coinjoin_min_equal: int = 3
    coinjoin_min_inputs: int = 3
    coinjoin_min_equal_fraction: float = 0.35
    dust_sat: int = 1_000
    dust_spray_min_outputs: int = 8
    round_unit_sat: int = 100_000  # 0.001 BTC
    change_min_confidence: float = 0.5
    change_value_ratio: float = 4.0
    peel_min_length: int = 4
    peel_max_inputs: int = 2
    peel_max_gap_seconds: float = (
        12 * 3600.0
    )  # hops of a real peel chain are hours apart, not days
    change_link_confidence: float = 0.6  # min confidence to merge change into sender's entity
    rapid_hop_seconds: float = 1_800.0
    layering_min_length: int = 3
    consolidation_min_inputs: int = 10
    batch_min_outputs: int = 10


@dataclass
class TxHeuristics:
    """Per-transaction (indexed like ``ds.tx``) and per-address results."""

    config: HeuristicConfig
    is_coinjoin: np.ndarray
    coinjoin_equal_outputs: np.ndarray
    is_dust_spray: np.ndarray
    dust_outputs: np.ndarray
    change_addr: np.ndarray  # address id of detected change, -1 if none
    change_confidence: np.ndarray
    is_round_payment: np.ndarray
    peel_chain_id: np.ndarray  # -1 if not in a qualifying chain
    peel_position: np.ndarray
    peel_length: np.ndarray
    layering_chain_id: np.ndarray
    layering_position: np.ndarray
    layering_length: np.ndarray
    is_consolidation: np.ndarray
    is_batch_payout: np.ndarray
    is_rapid_hop: np.ndarray
    first_recv_ts: np.ndarray  # per address, ns; -1 if never received
    spend_tx: np.ndarray  # per address, first spending tx idx; -1 if unspent
    chains: dict = field(default_factory=dict)  # chain_id -> ordered tx idx lists

    def counts(self) -> dict[str, int]:
        return {
            "coinjoin_txs": int(self.is_coinjoin.sum()),
            "dust_spray_txs": int(self.is_dust_spray.sum()),
            "change_detected_txs": int((self.change_addr >= 0).sum()),
            "peel_chain_txs": int((self.peel_chain_id >= 0).sum()),
            "peel_chains": int(len({c for c in self.peel_chain_id if c >= 0})),
            "layering_chain_txs": int((self.layering_chain_id >= 0).sum()),
            "layering_chains": int(len({c for c in self.layering_chain_id if c >= 0})),
            "consolidation_txs": int(self.is_consolidation.sum()),
            "batch_payout_txs": int(self.is_batch_payout.sum()),
        }


# ======================================================================
# Building blocks
# ======================================================================


def first_last_per_address(ds: RawDataset) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(first_recv_ts, first_seen_tx, spend_tx) per address id.

    ``ds.tx`` is time-sorted, so the minimum tx index is the earliest.
    """
    n_a = ds.n_addresses
    ts = ds.tx["ts"].to_numpy()
    out_tx = ds.outputs["tx_idx"].to_numpy()
    out_addr = ds.outputs["addr"].to_numpy()
    in_tx = ds.inputs["tx_idx"].to_numpy()
    in_addr = ds.inputs["addr"].to_numpy()

    first_recv_tx = np.full(n_a, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(first_recv_tx, out_addr, out_tx)
    first_spend_tx = np.full(n_a, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(first_spend_tx, in_addr, in_tx)

    first_seen = np.minimum(first_recv_tx, first_spend_tx)
    first_recv_ts = np.where(
        first_recv_tx < np.iinfo(np.int64).max, ts[np.minimum(first_recv_tx, len(ts) - 1)], -1
    )
    spend = np.where(first_spend_tx < np.iinfo(np.int64).max, first_spend_tx, -1)
    first_seen = np.where(first_seen < np.iinfo(np.int64).max, first_seen, -1)
    return first_recv_ts.astype(np.int64), first_seen.astype(np.int64), spend.astype(np.int64)


def follow_chains(
    next_idx: np.ndarray, valid: np.ndarray, min_length: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Group linked txs into chains.

    ``next_idx[i]`` is the tx that continues tx ``i`` (or -1).  Only txs with
    ``valid`` True participate.  Returns (chain_id, position, length, chains)
    for chains of at least ``min_length`` txs; others get -1.
    """
    n = len(next_idx)
    chain_id = np.full(n, -1, dtype=np.int64)
    position = np.full(n, -1, dtype=np.int32)
    length = np.zeros(n, dtype=np.int32)
    nxt = np.where(valid & (next_idx >= 0), next_idx, -1)
    nxt = np.where((nxt >= 0) & (nxt < n), nxt, -1)
    nxt = np.where((nxt >= 0) & valid[np.clip(nxt, 0, n - 1)], nxt, -1)

    prev = np.full(n, -1, dtype=np.int64)
    src = np.flatnonzero(nxt >= 0)
    prev[nxt[src]] = src  # if several txs point at one, the latest wins

    starts = np.flatnonzero(valid & (prev == -1) & (nxt >= 0))
    chains: dict[int, list[int]] = {}
    cid = 0
    for s in starts:
        path = [int(s)]
        cur = int(nxt[s])
        while cur >= 0 and len(path) <= n:
            path.append(cur)
            cur = int(nxt[cur])
        if len(path) >= min_length:
            for pos, t in enumerate(path):
                chain_id[t] = cid
                position[t] = pos
                length[t] = len(path)
            chains[cid] = path
            cid += 1
    return chain_id, position, length, chains


# ======================================================================
# Main entry point
# ======================================================================


def detect_heuristics(ds: RawDataset, cfg: HeuristicConfig | None = None) -> TxHeuristics:
    """Run every heuristic over ``ds`` and return per-tx / per-address flags."""
    cfg = cfg or HeuristicConfig()
    n = ds.n_tx
    tx = ds.tx
    n_in = tx["n_in"].to_numpy()
    n_out = tx["n_out"].to_numpy()
    ts = tx["ts"].to_numpy()
    outputs = ds.outputs
    out_tx = outputs["tx_idx"].to_numpy()
    out_sat = np.rint(outputs["amount"].to_numpy() * SAT).astype(np.int64)

    # ---- CoinJoin: equal-valued outputs ------------------------------
    eq = (
        pd.DataFrame({"t": out_tx, "v": out_sat})
        .groupby(["t", "v"], sort=False)
        .size()
        .groupby(level=0)
        .max()
        .reindex(np.arange(n), fill_value=0)
        .to_numpy()
    )
    distinct_inputs = (
        ds.inputs.groupby("tx_idx")["addr"]
        .nunique()
        .reindex(np.arange(n), fill_value=0)
        .to_numpy()
    )
    frac = np.divide(eq, np.maximum(n_out, 1))
    is_cj = (
        (eq >= cfg.coinjoin_min_equal)
        & (distinct_inputs >= cfg.coinjoin_min_inputs)
        & (frac >= cfg.coinjoin_min_equal_fraction)
    )

    # ---- dust spray --------------------------------------------------
    dust_mask = out_sat <= cfg.dust_sat
    dust_count = np.bincount(out_tx[dust_mask], minlength=n)
    is_dust = (dust_count >= cfg.dust_spray_min_outputs) & (
        dust_count >= 0.5 * np.maximum(n_out, 1)
    )

    # ---- address lifecycle ------------------------------------------
    first_recv_ts, first_seen_tx, spend_tx = first_last_per_address(ds)

    # ---- change-address detection (2-output, non-CoinJoin txs) -------
    change_addr = np.full(n, -1, dtype=np.int64)
    change_conf = np.zeros(n)
    round_pay = np.zeros(n, dtype=bool)
    cand = (n_out == 2) & (n_in >= 1) & ~is_cj
    cand_tx = np.flatnonzero(cand)
    if len(cand_tx):
        two = outputs[cand[out_tx]]
        a = two["addr"].to_numpy().reshape(-1, 2)
        v = np.rint(two["amount"].to_numpy() * SAT).astype(np.int64).reshape(-1, 2)
        t = two["tx_idx"].to_numpy().reshape(-1, 2)[:, 0]
        new = first_seen_tx[a] == t[:, None]
        rnd = (v % cfg.round_unit_sat) == 0
        # score each output as change
        score = np.zeros((len(t), 2))
        for k in (0, 1):
            o = 1 - k
            score[:, k] += 0.5 * (new[:, k] & ~new[:, o])
            score[:, k] += 0.3 * (~rnd[:, k] & rnd[:, o])
            score[:, k] += 0.15 * (new[:, k] & ~rnd[:, k] & new[:, o] & rnd[:, o])
            score[:, k] += 0.15 * (new[:, k] & ~rnd[:, k])
            score[:, k] += 0.45 * (
                v[:, k] >= cfg.change_value_ratio * v[:, o]
            )  # remainder is much larger
        best = np.argmax(score, axis=1)
        top = score[np.arange(len(t)), best]
        second = score[np.arange(len(t)), 1 - best]
        conf = np.clip(top - 0.5 * second, 0.0, 0.95)
        ok = conf >= cfg.change_min_confidence
        change_addr[t[ok]] = a[np.arange(len(t)), best][ok]
        change_conf[t[ok]] = conf[ok]
        round_pay[t] = rnd[np.arange(len(t)), 1 - best]

    # ---- peel chains --------------------------------------------------
    next_peel = np.full(n, -1, dtype=np.int64)
    has_change = change_addr >= 0
    nxt = np.where(has_change, spend_tx[np.where(has_change, change_addr, 0)], -1)
    nxt = np.where(nxt > np.arange(n), nxt, -1)
    gap_ok = (nxt >= 0) & (
        (ts[np.clip(nxt, 0, n - 1)] - ts) / NS_PER_S <= cfg.peel_max_gap_seconds
    )
    nxt = np.where(gap_ok, nxt, -1)
    next_peel = nxt
    peel_valid = has_change | (np.isin(np.arange(n), nxt[nxt >= 0]) & (n_out == 2))
    peel_valid &= (n_in <= cfg.peel_max_inputs) & ~is_cj
    peel_id, peel_pos, peel_len, peel_chains = follow_chains(
        next_peel, peel_valid, cfg.peel_min_length
    )

    # ---- rapid-hop / layering chains ---------------------------------
    inputs = ds.inputs
    in_tx = inputs["tx_idx"].to_numpy()
    dwell = (ts[in_tx] - first_recv_ts[inputs["addr"].to_numpy()]) / NS_PER_S
    # a tx is a rapid hop if ALL its inputs arrived within rapid_hop_seconds
    slow_in = np.bincount(
        in_tx, weights=(dwell > cfg.rapid_hop_seconds).astype(float), minlength=n
    )
    rapid = (n_out == 1) & (n_in >= 1) & (slow_in == 0) & ~is_cj
    out_addr_single = np.full(n, -1, dtype=np.int64)
    single = outputs[(n_out == 1)[out_tx]]
    out_addr_single[single["tx_idx"].to_numpy()] = single["addr"].to_numpy()
    nxt_l = np.where(
        out_addr_single >= 0, spend_tx[np.where(out_addr_single >= 0, out_addr_single, 0)], -1
    )
    nxt_l = np.where(nxt_l > np.arange(n), nxt_l, -1)
    lay_id, lay_pos, lay_len, lay_chains = follow_chains(nxt_l, rapid, cfg.layering_min_length)

    h = TxHeuristics(
        config=cfg,
        is_coinjoin=is_cj,
        coinjoin_equal_outputs=eq.astype(np.int32),
        is_dust_spray=is_dust,
        dust_outputs=dust_count.astype(np.int32),
        change_addr=change_addr,
        change_confidence=change_conf,
        is_round_payment=round_pay,
        peel_chain_id=peel_id,
        peel_position=peel_pos,
        peel_length=peel_len,
        layering_chain_id=lay_id,
        layering_position=lay_pos,
        layering_length=lay_len,
        is_consolidation=(n_in >= cfg.consolidation_min_inputs) & (n_out <= 2) & ~is_cj,
        is_batch_payout=(n_out >= cfg.batch_min_outputs) & ~is_cj & ~is_dust,
        is_rapid_hop=rapid,
        first_recv_ts=first_recv_ts,
        spend_tx=spend_tx,
        chains={"peel": peel_chains, "layering": lay_chains},
    )
    logger.info("Heuristics: %s", h.counts())
    return h


# ======================================================================
# Validation against planted ground truth (synthetic data only)
# ======================================================================


def evaluate_heuristics(ds: RawDataset, h: TxHeuristics) -> dict:
    """Precision/recall of each heuristic against the generator's ground truth.

    Chain heuristics are scored by *who sent* each flagged transaction (is the
    sender's true owner an illicit actor?), because scenario tags also cover
    victims' and buyers' payments.
    """
    if ds.truth is None:
        return {}
    scn = ds.truth.tx_scenario
    addr_ent = ds.truth.addr_entity
    is_ill = ds.truth.entities["is_illicit"].to_numpy()
    first_in = ds.inputs.drop_duplicates("tx_idx").set_index("tx_idx")["addr"]
    sender_addr = first_in.reindex(np.arange(ds.n_tx)).fillna(-1).to_numpy().astype(np.int64)
    sender_illicit = np.where(
        sender_addr >= 0, is_ill[addr_ent[np.maximum(sender_addr, 0)]], False
    )
    out: dict = {}

    def pr(pred: np.ndarray, actual: np.ndarray) -> dict:
        tp = int((pred & actual).sum())
        return {
            "flagged": int(pred.sum()),
            "true_positives": tp,
            "precision": tp / pred.sum() if pred.sum() else None,
            "recall": tp / actual.sum() if actual.sum() else None,
        }

    out["coinjoin"] = pr(h.is_coinjoin, scn == "coinjoin")
    out["dust_spray"] = pr(h.is_dust_spray, scn == "dust_attack")

    ent = ds.truth.entities
    ops = set(ent.loc[ent["role"] == "operator", "entity_id"])
    if h.chains["peel"]:
        sender_true = addr_ent[np.maximum(sender_addr, 0)]
        chain_ill = []
        hit_ops = set()
        for path in h.chains["peel"].values():
            ill = sender_illicit[path]
            chain_ill.append(float(ill.mean()))
            hit_ops |= ops & set(sender_true[path].tolist())
        out["peel_chains"] = {
            "chains_detected": len(chain_ill),
            "chains_mostly_illicit_sender": int(sum(c >= 0.5 for c in chain_ill)),
            "chain_precision": float(np.mean([c >= 0.5 for c in chain_ill])),
            "ransomware_operators_with_detected_chain": len(hit_ops),
            "ransomware_operators_total": len(ops),
            "operator_recall": len(hit_ops) / max(1, len(ops)),
        }
    mules = set(ent.loc[ent["role"].isin(["mule", "source"]), "entity_id"])
    if h.chains["layering"]:
        sender_true = addr_ent[np.maximum(sender_addr, 0)]
        ill_flags = [float(sender_illicit[p].mean()) for p in h.chains["layering"].values()]
        hit = set()
        for p in h.chains["layering"].values():
            hit |= mules & set(sender_true[p].tolist())
        out["layering_chains"] = {
            "chains_detected": len(ill_flags),
            "chain_precision": float(np.mean([c >= 0.5 for c in ill_flags])),
            "mule_or_source_recall": len(hit) / max(1, len(mules)),
        }

    det = np.flatnonzero((h.change_addr >= 0) & (sender_addr >= 0))
    if len(det):
        correct = addr_ent[sender_addr[det]] == addr_ent[h.change_addr[det]]
        out["change_detection"] = {
            "detected": int(len(det)),
            "correct": int(correct.sum()),
            "precision": float(correct.mean()),
            "coverage_of_2_output_txs": float(
                len(det) / max(1, int(((ds.tx["n_out"] == 2) & (ds.tx["n_in"] >= 1)).sum()))
            ),
        }
    return out
