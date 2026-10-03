"""Entity resolution and entity-level feature extraction.

Addresses are grouped into *entities* (probable single owners) with the
common-input-ownership heuristic: addresses spent together as inputs of one
transaction share a wallet.  CoinJoin transactions are excluded because their
inputs belong to different parties.  Each entity then gets a feature vector
covering structure, timing, pass-through behaviour, heuristic participation
and network obfuscation.  These features feed the classifier, the anomaly
model and the explanations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from backend.forensics.dataset import NS_PER_S, RawDataset
from backend.forensics.geo import flag_hosting_vpn, flag_tor
from backend.forensics.heuristics import TxHeuristics

logger = logging.getLogger(__name__)


@dataclass
class EntityTable:
    """Entities (address clusters) with features and the value-flow graph."""

    addr_cluster: np.ndarray  # address id -> entity id
    features: pd.DataFrame  # index = entity id
    flows: pd.DataFrame  # columns: src, dst, amount, ts, tx_idx  (entity ids)
    tx_sender: np.ndarray  # tx idx -> sender entity id (-1 coinbase / CoinJoin)
    n_entities: int

    def labels(self, ds: RawDataset) -> pd.DataFrame | None:
        """Ground-truth labels per entity (synthetic data only)."""
        if ds.truth is None:
            return None
        ent = ds.truth.entities
        true_illicit = ent["is_illicit"].to_numpy()
        scen = ent["scenario"].to_numpy()
        a_true = ds.truth.addr_entity
        df = pd.DataFrame(
            {
                "cluster": self.addr_cluster,
                "illicit": true_illicit[a_true].astype(float),
                "scenario": scen[a_true],
            }
        )
        share = df.groupby("cluster")["illicit"].mean()
        cnt = df[df["illicit"] > 0].groupby(["cluster", "scenario"]).size().reset_index(name="n")
        illicit_scn = (
            cnt.sort_values("n", ascending=False)
            .drop_duplicates("cluster")
            .set_index("cluster")["scenario"]
        )
        out = pd.DataFrame({"illicit_share": share})
        out["is_illicit"] = out["illicit_share"] >= 0.5
        out["scenario"] = illicit_scn.reindex(out.index).fillna("normal")
        out.loc[~out["is_illicit"], "scenario"] = "normal"
        return out.reindex(range(self.n_entities))


# ======================================================================
# Entity resolution
# ======================================================================


def cluster_addresses(
    ds: RawDataset, h: TxHeuristics, use_change_links: bool = True
) -> np.ndarray:
    """Common-input-ownership clustering -> dense entity id per address.

    With ``use_change_links`` a detected change output is merged into the
    sender's entity when the change heuristic is confident enough.
    """
    inputs = ds.inputs
    n_a = ds.n_addresses
    n_in = ds.tx["n_in"].to_numpy()
    use = (n_in[inputs["tx_idx"].to_numpy()] >= 2) & ~h.is_coinjoin[inputs["tx_idx"].to_numpy()]
    sub = inputs[use]
    first = sub.groupby("tx_idx")["addr"].transform("first").to_numpy()
    a = sub["addr"].to_numpy()
    if use_change_links:
        conf_ok = (
            (h.change_addr >= 0)
            & (h.change_confidence >= h.config.change_link_confidence)
            & ~h.is_coinjoin
        )
        t = np.flatnonzero(conf_ok)
        first_in = inputs.drop_duplicates("tx_idx").set_index("tx_idx")["addr"].reindex(t)
        keep = first_in.notna().to_numpy()
        a = np.concatenate([a, h.change_addr[t[keep]]])
        first = np.concatenate([first, first_in.to_numpy()[keep].astype(np.int64)])
    g = coo_matrix((np.ones(len(a), dtype=np.int8), (a, first)), shape=(n_a, n_a))
    _, labels = connected_components(g, directed=False)
    return labels.astype(np.int64)


def clustering_quality(ds: RawDataset, addr_cluster: np.ndarray) -> dict:
    """Purity / fragmentation of the clustering vs planted ownership."""
    if ds.truth is None:
        return {}
    t = ds.truth.addr_entity
    df = pd.DataFrame({"c": addr_cluster, "t": t})
    majority = df.groupby(["c", "t"]).size().groupby(level=0).max()
    sizes = df.groupby("c").size()
    purity = float(majority.sum() / sizes.sum())
    per_true = df.groupby("t")["c"].nunique()
    multi = df.groupby("t").size() >= 2
    return {
        "n_clusters": int(df["c"].nunique()),
        "n_true_entities": int(df["t"].nunique()),
        "address_weighted_purity": purity,
        "mean_clusters_per_multi_address_entity": float(per_true[multi].mean())
        if multi.any()
        else None,
        "share_true_entities_fully_merged": float((per_true[multi] == 1).mean())
        if multi.any()
        else None,
    }


# ======================================================================
# Features
# ======================================================================


def _wilson(k: np.ndarray, n: np.ndarray, z: float = 1.96) -> tuple[np.ndarray, np.ndarray]:
    n = np.maximum(n, 1)
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return np.clip(centre - half, 0, 1), np.clip(centre + half, 0, 1)


def first_observation(ds: RawDataset) -> pd.DataFrame:
    """Earliest observation per tx: origin IP, country, ASN, Tor and VPN flags."""
    o = ds.obs.sort_values("ts", kind="stable").drop_duplicates("tx_idx", keep="first")
    o = o.set_index("tx_idx").reindex(np.arange(ds.n_tx))
    src = o["src_ip"].fillna("").to_numpy()
    asn = o["asn"].fillna(0).to_numpy().astype(np.int64)
    return pd.DataFrame(
        {
            "ip": src,
            "country": o["country"].fillna("").to_numpy(),
            "asn": asn,
            "is_tor": flag_tor(src),
            "is_vpn": flag_hosting_vpn(asn),
            "obs_ts": o["ts"].fillna(0).to_numpy().astype(np.int64),
        }
    )


def build_entities(
    ds: RawDataset, h: TxHeuristics, hours_bin: int = 1, use_change_links: bool = True
) -> EntityTable:
    """Resolve entities and compute the feature table plus flow graph."""
    n = ds.n_tx
    tx = ds.tx
    ts = tx["ts"].to_numpy()
    n_in = tx["n_in"].to_numpy()
    n_out = tx["n_out"].to_numpy()
    cl = cluster_addresses(ds, h, use_change_links)
    n_ent = int(cl.max()) + 1 if len(cl) else 0

    inputs, outputs = ds.inputs, ds.outputs
    in_tx = inputs["tx_idx"].to_numpy()
    in_c = cl[inputs["addr"].to_numpy()]
    in_amt = inputs["amount"].to_numpy()
    out_tx = outputs["tx_idx"].to_numpy()
    out_c = cl[outputs["addr"].to_numpy()]
    out_amt = outputs["amount"].to_numpy()

    # sender entity per tx (first input's entity; -1 for coinbase and CoinJoin)
    tx_sender = np.full(n, -1, dtype=np.int64)
    first_in = inputs.drop_duplicates("tx_idx")
    tx_sender[first_in["tx_idx"].to_numpy()] = cl[first_in["addr"].to_numpy()]
    tx_sender[h.is_coinjoin] = -1

    # ---- sent side (per entity, unique txs) --------------------------
    sent_pairs = pd.DataFrame({"c": in_c, "t": in_tx, "a": in_amt})
    sent_tx = sent_pairs.groupby(["c", "t"], sort=False)["a"].sum().reset_index()
    sent_tx["ts"] = ts[sent_tx["t"].to_numpy()]
    sent_tx["n_out"] = n_out[sent_tx["t"].to_numpy()]
    sent_tx["n_in"] = n_in[sent_tx["t"].to_numpy()]

    # ---- received side (exclude change back to the sender) -----------
    key_in = np.unique(in_tx.astype(np.int64) * n_ent + in_c)
    key_out = out_tx.astype(np.int64) * n_ent + out_c
    external = ~np.isin(key_out, key_in)
    recv = pd.DataFrame({"c": out_c[external], "t": out_tx[external], "a": out_amt[external]})
    recv_tx = recv.groupby(["c", "t"], sort=False)["a"].sum().reset_index()
    recv_tx["ts"] = ts[recv_tx["t"].to_numpy()]

    # ---- flow graph -----------------------------------------------------
    snd = tx_sender[out_tx]
    m = external & (snd >= 0) & (snd != out_c)
    flows = pd.DataFrame(
        {
            "src": snd[m],
            "dst": out_c[m],
            "amount": out_amt[m],
            "ts": ts[out_tx[m]],
            "tx_idx": out_tx[m],
        }
    )
    # CoinJoin: attribute each output to every participating input entity, amount split.
    cj_rows = h.is_coinjoin[in_tx]
    if cj_rows.any():
        cj_in = pd.DataFrame({"t": in_tx[cj_rows], "src": in_c[cj_rows]}).drop_duplicates()
        cj_out_mask = h.is_coinjoin[out_tx]
        cj_out = pd.DataFrame(
            {"t": out_tx[cj_out_mask], "dst": out_c[cj_out_mask], "amount": out_amt[cj_out_mask]}
        )
        merged = cj_in.merge(cj_out, on="t")
        merged = merged[merged["src"] != merged["dst"]]
        if len(merged):
            cnt = merged.groupby("t")["src"].transform("nunique")
            merged["amount"] = merged["amount"] / cnt
            flows = pd.concat(
                [
                    flows,
                    pd.DataFrame(
                        {
                            "src": merged["src"],
                            "dst": merged["dst"],
                            "amount": merged["amount"],
                            "ts": ts[merged["t"].to_numpy()],
                            "tx_idx": merged["t"],
                        }
                    ),
                ],
                ignore_index=True,
            )

    F = pd.DataFrame(index=np.arange(n_ent))
    F["n_addresses"] = np.bincount(cl, minlength=n_ent)

    # volumes / counts
    g = sent_tx.groupby("c")
    F["n_sent_txs"] = g.size().reindex(F.index, fill_value=0)
    F["total_sent_btc"] = g["a"].sum().reindex(F.index, fill_value=0.0)
    F["mean_sent_btc"] = g["a"].mean().reindex(F.index, fill_value=0.0)
    F["std_sent_btc"] = g["a"].std().reindex(F.index, fill_value=0.0).fillna(0.0)
    F["mean_n_out_sent"] = g["n_out"].mean().reindex(F.index, fill_value=0.0)
    F["max_n_out_sent"] = g["n_out"].max().reindex(F.index, fill_value=0)
    F["mean_n_in_sent"] = g["n_in"].mean().reindex(F.index, fill_value=0.0)
    F["max_n_in_sent"] = g["n_in"].max().reindex(F.index, fill_value=0)
    gr = recv_tx.groupby("c")
    F["n_recv_txs"] = gr.size().reindex(F.index, fill_value=0)
    F["total_recv_btc"] = gr["a"].sum().reindex(F.index, fill_value=0.0)
    F["mean_recv_btc"] = gr["a"].mean().reindex(F.index, fill_value=0.0)
    F["n_events"] = F["n_sent_txs"] + F["n_recv_txs"]
    hi, lo = (
        np.maximum(F["total_recv_btc"], F["total_sent_btc"]),
        np.minimum(F["total_recv_btc"], F["total_sent_btc"]),
    )
    F["pass_through_ratio"] = np.where(hi > 0, lo / np.where(hi > 0, hi, 1), 0.0)
    F["recv_to_sent_txs"] = F["n_recv_txs"] / (F["n_sent_txs"] + 1)

    # degrees in the entity flow graph
    if len(flows):
        pairs = flows[["src", "dst"]].drop_duplicates()
        F["out_degree"] = pairs.groupby("src").size().reindex(F.index, fill_value=0)
        F["in_degree"] = pairs.groupby("dst").size().reindex(F.index, fill_value=0)
        fl = flows.groupby("src")["amount"].agg(["sum", "max"])
        F["flow_out_btc"] = fl["sum"].reindex(F.index, fill_value=0.0)
        F["max_single_out_btc"] = fl["max"].reindex(F.index, fill_value=0.0)
    else:
        for c in ("out_degree", "in_degree", "flow_out_btc", "max_single_out_btc"):
            F[c] = 0

    # amount structure: share of round payments, dust received
    rp = pd.DataFrame({"c": sent_tx["c"], "r": h.is_round_payment[sent_tx["t"].to_numpy()]})
    F["round_payment_share"] = rp.groupby("c")["r"].mean().reindex(F.index, fill_value=0.0)
    dust_out = outputs["amount"].to_numpy() * 1e8 <= h.config.dust_sat
    F["dust_received"] = np.bincount(out_c[dust_out], minlength=n_ent)

    # ---- temporal --------------------------------------------------------
    ev = (
        pd.concat([sent_tx[["c", "t", "ts"]], recv_tx[["c", "t", "ts"]]], ignore_index=True)
        .drop_duplicates(["c", "t"])
        .sort_values(["c", "ts"], kind="stable")
    )
    c_arr = ev["c"].to_numpy()
    t_arr = ev["ts"].to_numpy()
    first_ts = np.full(n_ent, -1, dtype=np.int64)
    last_ts = np.full(n_ent, -1, dtype=np.int64)
    if len(ev):
        first_ts[np.unique(c_arr)] = t_arr[np.r_[True, c_arr[1:] != c_arr[:-1]]]
        last_ts[np.unique(c_arr)] = t_arr[np.r_[c_arr[1:] != c_arr[:-1], True]]
    F["active_hours"] = np.where(first_ts >= 0, (last_ts - first_ts) / NS_PER_S / 3600.0, 0.0)
    dt = np.diff(t_arr) / NS_PER_S
    same = c_arr[1:] == c_arr[:-1]
    gaps = pd.DataFrame({"c": c_arr[1:][same], "dt": dt[same]})
    gg = gaps.groupby("c")["dt"]
    mu, sd, cnt = gg.mean(), gg.std().fillna(0.0), gg.size()
    burst = ((sd - mu) / (sd + mu + 1e-9)).where(cnt >= 3, 0.0)
    F["burstiness"] = burst.reindex(F.index, fill_value=0.0)
    F["median_gap_s"] = gg.median().reindex(F.index, fill_value=0.0)
    bucket = (t_arr // (NS_PER_S * 3600 * hours_bin)).astype(np.int64)
    per_hour = pd.DataFrame({"c": c_arr, "b": bucket}).groupby(["c", "b"]).size()
    F["peak_hour_events"] = per_hour.groupby(level=0).max().reindex(F.index, fill_value=0)
    F["peak_hour_share"] = F["peak_hour_events"] / np.maximum(F["n_events"], 1)

    # dwell: how long funds sit at an address before being spent
    dwell = (ts[in_tx] - h.first_recv_ts[inputs["addr"].to_numpy()]) / NS_PER_S
    dw = pd.DataFrame(
        {"c": in_c, "d": np.where(h.first_recv_ts[inputs["addr"].to_numpy()] >= 0, dwell, np.nan)}
    )
    dwg = dw.groupby("c")["d"]
    F["median_dwell_s"] = dwg.median().reindex(F.index).fillna(-1.0)
    F["rapid_spend_share"] = (
        (dw["d"] < h.config.rapid_hop_seconds)
        .groupby(dw["c"])
        .mean()
        .reindex(F.index, fill_value=0.0)
    )

    # ---- heuristic participation (as sender / participant) -------------
    def count_flag(flag: np.ndarray) -> pd.Series:
        s = sent_tx[flag[sent_tx["t"].to_numpy()]]
        return s.groupby("c").size().reindex(F.index, fill_value=0)

    F["coinjoin_txs"] = count_flag(h.is_coinjoin)
    F["dust_spray_txs"] = count_flag(h.is_dust_spray)
    F["peel_chain_txs"] = count_flag(h.peel_chain_id >= 0)
    F["layering_chain_txs"] = count_flag(h.layering_chain_id >= 0)
    F["consolidation_txs"] = count_flag(h.is_consolidation)
    F["batch_payout_txs"] = count_flag(h.is_batch_payout)
    F["max_peel_chain_len"] = (
        pd.DataFrame({"c": sent_tx["c"], "l": h.peel_length[sent_tx["t"].to_numpy()]})
        .groupby("c")["l"]
        .max()
        .reindex(F.index, fill_value=0)
    )
    F["max_layering_chain_len"] = (
        pd.DataFrame({"c": sent_tx["c"], "l": h.layering_length[sent_tx["t"].to_numpy()]})
        .groupby("c")["l"]
        .max()
        .reindex(F.index, fill_value=0)
    )
    # receives a burst of similar amounts (ransom-like collection)
    if len(recv_tx):
        rg = recv_tx.groupby("c")["a"]
        cv = (rg.std().fillna(0.0) / (rg.mean() + 1e-12)).where(rg.size() >= 8, np.nan)
        F["recv_amount_cv"] = cv.reindex(F.index).fillna(-1.0)
    else:
        F["recv_amount_cv"] = -1.0

    # ---- network layer -----------------------------------------------------
    fo = first_observation(ds)
    snt = sent_tx[["c", "t", "ts"]].copy()
    snt = snt[~h.is_coinjoin[snt["t"].to_numpy()]]
    t_idx = snt["t"].to_numpy()
    snt["ip"] = fo["ip"].to_numpy()[t_idx]
    snt["country"] = fo["country"].to_numpy()[t_idx]
    snt["tor"] = fo["is_tor"].to_numpy()[t_idx]
    snt["vpn"] = fo["is_vpn"].to_numpy()[t_idx]
    sg = snt.groupby("c")
    F["tor_share"] = sg["tor"].mean().reindex(F.index, fill_value=0.0)
    F["vpn_share"] = sg["vpn"].mean().reindex(F.index, fill_value=0.0)
    F["n_origin_ips"] = sg["ip"].nunique().reindex(F.index, fill_value=0)
    F["n_origin_countries"] = sg["country"].nunique().reindex(F.index, fill_value=0)
    ip_counts = snt.groupby(["c", "ip"]).size()
    top = ip_counts.groupby(level=0).max()
    n_sent_nocj = sg.size()
    k_top = top.reindex(F.index, fill_value=0).to_numpy()
    n_tot = n_sent_nocj.reindex(F.index, fill_value=0).to_numpy()
    F["top_ip_share"] = np.where(n_tot > 0, k_top / np.maximum(n_tot, 1), 0.0)
    lo_ci, hi_ci = _wilson(k_top.astype(float), n_tot.astype(float))
    F["wallet_ip_link_lo"] = np.where(n_tot > 0, lo_ci, 0.0)
    F["wallet_ip_link_hi"] = np.where(n_tot > 0, hi_ci, 0.0)
    # rapid geo-hops: consecutive sends from different countries within 10 minutes
    snt = snt.sort_values(["c", "ts"], kind="stable")
    cc = snt["c"].to_numpy()
    same_c = cc[1:] == cc[:-1]
    diff_country = snt["country"].to_numpy()[1:] != snt["country"].to_numpy()[:-1]
    quick = (np.diff(snt["ts"].to_numpy()) / NS_PER_S) < 600
    hops = (
        pd.Series((same_c & diff_country & quick).astype(int), index=cc[1:]).groupby(level=0).sum()
    )
    F["rapid_geo_hops"] = hops.reindex(F.index, fill_value=0)
    F["network_obfuscation"] = np.clip(
        0.6 * F["tor_share"]
        + 0.4 * F["vpn_share"]
        + 0.15 * np.minimum(F["rapid_geo_hops"], 3) / 3,
        0,
        1,
    )

    F = F.astype({c: "float64" for c in F.columns})
    logger.info("Entities: %d clusters, %d features", n_ent, F.shape[1])
    return EntityTable(
        addr_cluster=cl, features=F, flows=flows, tx_sender=tx_sender, n_entities=n_ent
    )


FEATURE_DESCRIPTIONS: dict[str, str] = {
    "n_addresses": "number of addresses controlled",
    "n_sent_txs": "transactions sent",
    "total_sent_btc": "total BTC sent",
    "mean_n_out_sent": "average outputs per transaction it sends",
    "max_n_out_sent": "largest number of outputs in one transaction it sent",
    "mean_n_in_sent": "average inputs per transaction it sends",
    "max_n_in_sent": "largest number of inputs in one transaction it sent",
    "n_recv_txs": "transactions received",
    "total_recv_btc": "total BTC received",
    "pass_through_ratio": "share of received value that is forwarded on",
    "out_degree": "distinct entities it pays",
    "in_degree": "distinct entities that pay it",
    "round_payment_share": "share of payments that are round amounts",
    "dust_received": "dust outputs received",
    "active_hours": "hours between first and last activity",
    "burstiness": "burstiness of activity (1 = highly bursty)",
    "peak_hour_share": "share of all activity packed into its busiest hour",
    "median_dwell_s": "median seconds funds sit before being spent",
    "rapid_spend_share": "share of coins spent within 30 minutes of arrival",
    "coinjoin_txs": "CoinJoin transactions joined",
    "dust_spray_txs": "dust-spray transactions sent",
    "peel_chain_txs": "transactions inside peel chains",
    "layering_chain_txs": "transactions inside rapid-hop layering chains",
    "consolidation_txs": "many-input consolidation transactions",
    "batch_payout_txs": "many-output batch payouts",
    "max_peel_chain_len": "longest peel chain it took part in",
    "max_layering_chain_len": "longest rapid-hop chain it took part in",
    "recv_amount_cv": "spread of received amounts (low = suspiciously uniform)",
    "tor_share": "share of its transactions first seen via Tor exits",
    "vpn_share": "share of its transactions first seen via hosting/VPN networks",
    "n_origin_ips": "distinct origin IPs",
    "n_origin_countries": "distinct origin countries",
    "top_ip_share": "share of transactions from its most common origin IP",
    "rapid_geo_hops": "rapid country changes between consecutive sends",
    "network_obfuscation": "composite network obfuscation score",
    "median_gap_s": "median seconds between its transactions",
}
