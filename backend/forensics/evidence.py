"""Evidence assembly and plain-language summaries for one entity.

Every statement is tagged with its provenance, following ChainTrace's rule
that observations, heuristic inferences and model predictions are never
presented as one another:

* ``observation``      a fact readable directly from the data
* ``inference``        a published heuristic applied to observations
* ``model_prediction`` output of the trained model
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.forensics.dataset import NS_PER_S, RawDataset

SAT = 100_000_000


def iso(ns: int) -> str:
    return pd.Timestamp(int(ns), tz="UTC").strftime("%Y-%m-%d %H:%M:%SZ")


def humanize_seconds(s: float) -> str:
    if s < 0:
        return "n/a"
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.0f} min"
    if s < 172800:
        return f"{s / 3600:.1f} h"
    return f"{s / 86400:.1f} days"


def entity_addresses(res, eid: int, limit: int = 10) -> tuple[list[str], int]:
    idx = np.flatnonzero(res.et.addr_cluster == eid)
    return [str(res.ds.addresses[i]) for i in idx[:limit]], int(len(idx))


def entity_tx_index(res, eid: int) -> tuple[np.ndarray, np.ndarray]:
    """(sent tx indices, received tx indices) for an entity, sorted by time."""
    ds, cl = res.ds, res.et.addr_cluster
    in_c = cl[ds.inputs["addr"].to_numpy()]
    sent = np.unique(ds.inputs["tx_idx"].to_numpy()[in_c == eid])
    out_c = cl[ds.outputs["addr"].to_numpy()]
    recv = np.unique(ds.outputs["tx_idx"].to_numpy()[out_c == eid])
    return sent, np.setdiff1d(recv, sent)


def _txids(ds: RawDataset, idx, n: int = 5) -> list[str]:
    return [str(ds.tx["txid"].to_numpy()[i]) for i in list(idx)[:n]]


def heuristic_hits(res, eid: int, sent: np.ndarray) -> list[dict]:
    """Structural patterns this entity took part in, with supporting txids."""
    ds, h = res.ds, res.h
    hits: list[dict] = []
    ts = ds.tx["ts"].to_numpy()
    n_out = ds.tx["n_out"].to_numpy()

    def chain_hit(kind: str, ids: np.ndarray, lens: np.ndarray, chains: dict, label: str):
        mine = [t for t in sent if ids[t] >= 0]
        if not mine:
            return
        cid = int(ids[mine[0]])
        path = chains[cid]
        gaps = np.diff(ts[path]) / NS_PER_S
        detail = (
            f"Took part in a {len(path)}-transaction {label}; median {humanize_seconds(float(np.median(gaps)))} "
            f"between hops."
        )
        if kind == "peel":
            inp = ds.inputs.groupby("tx_idx")["amount"].sum()
            amounts = inp.reindex(path).to_numpy()
            detail += f" Value carried fell from {amounts[0]:.4f} to {amounts[-1]:.4f} BTC."
        hits.append(
            {
                "code": f"{kind}_chain",
                "label": label,
                "type": "inference",
                "detail": detail,
                "chain_length": len(path),
                "txids": _txids(ds, path),
            }
        )

    chain_hit("peel", h.peel_chain_id, h.peel_length, h.chains["peel"], "peel chain")
    chain_hit(
        "layering",
        h.layering_chain_id,
        h.layering_length,
        h.chains["layering"],
        "rapid-hop layering chain",
    )

    cj = [t for t in sent if h.is_coinjoin[t]]
    if cj:
        hits.append(
            {
                "code": "coinjoin",
                "label": "CoinJoin participation",
                "type": "inference",
                "detail": f"Joined {len(cj)} CoinJoin transaction(s) with equal-valued outputs "
                f"(up to {int(h.coinjoin_equal_outputs[cj].max())} identical outputs).",
                "txids": _txids(ds, cj),
            }
        )
    dust = [t for t in sent if h.is_dust_spray[t]]
    if dust:
        hits.append(
            {
                "code": "dust_spray",
                "label": "dust spray",
                "type": "observation",
                "detail": f"Sent {int(h.dust_outputs[dust].sum())} dust outputs (<= {h.config.dust_sat} sat) "
                f"across {len(dust)} transaction(s).",
                "txids": _txids(ds, dust),
            }
        )
    cons = [t for t in sent if h.is_consolidation[t]]
    if cons:
        hits.append(
            {
                "code": "consolidation",
                "label": "consolidation",
                "type": "observation",
                "detail": f"Merged many inputs into one output in {len(cons)} transaction(s).",
                "txids": _txids(ds, cons),
            }
        )
    batch = [t for t in sent if h.is_batch_payout[t]]
    if batch:
        hits.append(
            {
                "code": "batch_payout",
                "label": "batch payout",
                "type": "observation",
                "detail": f"Paid {int(n_out[batch].max())} outputs at once in {len(batch)} transaction(s).",
                "txids": _txids(ds, batch),
            }
        )
    return hits


def network_profile(res, eid: int, sent: np.ndarray) -> dict:
    """Network-layer evidence: where the entity's transactions first appeared."""
    ds = res.ds
    if len(sent) == 0:
        return {"sent_txs": 0}
    fo = res.first_obs.iloc[sent]
    df = pd.DataFrame(
        {
            "ip": fo["ip"].to_numpy(),
            "country": fo["country"].to_numpy(),
            "asn": fo["asn"].to_numpy(),
            "tor": fo["is_tor"].to_numpy(),
            "vpn": fo["is_vpn"].to_numpy(),
            "ts": ds.tx["ts"].to_numpy()[sent],
        }
    )
    g = (
        df.groupby("ip")
        .agg(
            n=("ip", "size"),
            country=("country", "first"),
            asn=("asn", "first"),
            tor=("tor", "first"),
            vpn=("vpn", "first"),
        )
        .sort_values("n", ascending=False)
    )
    n = len(df)
    top = g.iloc[0]
    from backend.forensics.entities import _wilson

    lo, hi = _wilson(np.array([float(top["n"])]), np.array([float(n)]))
    order = df.sort_values("ts")
    hops = (
        int(
            (
                (order["country"].to_numpy()[1:] != order["country"].to_numpy()[:-1])
                & (np.diff(order["ts"].to_numpy()) / NS_PER_S < 600)
            ).sum()
        )
        if n > 1
        else 0
    )
    return {
        "sent_txs": n,
        "top_origins": [
            {
                "ip": ip,
                "count": int(r["n"]),
                "country": r["country"],
                "asn": int(r["asn"]),
                "is_tor_exit": bool(r["tor"]),
                "is_hosting_or_vpn": bool(r["vpn"]),
            }
            for ip, r in g.head(5).iterrows()
        ],
        "tor_share": float(df["tor"].mean()),
        "vpn_share": float(df["vpn"].mean()),
        "distinct_ips": int(g.shape[0]),
        "distinct_countries": int(df["country"].nunique()),
        "rapid_country_changes": hops,
        "wallet_ip_link": {
            "ip": str(g.index[0]),
            "share": float(top["n"] / n),
            "ci95": [float(lo[0]), float(hi[0])],
            "interpretation": (
                "Correlation, not attribution: a stable origin IP suggests a fixed host, but "
                "NAT, VPNs and shared infrastructure mean an IP is never proof of identity."
            ),
        },
    }


def key_transactions(
    res, eid: int, sent: np.ndarray, recv: np.ndarray, limit: int = 12
) -> list[dict]:
    ds = res.ds
    tx = ds.tx
    in_amt = ds.inputs[ds.inputs["tx_idx"].isin(sent)].groupby("tx_idx")["amount"].sum()
    cl = res.et.addr_cluster
    o = ds.outputs[ds.outputs["tx_idx"].isin(recv)]
    rec_amt = o[cl[o["addr"].to_numpy()] == eid].groupby("tx_idx")["amount"].sum()
    rows = [(int(t), "sent", float(in_amt.get(t, 0.0))) for t in sent] + [
        (int(t), "received", float(rec_amt.get(t, 0.0))) for t in recv
    ]
    # most valuable, then keep chronological order for readability
    rows = sorted(rows, key=lambda r: -r[2])[:limit]
    rows = sorted(rows, key=lambda r: tx["ts"].to_numpy()[r[0]])
    return [
        {
            "txid": str(tx["txid"].to_numpy()[i]),
            "time": iso(tx["ts"].to_numpy()[i]),
            "role": role,
            "amount_btc": round(a, 8),
            "n_in": int(tx["n_in"].to_numpy()[i]),
            "n_out": int(tx["n_out"].to_numpy()[i]),
        }
        for i, role, a in rows
    ]


def build_entity_report(res, eid: int) -> dict:
    """Full evidence package for one entity (used by the API, report and summary)."""
    row = res.scores.loc[eid]
    feats = res.X.loc[[eid]]
    sent, recv = entity_tx_index(res, eid)
    addrs, n_addr = entity_addresses(res, eid)
    hits = heuristic_hits(res, eid, sent)
    net = network_profile(res, eid, sent)
    model = res.model_for(eid)
    p, lo, hi = model.predict_interval(feats)
    contrib = model.explain(feats)[0]
    f = res.et.features.loc[eid]
    ts_all = (
        res.ds.tx["ts"].to_numpy()[np.concatenate([sent, recv])]
        if len(sent) + len(recv)
        else np.array([0])
    )
    group = int(row["peer_group"])
    report = {
        "entity_id": int(eid),
        "label": f"E-{eid}",
        "addresses": addrs,
        "n_addresses": n_addr,
        "first_seen": iso(ts_all.min()),
        "last_seen": iso(ts_all.max()),
        "activity": {
            "sent_txs": int(f["n_sent_txs"]),
            "received_txs": int(f["n_recv_txs"]),
            "total_sent_btc": round(float(f["total_sent_btc"]), 6),
            "total_received_btc": round(float(f["total_recv_btc"]), 6),
            "pass_through_ratio": round(float(f["pass_through_ratio"]), 4),
            "median_hold_time": humanize_seconds(float(f["median_dwell_s"])),
            "counterparties_out": int(f["out_degree"]),
            "counterparties_in": int(f["in_degree"]),
        },
        "score": {
            "final": float(row["score"]),
            "tier": str(row["tier"]),
            "rank": int(row["rank"]),
            "of": int(res.n_ranked),
            "illicit_probability": float(p[0]),
            "interval90": [float(lo[0]), float(hi[0])],
            "anomaly_percentile": float(row["anomaly"]),
            "structural_heuristic_score": float(row["heuristic"]),
            "network_obfuscation": float(row["network"]),
            "temporal_burst": float(row["burst"]),
            "active_signals": str(row["active_signals"]).split(","),
        },
        "heuristic_evidence": hits,
        "network_evidence": net,
        "model_contributions": contrib,
        "peer_group": {
            "id": group,
            **(res.peer.descriptions.get(group, {}) if group >= 0 else {}),
        },
        "key_transactions": key_transactions(res, eid, sent, recv),
        "provenance": {
            "dataset": res.ds.report.source if res.ds.report else "",
            "synthetic_data": res.ds.truth is not None,
            "model_version": res.model_version,
            "label_source": res.label_source,
        },
        "disclaimer": (
            "Investigative triage lead. Observations are facts in the data; heuristics are inferences; "
            "probabilities are model output. None of these establishes guilt or identity."
        ),
    }
    if res.ds.truth is not None and res.labels is not None and eid in res.labels.index:
        report["ground_truth_synthetic"] = {
            "is_illicit": bool(res.labels.loc[eid, "is_illicit"]),
            "scenario": str(res.labels.loc[eid, "scenario"]),
        }
    report["summary"] = summarize(report)
    return report


def summarize(r: dict) -> str:
    """Template-based analyst summary (fully offline, deterministic)."""
    sc, act = r["score"], r["activity"]
    p = sc["illicit_probability"]
    lo, hi = sc["interval90"]
    parts = [
        f"{r['label']} ({r['n_addresses']} address{'es' if r['n_addresses'] != 1 else ''}) ranks "
        f"#{sc['rank']} of {sc['of']:,} with priority {sc['final']:.0f}/100 ({sc['tier'].upper()}). "
        f"Model illicit likelihood {p:.0%} (90% range {lo:.0%}-{hi:.0%})."
    ]
    reasons = []
    if act["pass_through_ratio"] >= 0.9 and act["sent_txs"] and act["received_txs"]:
        reasons.append(
            f"it forwarded {act['pass_through_ratio']:.0%} of the value it received, holding funds a median of "
            f"{act['median_hold_time']}"
        )
    for h in r["heuristic_evidence"][:3]:
        reasons.append(
            h["detail"]
            .rstrip(".")
            .replace("Took part in", "it took part in")
            .replace("Joined", "it joined")
            .replace("Sent", "it sent")
            .replace("Paid", "it paid")
            .replace("Merged", "it merged")
        )
    net = r["network_evidence"]
    if net.get("tor_share", 0) >= 0.25:
        reasons.append(
            f"{net['tor_share']:.0%} of its transactions first appeared from Tor exit nodes"
        )
    if net.get("vpn_share", 0) >= 0.25:
        reasons.append(f"{net['vpn_share']:.0%} first appeared from hosting/VPN networks")
    if net.get("rapid_country_changes", 0) >= 2:
        reasons.append(
            f"its origin jumped between countries {net['rapid_country_changes']} times within minutes"
        )
    if reasons:
        parts.append("Why it was flagged: " + "; ".join(reasons) + ".")
    top = [c for c in r["model_contributions"] if c["contribution"] > 0][:2]
    if top:
        parts.append(
            "The model weighted most: "
            + ", ".join(f"{c['description']} (+{c['contribution'] * 100:.0f} pts)" for c in top)
            + "."
        )
    parts.append("This is a triage lead, not proof of wrongdoing.")
    return " ".join(parts)
