"""Link-analysis data: neighbourhood subgraph, timed flow events, money trails.

``entity_subgraph`` returns nodes plus a list of *timed* flow events so the
UI can replay fund movement with a time slider.  ``money_trail`` follows the
largest onward (or inbound) payment in time order until it reaches a
service-like entity (an exchange-style hub) or runs out of road.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from backend.forensics.evidence import iso


def _edges(res) -> pd.DataFrame:
    """Aggregated directed edges, cached on the result object."""
    cache = res.__dict__.setdefault("_cache", {})
    if "edges" not in cache:
        f = res.et.flows
        cache["edges"] = (
            f.groupby(["src", "dst"], sort=False)
            .agg(
                amount=("amount", "sum"),
                n_tx=("amount", "size"),
                first_ts=("ts", "min"),
                last_ts=("ts", "max"),
            )
            .reset_index()
        )
    return cache["edges"]


def _flows_by_src(res) -> tuple[pd.DataFrame, np.ndarray]:
    cache = res.__dict__.setdefault("_cache", {})
    if "by_src" not in cache:
        f = res.et.flows.sort_values(["src", "ts"], kind="stable").reset_index(drop=True)
        cache["by_src"] = (f, f["src"].to_numpy())
    return cache["by_src"]


def _flows_by_dst(res) -> tuple[pd.DataFrame, np.ndarray]:
    cache = res.__dict__.setdefault("_cache", {})
    if "by_dst" not in cache:
        f = res.et.flows.sort_values(["dst", "ts"], kind="stable").reset_index(drop=True)
        cache["by_dst"] = (f, f["dst"].to_numpy())
    return cache["by_dst"]


def node_info(res, eid: int) -> dict:
    s = res.scores.loc[eid]
    f = res.et.features.loc[eid]
    info = {
        "id": int(eid),
        "label": f"E-{eid}",
        "score": float(s["score"]),
        "tier": str(s["tier"]),
        "p": float(s["p"]),
        "rank": int(s["rank"]),
        "is_service": bool(s["is_service"]),
        "n_addresses": int(f["n_addresses"]),
        "peer_group": int(s["peer_group"]),
        "is_seed": bool(s["is_seed"]),
        "seed_hops": int(s["seed_hops"]),
    }
    if res.labels is not None and eid in res.labels.index:
        info["truth_illicit"] = bool(res.labels.loc[eid, "is_illicit"])
        info["truth_scenario"] = str(res.labels.loc[eid, "scenario"])
    return info


def entity_subgraph(
    res, eid: int, radius: int = 2, max_nodes: int = 60, max_events: int = 800
) -> dict:
    """Neighbourhood of ``eid`` with timed flow events.

    Hubs (services) are kept as nodes but not expanded further, so one
    exchange does not pull the whole network into the picture.
    """
    edges = _edges(res)
    nodes = {int(eid)}
    frontier = {int(eid)}
    svc = res.scores["is_service"].to_numpy()
    for _ in range(radius):
        expand = np.array([n for n in frontier if n == eid or not svc[n]], dtype=np.int64)
        if len(expand) == 0:
            break
        m = edges["src"].isin(expand) | edges["dst"].isin(expand)
        sub = edges[m]
        other = np.where(sub["src"].isin(expand), sub["dst"], sub["src"])
        weight = pd.Series(sub["amount"].to_numpy(), index=other).groupby(level=0).sum()
        weight = weight.drop(index=list(nodes & set(weight.index)), errors="ignore").sort_values(
            ascending=False
        )
        room = max_nodes - len(nodes)
        new = set(int(x) for x in weight.index[: max(room, 0)])
        frontier = new
        nodes |= new
        if len(nodes) >= max_nodes:
            break

    trail_fwd = money_trail(res, eid, "forward")
    trail_back = money_trail(res, eid, "back")
    for step in trail_fwd + trail_back:  # always show the whole money trail
        nodes.add(int(step["entity"]))

    node_arr = np.array(sorted(nodes), dtype=np.int64)
    e = edges[edges["src"].isin(node_arr) & edges["dst"].isin(node_arr)]
    flows = res.et.flows
    fm = flows["src"].isin(node_arr) & flows["dst"].isin(node_arr)
    ev = flows[fm].sort_values("amount", ascending=False).head(max_events).sort_values("ts")
    txids = res.ds.tx["txid"].to_numpy()
    events = [
        {
            "src": int(s),
            "dst": int(d),
            "ts": int(t),
            "time": iso(t),
            "amount": round(float(a), 8),
            "txid": str(txids[x]),
        }
        for s, d, t, a, x in zip(
            ev["src"], ev["dst"], ev["ts"], ev["amount"], ev["tx_idx"], strict=True
        )
    ]
    return {
        "center": int(eid),
        "nodes": [node_info(res, int(n)) for n in node_arr],
        "edges": [
            {
                "src": int(r.src),
                "dst": int(r.dst),
                "amount": round(float(r.amount), 8),
                "n_tx": int(r.n_tx),
                "first_ts": int(r.first_ts),
                "last_ts": int(r.last_ts),
            }
            for r in e.itertuples(index=False)
        ],
        "events": events,
        "time_range": [int(ev["ts"].min()), int(ev["ts"].max())] if len(ev) else [0, 0],
        "trail_forward": trail_fwd,
        "trail_back": trail_back,
        "truncated": len(nodes) >= max_nodes,
    }


def money_trail(res, eid: int, direction: str = "forward", max_hops: int = 15) -> list[dict]:
    """Follow the largest payment onward (or backward) in time order."""
    fwd = direction == "forward"
    f, key = _flows_by_src(res) if fwd else _flows_by_dst(res)
    svc = res.scores["is_service"].to_numpy()
    path = [{"entity": int(eid), "label": f"E-{eid}", "ts": None, "time": None, "amount": None}]
    cur, t_cur = int(eid), (-(2**62) if fwd else 2**62)
    seen = {cur}
    for _ in range(max_hops):
        lo, hi = np.searchsorted(key, cur, "left"), np.searchsorted(key, cur, "right")
        seg = f.iloc[lo:hi]
        seg = seg[seg["ts"] > t_cur] if fwd else seg[seg["ts"] < t_cur]
        seg = seg[~seg["dst" if fwd else "src"].isin(seen)]
        if len(seg) == 0:
            break
        row = seg.loc[seg["amount"].idxmax()]
        nxt = int(row["dst" if fwd else "src"])
        t_cur = int(row["ts"])
        path.append(
            {
                "entity": nxt,
                "label": f"E-{nxt}",
                "ts": t_cur,
                "time": iso(t_cur),
                "amount": round(float(row["amount"]), 8),
                "is_service": bool(svc[nxt]),
                "txid": str(res.ds.tx["txid"].to_numpy()[int(row["tx_idx"])]),
            }
        )
        seen.add(nxt)
        cur = nxt
        if svc[nxt]:
            break
    return path if len(path) > 1 else []
