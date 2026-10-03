"""Heuristics on small hand-built transaction graphs with known answers."""

import numpy as np

from backend.forensics.entities import build_entities, cluster_addresses
from backend.forensics.heuristics import detect_heuristics
from tests.forensics_helpers import make_ds


def test_coinjoin_detected_and_excluded_from_clustering():
    # five independent parties, each pays 0.1 to a fresh address
    ins = [(f"p{i}", 0.2) for i in range(5)]
    outs = [(f"o{i}", 0.1) for i in range(5)] + [(f"c{i}", 0.0999) for i in range(5)]
    ds = make_ds([{"t": 0, "ins": ins, "outs": outs}])
    h = detect_heuristics(ds)
    assert h.is_coinjoin[0]
    cl = cluster_addresses(ds, h)
    p_ids = [int(np.flatnonzero(ds.addresses == f"p{i}")[0]) for i in range(5)]
    assert len({cl[i] for i in p_ids}) == 5  # CoinJoin inputs are NOT merged


def test_common_input_ownership_merges_non_coinjoin():
    ds = make_ds([{"t": 0, "ins": [("a", 0.5), ("b", 0.5)], "outs": [("z", 0.999)]}])
    cl = cluster_addresses(ds, detect_heuristics(ds))
    ia, ib = (int(np.flatnonzero(ds.addresses == x)[0]) for x in "ab")
    assert cl[ia] == cl[ib]


def test_dust_spray_detected():
    outs = [(f"v{i}", 0.00000546) for i in range(20)]
    ds = make_ds([{"t": 0, "ins": [("atk", 0.01)], "outs": outs + [("chg", 0.0098)]}])
    h = detect_heuristics(ds)
    assert h.is_dust_spray[0] and h.dust_outputs[0] == 20


def test_normal_payment_not_flagged():
    ds = make_ds([{"t": 0, "ins": [("a", 1.0)], "outs": [("b", 0.25), ("a2", 0.7499)]}])
    h = detect_heuristics(ds)
    assert not h.is_coinjoin[0] and not h.is_dust_spray[0] and not h.is_consolidation[0]


def test_change_detection_prefers_new_nonround_output():
    # payment is round (0.1), change is a never-seen address with an odd amount
    ds = make_ds([{"t": 0, "ins": [("a", 1.0)], "outs": [("dest", 0.1), ("chg", 0.8999)]}])
    h = detect_heuristics(ds)
    chg = int(np.flatnonzero(ds.addresses == "chg")[0])
    assert h.change_addr[0] == chg and h.change_confidence[0] >= 0.5


def _peel_chain(hops: int, gap_s: float):
    txs, coin, amt = [], "start", 10.0
    txs.append({"t": 0, "ins": [("src", 10.5)], "outs": [("start", amt), ("srcchg", 0.4998)]})
    for i in range(hops):
        peel = round(amt * 0.05, 6)
        new_amt = round(amt - peel - 0.0001, 6)
        txs.append(
            {
                "t": (i + 1) * gap_s,
                "ins": [(coin, amt)],
                "outs": [(f"peel{i}", peel), (f"chg{i}", new_amt)],
            }
        )
        coin, amt = f"chg{i}", new_amt
    return txs


def test_peel_chain_detected_when_hops_are_close_in_time():
    ds = make_ds(_peel_chain(8, gap_s=1800))
    h = detect_heuristics(ds)
    assert (h.peel_chain_id >= 0).sum() >= 6
    assert max(h.peel_length) >= 6


def test_slow_chain_is_not_a_peel_chain():
    # identical structure but hops are days apart: ordinary wallet behaviour
    ds = make_ds(_peel_chain(8, gap_s=3 * 86400))
    assert (detect_heuristics(ds).peel_chain_id >= 0).sum() == 0


def test_layering_chain_detected():
    txs = [{"t": 0, "ins": [("src", 5.0)], "outs": [("m0", 4.9999)]}]
    for i in range(5):
        txs.append(
            {
                "t": 300 * (i + 1),
                "ins": [(f"m{i}", 4.9999 - i * 1e-4)],
                "outs": [(f"m{i + 1}", 4.9998 - i * 1e-4)],
            }
        )
    ds = make_ds(txs)
    h = detect_heuristics(ds)
    assert (h.layering_chain_id >= 0).sum() >= 4
    assert h.layering_length.max() >= 4


def test_consolidation_and_batch_payout():
    cons = {"t": 0, "ins": [(f"d{i}", 0.1) for i in range(12)], "outs": [("hot", 1.1999)]}
    batch = {
        "t": 10,
        "ins": [("hot", 1.1999)],
        "outs": [(f"w{i}", 0.05) for i in range(12)] + [("hot2", 0.5998)],
    }
    h = detect_heuristics(make_ds([cons, batch]))
    assert h.is_consolidation[0] and not h.is_consolidation[1]
    assert h.is_batch_payout[1]


def test_entity_features_pass_through_and_dwell():
    txs = [
        {"t": 0, "ins": [("src", 5.0)], "outs": [("mule", 4.9999)]},
        {"t": 60, "ins": [("mule", 4.9999)], "outs": [("out", 4.9998)]},
    ]
    ds = make_ds(txs)
    et = build_entities(ds, detect_heuristics(ds))
    mule_ent = int(et.addr_cluster[np.flatnonzero(ds.addresses == "mule")[0]])
    f = et.features.loc[mule_ent]
    assert f["pass_through_ratio"] > 0.99
    assert f["median_dwell_s"] == 60.0
    assert f["rapid_spend_share"] == 1.0
