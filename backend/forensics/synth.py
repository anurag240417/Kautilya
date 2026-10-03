"""Synthetic Bitcoin transaction + network dataset with planted laundering.

Produces a ``RawDataset`` in the problem-statement schema, with ground truth,
from a small UTXO-accurate simulation (inputs always equal outputs + fee).

Illicit scenarios (each spawns fresh entities, so scenarios can be held out):

* ``ransomware``      victims pay a burst of similar amounts -> consolidation ->
                      peel chain to cash-out -> exchange
* ``darknet_market``  many small deposits (fan-in) -> batch vendor payouts ->
                      vendors cash out (layering / CoinJoin / split deposits)
* ``layering``        rapid single-output hops through mule entities
* ``dust_attack``     one sender spraying 546-sat outputs at many addresses

Licit look-alikes deliberately overlap the illicit patterns so detection is
not trivial: exchanges that batch-pay and consolidate, merchants with heavy
fan-in, payment processors that forward within minutes, privacy users on Tor
who join CoinJoin rounds.

Everything here is synthetic.  No real addresses, IPs or transactions.
"""

from __future__ import annotations

import heapq
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from backend.forensics.dataset import NS_PER_S, GroundTruth, IngestReport, RawDataset
from backend.forensics.geo import default_geoip, hosting_vpn_asns, tor_exit_ips

logger = logging.getLogger(__name__)

SAT = 100_000_000
DUST_SAT = 546
SCENARIOS = ("ransomware", "darknet_market", "layering", "dust_attack")
EPOCH_START = pd.Timestamp("2025-01-01", tz="UTC").value  # ns

_B58 = np.frombuffer(b"123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz", dtype="S1")
_BECH = np.frombuffer(b"qpzry9x8gf2tvdw0s3jn54khce6mua7l", dtype="S1")
_SCRIPTS = ("P2PKH", "P2SH", "P2WPKH", "TAPROOT")


@dataclass
class SynthConfig:
    """Size and behaviour knobs.  ``n_tx`` is a target; the result is close, not exact."""

    n_tx: int = 20_000
    days: float = 30.0
    seed: int = 7
    n_sensors: int = 3
    # Campaign counts; None = scale with n_tx.
    n_ransomware: int | None = None
    n_darknet: int | None = None
    n_layering: int | None = None
    n_dust: int | None = None


class _Ent:
    __slots__ = (
        "id",
        "kind",
        "illicit",
        "scenario",
        "role",
        "script",
        "utxos",
        "addrs",
        "ips",
        "tor_p",
        "vpn_p",
        "deposit",
        "camp",
    )

    def __init__(
        self,
        eid,
        kind,
        script,
        illicit=False,
        scenario="normal",
        role="user",
        tor_p=0.0,
        vpn_p=0.0,
        camp=-1,
    ):
        self.id = eid
        self.kind = kind
        self.illicit = illicit
        self.scenario = scenario
        self.role = role
        self.script = script
        self.utxos: list[tuple[int, int]] = []
        self.addrs: list[int] = []
        self.ips: list[str] = []
        self.tor_p = tor_p
        self.vpn_p = vpn_p
        self.deposit: dict[int, int] = {}
        self.camp = camp


class _Sim:
    def __init__(self, cfg: SynthConfig) -> None:
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)
        self.t0 = 0.0
        self.T = cfg.days * 86400.0
        self.heap: list = []
        self.seq = 0
        self.ents: list[_Ent] = []
        self.addr_ent: list[int] = []
        self.addr_script: list[int] = []
        # tx storage (python lists -> arrays at end)
        self.tx_ts: list[float] = []
        self.tx_fee: list[int] = []
        self.tx_script: list[int] = []
        self.tx_scn: list[str] = []
        self.in_tx: list[int] = []
        self.in_addr: list[int] = []
        self.in_amt: list[int] = []
        self.out_tx: list[int] = []
        self.out_addr: list[int] = []
        self.out_amt: list[int] = []
        self.obs: list[tuple] = []  # (tx, ts, src_ip, dst_ip, sport, dport, kind)
        self.tor = sorted(tor_exit_ips()) or ["198.51.100.10"]
        self.sensors = [self._rand_ip() for _ in range(cfg.n_sensors)]
        self.exchanges: list[_Ent] = []
        self.merchants: list[_Ent] = []
        self.processors: list[_Ent] = []
        self.users: list[_Ent] = []
        self.privacy: list[_Ent] = []
        self.miner = self._ent("miner", 0, role="miner")
        self.campaigns = 0

    # ---------------- plumbing ----------------
    def _rand_ip(self) -> str:
        r = self.rng
        # first octets that the builtin GeoIP table covers
        first = int(
            r.choice(
                [
                    3,
                    6,
                    8,
                    11,
                    13,
                    16,
                    20,
                    25,
                    32,
                    38,
                    45,
                    48,
                    52,
                    55,
                    62,
                    77,
                    82,
                    91,
                    103,
                    110,
                    122,
                    130,
                    150,
                    170,
                    185,
                    190,
                    201,
                ]
            )
        )
        return f"{first}.{r.integers(0, 256)}.{r.integers(0, 256)}.{r.integers(1, 255)}"

    def _ent(self, kind, script=None, **kw) -> _Ent:
        script = int(self.rng.choice(4, p=[0.35, 0.2, 0.35, 0.1])) if script is None else script
        e = _Ent(len(self.ents), kind, script, **kw)
        e.ips = [self._rand_ip() for _ in range(1 + int(self.rng.integers(0, 2)))]
        self.ents.append(e)
        return e

    def new_addr(self, e: _Ent) -> int:
        a = len(self.addr_ent)
        self.addr_ent.append(e.id)
        self.addr_script.append(e.script)
        e.addrs.append(a)
        return a

    def schedule(self, t: float, fn) -> None:
        if t < self.T * 1.05:
            self.seq += 1
            heapq.heappush(self.heap, (t, self.seq, fn))

    def _origin(self, e: _Ent) -> tuple[str, int]:
        r = self.rng.random()
        if r < e.tor_p:
            return self.tor[int(self.rng.integers(len(self.tor)))], 1
        if r < e.tor_p + e.vpn_p:
            return self._rand_ip(), 2
        return e.ips[int(self.rng.integers(len(e.ips)))], 0

    # ---------------- core: emit a transaction ----------------
    def emit(self, ts, sender: _Ent | None, inputs, outputs, scenario="normal", fee=None):
        rng = self.rng
        n_in, n_out = len(inputs), len(outputs)
        vbytes = 10 + 68 * n_in + 31 * n_out
        if fee is None:
            fee = int(vbytes * rng.uniform(4, 60)) if n_in else 0
        tx = len(self.tx_ts)
        self.tx_ts.append(ts)
        self.tx_fee.append(fee)
        script = sender.script if sender is not None else 0
        self.tx_script.append(script)
        self.tx_scn.append(scenario)
        for a, v in inputs:
            self.in_tx.append(tx)
            self.in_addr.append(a)
            self.in_amt.append(v)
        outs = []
        for a, v in outputs:
            self.out_tx.append(tx)
            self.out_addr.append(a)
            self.out_amt.append(v)
            owner = self.ents[self.addr_ent[a]]
            owner.utxos.append((a, v))
            outs.append((a, v))
        # network observations
        if sender is not None:
            ip, kind = self._origin(sender)
        else:
            ip, kind = self._rand_ip(), 0
        sensor = self.sensors[int(rng.integers(len(self.sensors)))]
        self.obs.append(
            (
                tx,
                ts + rng.uniform(0.01, 0.4),
                ip,
                sensor,
                int(rng.integers(1024, 65535)),
                8333,
                kind,
            )
        )
        for _ in range(int(rng.integers(0, 3))):
            self.obs.append(
                (
                    tx,
                    ts + rng.uniform(0.2, 4.0),
                    self._rand_ip(),
                    sensor,
                    int(rng.integers(1024, 65535)),
                    8333,
                    0,
                )
            )
        return outs

    def select(self, e: _Ent, target: int, combine: bool = True):
        """Pick utxos covering ``target`` sat; removes them from the wallet. None if broke."""
        if not e.utxos:
            return None
        rng = self.rng
        cover = [u for u in e.utxos if u[1] >= target]
        if cover and (not combine or rng.random() < 0.7):
            pick = (
                min(cover, key=lambda u: u[1])
                if rng.random() < 0.6
                else max(cover, key=lambda u: u[1])
            )
            e.utxos.remove(pick)
            return [pick]
        order = sorted(e.utxos, key=lambda u: -u[1])
        chosen, tot = [], 0
        for u in order[:12]:
            chosen.append(u)
            tot += u[1]
            if tot >= target:
                break
        if tot < target:
            return None
        for u in chosen:
            e.utxos.remove(u)
        return chosen

    def pay(self, ts, e: _Ent, dest_outputs, scenario="normal", change_to_new=True) -> bool:
        """Spend from ``e`` to ``[(addr, sat), ...]``, change to a fresh address of ``e``."""
        rng = self.rng
        total = sum(v for _, v in dest_outputs)
        est_fee = int((10 + 68 * 2 + 31 * (len(dest_outputs) + 1)) * 60)
        ins = self.select(e, total + est_fee)
        if ins is None:
            return False
        tin = sum(v for _, v in ins)
        n_out = len(dest_outputs) + 1
        fee = int((10 + 68 * len(ins) + 31 * n_out) * rng.uniform(4, 60))
        change = tin - total - fee
        if tin < total:
            e.utxos.extend(ins)
            return False
        outs = list(dest_outputs)
        if change > DUST_SAT * 2:
            ca = self.new_addr(e) if change_to_new else e.addrs[0]
            outs.insert(int(rng.integers(0, len(outs) + 1)), (ca, change))
        self.emit(ts, e, ins, outs, scenario, fee=tin - sum(v for _, v in outs))
        return True

    def sweep(self, ts, e: _Ent, dest: int, scenario="normal", max_inputs=40) -> bool:
        """Spend up to ``max_inputs`` of e's coins into one output (consolidation/forward)."""
        if not e.utxos:
            return False
        ins = sorted(e.utxos, key=lambda u: -u[1])[:max_inputs]
        for u in ins:
            e.utxos.remove(u)
        tin = sum(v for _, v in ins)
        fee = int((10 + 68 * len(ins) + 31) * self.rng.uniform(4, 60))
        if tin - fee < DUST_SAT * 2:
            e.utxos.extend(ins)
            return False
        self.emit(ts, e, ins, [(dest, tin - fee)], scenario, fee=fee)
        return True

    def exch_deposit(self, ex: _Ent, user_id: int) -> int:
        a = ex.deposit.get(user_id)
        if a is None:
            a = self.new_addr(ex)
            ex.deposit[user_id] = a
        return a

    def coinjoin(self, ts, participants: list[_Ent], scenario="coinjoin") -> bool:
        rng = self.rng
        denom = int(rng.choice([0.01, 0.05, 0.1, 0.5]) * SAT)
        ins_all, outs, used = [], [], []
        fee_each = int(150 * 68 * rng.uniform(4, 40) / 68)
        for p in participants:
            ins = self.select(p, denom + fee_each, combine=False)
            if ins is None:
                continue
            tin = sum(v for _, v in ins)
            ins_all.extend(ins)
            outs.append((self.new_addr(p), denom))
            change = tin - denom - fee_each
            if change > DUST_SAT * 2:
                outs.append((self.new_addr(p), change))
            used.append(p)
        if len(used) < 4:
            for a, v in ins_all:  # roll back: return coins to their owners
                self.ents[self.addr_ent[a]].utxos.append((a, v))
            return False
        order = rng.permutation(len(outs))
        outs = [outs[i] for i in order]
        ins_all = [ins_all[i] for i in rng.permutation(len(ins_all))]
        tin = sum(v for _, v in ins_all)
        self.emit(ts, used[0], ins_all, outs, scenario, fee=tin - sum(v for _, v in outs))
        return True

    # ---------------- background economy ----------------
    def build_economy(self) -> None:
        cfg, rng = self.cfg, self.rng
        n_users = max(60, int(cfg.n_tx * 0.16))
        n_ex = max(2, n_users // 900)
        n_merch = max(3, n_users // 120)
        n_proc = max(2, n_users // 700)

        for _ in range(n_ex):
            e = self._ent("exchange", role="exchange", tor_p=0.0)
            for _ in range(3):
                self.new_addr(e)
            self.exchanges.append(e)
            self.fund(e, int(rng.uniform(50, 200) * SAT), k=6)
        for _ in range(n_merch):
            e = self._ent("merchant", role="merchant")
            for _ in range(int(rng.integers(1, 4))):
                self.new_addr(e)
            self.merchants.append(e)
        for _ in range(n_proc):
            e = self._ent("processor", role="processor")
            for _ in range(2):
                self.new_addr(e)
            self.processors.append(e)
        for i in range(n_users):
            priv = rng.random() < 0.08
            e = self._ent(
                "privacy_user" if priv else "user",
                role="user",
                tor_p=float(rng.uniform(0.3, 0.7))
                if priv
                else (0.02 if rng.random() < 0.1 else 0.0),
                vpn_p=0.05 if rng.random() < 0.15 else 0.0,
            )
            self.new_addr(e)
            self.users.append(e)
            if priv:
                self.privacy.append(e)
            self.fund(
                e,
                int(np.exp(rng.normal(np.log(0.4), 1.0)) * SAT) + 200_000,
                k=int(rng.integers(1, 4)),
            )
            for _ in range(int(rng.poisson(4.5))):
                self.schedule(
                    rng.uniform(3600, self.T), (lambda e=e: lambda t: self.user_payment(t, e))()
                )

        # exchange activity
        for ex in self.exchanges:
            for _ in range(int(self.cfg.days * max(1.0, n_users / 4000))):
                self.schedule(
                    rng.uniform(86400, self.T),
                    (lambda ex=ex: lambda t: self.exchange_batch(t, ex))(),
                )
            for d in range(2, int(self.cfg.days), max(1, 3)):
                self.schedule(
                    d * 86400 + rng.uniform(0, 3600),
                    (lambda ex=ex: lambda t: self.exchange_consolidate(t, ex))(),
                )
        # legit CoinJoin rounds
        for _ in range(max(2, cfg.n_tx // 1500)):
            self.schedule(rng.uniform(86400, self.T), self.cj_round)

    def fund(self, e: _Ent, amount_sat: int, k: int = 1) -> None:
        rng = self.rng
        per = amount_sat // k
        outs = [
            (self.new_addr(e) if e.kind != "exchange" else e.addrs[i % len(e.addrs)], per)
            for i in range(k)
        ]
        self.emit(rng.uniform(0, 3000), None, [], outs, "normal", fee=0)

    def cj_round(self, t):
        if len(self.privacy) < 4:
            return
        rng = self.rng
        idx = rng.choice(
            len(self.privacy), size=min(len(self.privacy), int(rng.integers(5, 9))), replace=False
        )
        self.coinjoin(t, [self.privacy[i] for i in idx])

    def user_payment(self, t, u: _Ent) -> None:
        rng = self.rng
        bal = sum(v for _, v in u.utxos)
        if bal < 300_000:
            return
        amt = int(min(np.exp(rng.normal(np.log(0.01), 1.3)) * SAT, bal * 0.6))
        if rng.random() < 0.35:
            amt = max(100_000, (amt // 100_000) * 100_000)
        r = rng.random()
        if r < 0.72:
            other = self.users[int(rng.integers(len(self.users)))]
            if other.id == u.id:
                return
            dest = (
                self.new_addr(other)
                if (rng.random() < 0.6 or not other.addrs)
                else other.addrs[int(rng.integers(len(other.addrs)))]
            )
        elif r < 0.84:
            m = self.merchants[int(rng.integers(len(self.merchants)))]
            dest = m.addrs[int(rng.integers(len(m.addrs)))]
        elif r < 0.94:
            ex = self.exchanges[int(rng.integers(len(self.exchanges)))]
            dest = self.exch_deposit(ex, u.id)
        else:
            p = self.processors[int(rng.integers(len(self.processors)))]
            dest = p.addrs[int(rng.integers(len(p.addrs)))]
            if self.pay(t, u, [(dest, amt)]):
                self.schedule(
                    t + rng.uniform(120, 1200),
                    (lambda p=p: lambda tt: self.processor_forward(tt, p))(),
                )
            return
        self.pay(t, u, [(dest, amt)])

    def processor_forward(self, t, p: _Ent) -> None:
        rng = self.rng
        tgt = self.users[int(rng.integers(len(self.users)))]
        self.sweep(t, p, self.new_addr(tgt), max_inputs=3)

    def exchange_batch(self, t, ex: _Ent) -> None:
        rng = self.rng
        k = int(rng.integers(5, 26))
        outs = []
        for _ in range(k):
            u = self.users[int(rng.integers(len(self.users)))]
            outs.append(
                (self.new_addr(u), int(np.exp(rng.normal(np.log(0.02), 0.9)) * SAT) + 50_000)
            )
        self.pay(t, ex, outs, change_to_new=False)

    def exchange_consolidate(self, t, ex: _Ent) -> None:
        if len(ex.utxos) >= 6:
            self.sweep(t, ex, ex.addrs[0], max_inputs=40)

    # ---------------- illicit scenarios ----------------
    def _illicit(self, kind, scenario, role, camp, tor=(0.4, 0.8), vpn=(0.1, 0.4)) -> _Ent:
        rng = self.rng
        return self._ent(
            kind,
            illicit=True,
            scenario=scenario,
            role=role,
            camp=camp,
            tor_p=float(rng.uniform(*tor)),
            vpn_p=float(rng.uniform(*vpn)),
        )

    def _exchange_dest(self, for_key: int) -> int:
        ex = self.exchanges[int(self.rng.integers(len(self.exchanges)))]
        return self.exch_deposit(ex, 10_000_000 + for_key)

    def launch_ransomware(self, t0: float) -> None:
        rng = self.rng
        camp = self.campaigns
        self.campaigns += 1
        op = self._illicit("ransomware_operator", "ransomware", "operator", camp)
        collectors = [self.new_addr(op) for _ in range(int(rng.integers(3, 7)))]
        ransom = float(rng.choice([0.05, 0.1, 0.25, 0.5]))
        n_v = int(rng.integers(15, 45))
        last = t0
        for _ in range(n_v):
            tv = t0 + float(rng.exponential(14 * 3600))
            last = max(last, tv)
            v = self.users[int(rng.integers(len(self.users)))]
            amt = int(ransom * SAT * rng.uniform(0.9, 1.1))
            a = collectors[int(rng.integers(len(collectors)))]
            self.schedule(
                tv,
                (lambda v=v, a=a, amt=amt: lambda t: self.pay(t, v, [(a, amt)], "ransomware"))(),
            )
        cashout = self._illicit("cashout_mule", "ransomware", "cashout", camp, tor=(0.2, 0.5))
        tc = last + float(rng.uniform(2, 12)) * 3600
        hops = int(rng.integers(6, 15))
        use_cj = rng.random() < 0.3

        def consolidate(t):
            dest = self.new_addr(op)
            if self.sweep(t, op, dest, "ransomware"):
                self.schedule(t + float(rng.uniform(600, 6 * 3600)), lambda tt: peel(tt, hops))

        def peel(t, left):
            if not op.utxos:
                return
            if left <= 0:
                self.sweep(t, op, self._exchange_dest(camp), "ransomware")
                return
            tot = sum(v for _, v in op.utxos)
            peel_amt = max(200_000, int(tot * rng.uniform(0.02, 0.12)))
            ok = self.pay(t, op, [(self.new_addr(cashout), peel_amt)], "ransomware")
            if ok:
                # cash-out entity forwards some peeled funds onward to an exchange
                if rng.random() < 0.4:
                    self.schedule(
                        t + float(rng.uniform(900, 4 * 3600)),
                        lambda tt: self.sweep(
                            tt, cashout, self._exchange_dest(camp), "ransomware", 5
                        ),
                    )
            if use_cj and left == hops // 2:
                parts = (
                    [op] + [self.privacy[int(rng.integers(len(self.privacy)))] for _ in range(5)]
                    if self.privacy
                    else []
                )
                if parts:
                    self.coinjoin(t + 120, list(dict.fromkeys(parts)))
            self.schedule(
                t + float(np.exp(rng.normal(np.log(1800), 1.0))), lambda tt: peel(tt, left - 1)
            )

        self.schedule(tc, consolidate)

    def launch_darknet(self, t0: float) -> None:
        rng = self.rng
        camp = self.campaigns
        self.campaigns += 1
        mk = self._illicit("darknet_market", "darknet_market", "escrow", camp, tor=(0.3, 0.6))
        escrow = [self.new_addr(mk) for _ in range(5)]
        vendors = [
            self._illicit("darknet_vendor", "darknet_market", "vendor", camp, tor=(0.5, 0.9))
            for _ in range(int(rng.integers(3, 7)))
        ]
        n_buy = int(rng.integers(60, 160))
        span = float(rng.uniform(6, 14)) * 86400
        for _ in range(n_buy):
            tb = t0 + float(rng.uniform(0, span))
            b = self.users[int(rng.integers(len(self.users)))]
            amt = int(np.exp(rng.normal(np.log(0.01), 0.8)) * SAT) + 50_000
            a = escrow[int(rng.integers(len(escrow)))]
            self.schedule(
                tb,
                (
                    lambda b=b, a=a, amt=amt: (
                        lambda t: self.pay(t, b, [(a, amt)], "darknet_market")
                    )
                )(),
            )

        def payout(t):
            bal = sum(v for _, v in mk.utxos)
            if bal < 2_000_000:
                return
            share = bal * rng.uniform(0.4, 0.8)
            outs, weights = [], rng.dirichlet(np.ones(len(vendors)))
            for vnd, w in zip(vendors, weights, strict=True):
                outs.append((self.new_addr(vnd), max(150_000, int(share * w))))
            self.pay(t, mk, outs, "darknet_market")
            for vnd in vendors:
                self.schedule(
                    t + float(rng.uniform(6 * 3600, 3 * 86400)),
                    (lambda vnd=vnd: lambda tt: self.vendor_cashout(tt, vnd, camp))(),
                )

        d = 1.0
        while d * 86400 < span + 2 * 86400:
            self.schedule(t0 + d * 86400 + float(rng.uniform(0, 4 * 3600)), payout)
            d += float(rng.uniform(1.0, 2.0))

    def vendor_cashout(self, t, v: _Ent, camp: int) -> None:
        rng = self.rng
        if not v.utxos:
            return
        r = rng.random()
        if r < 0.45:
            self.layer_chain(t, v, int(rng.integers(3, 7)), v.scenario, camp)
        elif r < 0.75 and self.privacy:
            parts = [v] + [self.privacy[int(rng.integers(len(self.privacy)))] for _ in range(5)]
            if self.coinjoin(t, list(dict.fromkeys(parts))):
                self.schedule(
                    t + float(rng.uniform(3600, 86400)),
                    lambda tt: self.sweep(tt, v, self._exchange_dest(camp), v.scenario),
                )
        else:
            k = int(rng.integers(2, 5))
            dest = self._exchange_dest(camp)
            for i in range(k):
                self.schedule(
                    t + i * float(rng.uniform(1800, 7200)),
                    lambda tt: (
                        self.pay(
                            tt,
                            v,
                            [
                                (
                                    dest,
                                    max(
                                        200_000,
                                        int(sum(x for _, x in v.utxos) * rng.uniform(0.3, 0.9)),
                                    ),
                                )
                            ],
                            v.scenario,
                        )
                        if v.utxos
                        else None
                    ),
                )

    def layer_chain(self, t, src: _Ent, length: int, scenario: str, camp: int) -> None:
        rng = self.rng
        holder = src
        tt = t
        for i in range(length):
            mule = self._illicit(
                "layering_mule", scenario, "mule", camp, tor=(0.1, 0.4), vpn=(0.1, 0.4)
            )
            dest = self.new_addr(mule)
            cur = holder
            self.schedule(
                tt, (lambda cur=cur, dest=dest: lambda x: self.sweep(x, cur, dest, scenario, 40))()
            )
            holder = mule
            tt += float(rng.uniform(60, 1200))
        final = self._exchange_dest(camp)
        self.schedule(tt, (lambda cur=holder: lambda x: self.sweep(x, cur, final, scenario, 40))())

    def launch_layering(self, t0: float) -> None:
        rng = self.rng
        camp = self.campaigns
        self.campaigns += 1
        src = self._illicit("layering_source", "layering", "source", camp)
        a = self.new_addr(src)
        amt = int(rng.uniform(0.5, 6) * SAT)
        self.schedule(t0 - 10, lambda t: self.emit(t, None, [], [(a, amt)], "layering", fee=0))
        self.schedule(
            t0, lambda t: self.layer_chain(t, src, int(rng.integers(5, 11)), "layering", camp)
        )

    def launch_dust(self, t0: float) -> None:
        rng = self.rng
        camp = self.campaigns
        self.campaigns += 1
        at = self._illicit("dust_attacker", "dust_attack", "dust_attacker", camp, tor=(0.3, 0.7))
        a = self.new_addr(at)
        self.schedule(
            t0 - 10, lambda t: self.emit(t, None, [], [(a, int(0.05 * SAT))], "dust_attack", fee=0)
        )
        n_targets = int(rng.integers(40, 130))
        targets = []
        for _ in range(n_targets):
            u = self.users[int(rng.integers(len(self.users)))]
            targets.append(
                u.addrs[int(rng.integers(len(u.addrs)))] if u.addrs else self.new_addr(u)
            )
        per_tx = int(rng.integers(20, 60))
        for k in range(0, n_targets, per_tx):
            chunk = targets[k : k + per_tx]
            self.schedule(
                t0 + (k // per_tx) * float(rng.uniform(300, 3600)),
                (
                    lambda chunk=chunk: (
                        lambda t: self.pay(t, at, [(x, DUST_SAT) for x in chunk], "dust_attack")
                    )
                )(),
            )

    # ---------------- run ----------------
    def run(self) -> RawDataset:
        cfg, rng = self.cfg, self.rng
        self.build_economy()
        n_r = cfg.n_ransomware if cfg.n_ransomware is not None else max(3, cfg.n_tx // 4000)
        n_d = cfg.n_darknet if cfg.n_darknet is not None else max(2, cfg.n_tx // 7000)
        n_l = cfg.n_layering if cfg.n_layering is not None else max(3, cfg.n_tx // 2500)
        n_du = cfg.n_dust if cfg.n_dust is not None else max(2, cfg.n_tx // 4000)
        late = self.T * 0.55  # leave room for campaigns to play out
        for _ in range(n_r):
            self.launch_ransomware(float(rng.uniform(86400, late)))
        for _ in range(n_d):
            self.launch_darknet(float(rng.uniform(86400, late)))
        for _ in range(n_l):
            self.launch_layering(float(rng.uniform(86400, late)))
        for _ in range(n_du):
            self.launch_dust(float(rng.uniform(86400, late)))

        while self.heap:
            t, _, fn = heapq.heappop(self.heap)
            fn(t)
        return self.finish()

    def finish(self) -> RawDataset:
        rng = self.rng
        n_tx = len(self.tx_ts)
        ts_s = np.array(self.tx_ts)
        order = np.argsort(ts_s, kind="stable")
        rank = np.empty(n_tx, dtype=np.int64)
        rank[order] = np.arange(n_tx)

        addr_script = np.array(self.addr_script, dtype=np.int8)
        addresses = _make_addresses(rng, addr_script)
        txids = _make_txids(rng, n_tx)

        def ns(x):
            return EPOCH_START + (np.asarray(x) * NS_PER_S).astype(np.int64)

        tx = pd.DataFrame(
            {
                "txid": txids[order],
                "ts": ns(ts_s[order]),
                "fee": np.array(self.tx_fee, dtype=np.float64)[order] / SAT,
                "script_type": np.array(_SCRIPTS, dtype=object)[np.array(self.tx_script)[order]],
            }
        )
        inputs = pd.DataFrame(
            {
                "tx_idx": rank[np.array(self.in_tx, dtype=np.int64)],
                "addr": np.array(self.in_addr, dtype=np.int32),
                "amount": np.array(self.in_amt, dtype=np.float64) / SAT,
            }
        )
        outputs = pd.DataFrame(
            {
                "tx_idx": rank[np.array(self.out_tx, dtype=np.int64)],
                "addr": np.array(self.out_addr, dtype=np.int32),
                "amount": np.array(self.out_amt, dtype=np.float64) / SAT,
            }
        )
        inputs = inputs.sort_values("tx_idx", kind="stable").reset_index(drop=True)
        outputs = outputs.sort_values("tx_idx", kind="stable").reset_index(drop=True)
        outputs["pos"] = (outputs.groupby("tx_idx").cumcount()).astype(np.int32)
        tx["n_in"] = (
            inputs.groupby("tx_idx")
            .size()
            .reindex(np.arange(n_tx), fill_value=0)
            .to_numpy()
            .astype(np.int32)
        )
        tx["n_out"] = (
            outputs.groupby("tx_idx")
            .size()
            .reindex(np.arange(n_tx), fill_value=0)
            .to_numpy()
            .astype(np.int32)
        )

        o = self.obs
        o_tx = rank[np.array([x[0] for x in o], dtype=np.int64)]
        o_ts = np.array([x[1] for x in o])
        kinds = np.array([x[6] for x in o], dtype=np.int8)
        src_ip = np.array([x[2] for x in o], dtype=object)
        obs = pd.DataFrame(
            {
                "tx_idx": o_tx,
                "ts": ns(o_ts),
                "src_ip": src_ip,
                "dst_ip": np.array([x[3] for x in o], dtype=object),
                "src_port": np.array([x[4] for x in o], dtype=np.int32),
                "dst_port": np.array([x[5] for x in o], dtype=np.int32),
            }
        )
        country, asn = default_geoip().lookup_many(src_ip)
        vpn_asn = sorted(hosting_vpn_asns()) or [9009]
        vm = kinds == 2
        asn = asn.copy()
        asn[vm] = rng.choice(vpn_asn, size=int(vm.sum()))
        obs["country"] = country
        obs["asn"] = asn
        obs = obs.sort_values(["ts"], kind="stable").reset_index(drop=True)

        entities = pd.DataFrame(
            {
                "entity_id": [e.id for e in self.ents],
                "is_illicit": [e.illicit for e in self.ents],
                "scenario": [e.scenario for e in self.ents],
                "role": [e.role for e in self.ents],
                "kind": [e.kind for e in self.ents],
                "campaign": [e.camp for e in self.ents],
            }
        )
        truth = GroundTruth(
            addr_entity=np.array(self.addr_ent, dtype=np.int64),
            entities=entities,
            tx_scenario=np.array(self.tx_scn, dtype=object)[order],
        )
        report = IngestReport(source=f"synthetic(seed={self.cfg.seed})", rows_read=len(obs))
        ds = RawDataset(
            addresses=addresses,
            tx=tx,
            inputs=inputs,
            outputs=outputs,
            obs=obs,
            truth=truth,
            report=report,
        )
        logger.info("Synthetic dataset: %s", ds.summary())
        return ds


def _make_addresses(rng: np.random.Generator, scripts: np.ndarray) -> np.ndarray:
    out = np.empty(len(scripts), dtype=object)
    for code, prefix, alpha, length in (
        (0, b"1", _B58, 33),
        (1, b"3", _B58, 33),
        (2, b"bc1q", _BECH, 38),
        (3, b"bc1p", _BECH, 58),
    ):
        idx = np.flatnonzero(scripts == code)
        if len(idx) == 0:
            continue
        chars = alpha[rng.integers(0, len(alpha), (len(idx), length))]
        strs = chars.view(f"S{length}").reshape(-1)
        out[idx] = [(prefix + s).decode() for s in strs]
    return out


def _make_txids(rng: np.random.Generator, n: int) -> np.ndarray:
    raw = rng.bytes(32 * n).hex()
    return np.array([raw[i * 64 : (i + 1) * 64] for i in range(n)], dtype=object)


def generate_dataset(cfg: SynthConfig | None = None, **kw) -> RawDataset:
    """Generate a synthetic ``RawDataset`` (with ground truth)."""
    cfg = cfg or SynthConfig(**kw)
    return _Sim(cfg).run()
