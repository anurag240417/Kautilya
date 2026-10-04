"""Offline IP intelligence: country/ASN lookup plus Tor and hosting/VPN flags.

All data is local.  Three lookup tiers, best first:

1. **Real open databases** dropped into ``backend/reference/geoip/`` (or the
   directory named by ``KAUTILYA_GEOIP_DIR``):

   * ``dbip-country-lite.csv``  - DB-IP "IP to Country Lite" (CC-BY 4.0):
     rows of ``start_ip,end_ip,country``
   * ``asn.csv`` - either DB-IP "IP to ASN Lite" (``start_ip,end_ip,asn,org``)
     or MaxMind GeoLite2-ASN-Blocks-IPv4 (``network,asn,org``)

   See ``docs/GEOIP_SETUP.md`` for how to fetch these once, on a connected
   machine, and carry them over.
2. The coarse built-in first-octet table used by the synthetic generator.

Tor exit nodes come from ``reference/tor_exit_nodes.txt`` (one IP per line;
the bundled file is a synthetic sample - replace with the Tor Project bulk
exit list).  Hosting/VPN providers are flagged by ASN from
``reference/vpn_hosting_asns.txt``.
"""

from __future__ import annotations

import ipaddress
import logging
import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REFERENCE_DIR = Path(__file__).resolve().parent.parent / "reference"


def geoip_dir() -> Path:
    return Path(os.environ.get("KAUTILYA_GEOIP_DIR", str(REFERENCE_DIR / "geoip")))


# ======================================================================
# IP helpers
# ======================================================================


def ip_to_int(ips: pd.Series | np.ndarray | list[str]) -> np.ndarray:
    """IPv4 strings -> int64 array; invalid or IPv6 entries -> -1."""
    s = pd.Series(ips, dtype="object").fillna("").astype(str)
    parts = s.str.split(".", expand=True)
    if parts.shape[1] != 4:
        return np.full(len(s), -1, dtype=np.int64)
    nums = parts.apply(pd.to_numeric, errors="coerce")
    valid = nums.notna().all(axis=1) & (nums.ge(0) & nums.le(255)).all(axis=1)
    out = (
        nums[0].fillna(0).astype(np.int64) * 16777216
        + nums[1].fillna(0).astype(np.int64) * 65536
        + nums[2].fillna(0).astype(np.int64) * 256
        + nums[3].fillna(0).astype(np.int64)
    ).to_numpy()
    return np.where(valid.to_numpy(), out, -1)


def int_to_ip(value: int) -> str:
    return str(ipaddress.IPv4Address(int(value)))


class _RangeTable:
    """Sorted, non-overlapping [start, end] -> value table with vectorised lookup."""

    def __init__(self, starts: np.ndarray, ends: np.ndarray, values: np.ndarray) -> None:
        order = np.argsort(starts, kind="stable")
        self.starts = starts[order]
        self.ends = ends[order]
        self.values = values[order]

    def lookup(self, ips: np.ndarray, default):
        if len(self.starts) == 0:
            return np.full(len(ips), default, dtype=object if default is None else type(default))
        idx = np.searchsorted(self.starts, ips, side="right") - 1
        safe = np.clip(idx, 0, len(self.starts) - 1)
        hit = (idx >= 0) & (ips >= 0) & (ips <= self.ends[safe])
        out = np.where(hit, self.values[safe], default)
        return out


def _load_country_csv(path: Path) -> _RangeTable:
    df = pd.read_csv(path, header=None, names=["start", "end", "country"], dtype=str)
    starts = ip_to_int(df["start"])
    ends = ip_to_int(df["end"])
    keep = (starts >= 0) & (ends >= 0)  # IPv4 only
    return _RangeTable(starts[keep], ends[keep], df["country"].to_numpy()[keep].astype(object))


def _load_asn_csv(path: Path) -> _RangeTable:
    df = pd.read_csv(path, header=None, dtype=str)
    if (
        df[0].astype(str).str.contains("/", regex=False).any()
    ):  # MaxMind: network,asn,org (+ header row)
        df = df[df[0].str.contains("/", na=False)]
        nets = [ipaddress.ip_network(n, strict=False) for n in df[0]]
        v4 = [isinstance(n, ipaddress.IPv4Network) for n in nets]
        starts = np.array(
            [int(n.network_address) if ok else -1 for n, ok in zip(nets, v4, strict=True)]
        )
        ends = np.array(
            [int(n.broadcast_address) if ok else -1 for n, ok in zip(nets, v4, strict=True)]
        )
        asns = pd.to_numeric(df[1], errors="coerce").fillna(0).astype(np.int64).to_numpy()
    else:  # DB-IP: start,end,asn,org
        starts = ip_to_int(df[0])
        ends = ip_to_int(df[1])
        asns = pd.to_numeric(df[2], errors="coerce").fillna(0).astype(np.int64).to_numpy()
    keep = starts >= 0
    return _RangeTable(starts[keep], ends[keep], asns[keep])


class GeoIPDatabase:
    """Vectorised IPv4 country/ASN resolver with layered fallbacks."""

    def __init__(self, directory: Path | None = None) -> None:
        directory = directory or geoip_dir()
        self.country_table: _RangeTable | None = None
        self.asn_table: _RangeTable | None = None
        self.source = "builtin_first_octet"

        cpath = directory / "dbip-country-lite.csv"
        apath = directory / "asn.csv"
        try:
            if cpath.exists():
                self.country_table = _load_country_csv(cpath)
                self.source = "dbip_country_lite"
            if apath.exists():
                self.asn_table = _load_asn_csv(apath)
                self.source += "+asn_csv"
        except Exception:  # corrupted user-supplied file must not kill the app
            logger.exception(
                "Failed loading GeoIP database from %s; using builtin table", directory
            )
            self.country_table = self.asn_table = None
            self.source = "builtin_first_octet"

        from backend.generator.geoip import _BUILTIN_RANGES

        self._oct_country = np.full(256, "", dtype=object)
        self._oct_asn = np.zeros(256, dtype=np.int64)
        for lo, hi, country, asn in _BUILTIN_RANGES:
            self._oct_country[lo : hi + 1] = country
            self._oct_asn[lo : hi + 1] = asn

    def lookup_many(self, ips) -> tuple[np.ndarray, np.ndarray]:
        """Return (country codes, ASNs); unknown -> ``""`` / ``0``."""
        ints = ip_to_int(ips)
        first = np.where(ints >= 0, ints // 16777216, 0)
        country = np.where(ints >= 0, self._oct_country[first], "")
        asn = np.where(ints >= 0, self._oct_asn[first], 0)
        if self.country_table is not None:
            c = self.country_table.lookup(ints, None)
            country = np.where(pd.notna(c), c, country)
        if self.asn_table is not None:
            a = self.asn_table.lookup(ints, 0)
            asn = np.where(a > 0, a, asn)
        return country.astype(object), asn.astype(np.int64)


@lru_cache(maxsize=1)
def default_geoip() -> GeoIPDatabase:
    return GeoIPDatabase()


# ======================================================================
# Tor / hosting / VPN reference lists
# ======================================================================


def _read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return [
        ln.strip()
        for ln in path.read_text(encoding="utf-8").splitlines()
        if ln.strip() and not ln.strip().startswith("#")
    ]


@lru_cache(maxsize=1)
def tor_exit_ips() -> frozenset[str]:
    return frozenset(_read_lines(REFERENCE_DIR / "tor_exit_nodes.txt"))


@lru_cache(maxsize=1)
def hosting_vpn_asns() -> frozenset[int]:
    out = set()
    for ln in _read_lines(REFERENCE_DIR / "vpn_hosting_asns.txt"):
        tok = ln.split()[0].upper().removeprefix("AS")
        if tok.isdigit():
            out.add(int(tok))
    return frozenset(out)


def flag_tor(ips) -> np.ndarray:
    """Boolean array: IP is in the Tor exit list."""
    tor = tor_exit_ips()
    return pd.Series(ips, dtype="object").isin(tor).to_numpy()


def flag_hosting_vpn(asns: np.ndarray) -> np.ndarray:
    """Boolean array: ASN belongs to a known hosting/VPN provider."""
    return pd.Series(asns).isin(hosting_vpn_asns()).to_numpy()


def resolve_observation_geo(ds, geoip: GeoIPDatabase | None = None) -> int:
    """Fill empty country/ASN on ``ds.obs`` from ``src_ip``. Returns rows filled."""
    geoip = geoip or default_geoip()
    obs = ds.obs
    need = (obs["country"].to_numpy() == "") | (obs["asn"].to_numpy() == 0)
    if not need.any():
        return 0
    country, asn = geoip.lookup_many(obs.loc[need, "src_ip"].to_numpy())
    cur_c = obs["country"].to_numpy(dtype=object).copy()
    cur_a = obs["asn"].to_numpy().copy()
    empty_c = cur_c[need] == ""
    empty_a = cur_a[need] == 0
    cur_c[need] = np.where(empty_c, country, cur_c[need])
    cur_a[need] = np.where(empty_a, asn, cur_a[need])
    obs["country"] = cur_c
    obs["asn"] = cur_a
    return int(need.sum())
