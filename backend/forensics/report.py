"""Printable case report (standalone HTML) with chain-of-custody hashing.

The report embeds the full evidence package as JSON and records its SHA-256
digest, the dataset fingerprint, the model version and the configuration.
``verify_report`` recomputes the digest from the embedded JSON, so a report
can be checked offline for tampering.  Save as PDF from any browser
(print stylesheet included).
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import re
import sys
from datetime import UTC, datetime
from pathlib import Path

from backend.forensics.cases import dataset_fingerprint
from backend.forensics.evidence import build_entity_report
from backend.forensics.graphview import entity_subgraph

TOOL_VERSION = "chaintrace-forensics/0.2"
_JSON_RE = re.compile(r'<script type="application/json" id="evidence">(.*?)</script>', re.S)


def canonical_json(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def build_evidence_package(
    res,
    entity_ids: list[int],
    title: str,
    analyst: str,
    notes: str = "",
    now: datetime | None = None,
) -> dict:
    """Everything the report states, as one JSON-serialisable object."""
    now = now or datetime.now(UTC)
    entities = [build_entity_report(res, int(e)) for e in entity_ids]
    cfg = {k: getattr(res.cfg, k) for k in res.cfg.__dataclass_fields__}
    return {
        "title": title,
        "analyst": analyst,
        "notes": notes,
        "generated_utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "tool": TOOL_VERSION,
        "dataset": {
            "source": res.ds.report.source if res.ds.report else "",
            "summary": res.ds.summary(),
            "fingerprint_sha256": dataset_fingerprint(res.ds),
            "synthetic_data": res.ds.truth is not None,
        },
        "model": {
            "version": res.model_version,
            "label_source": res.label_source,
            "config": cfg,
            "config_sha256": sha256_hex(canonical_json(cfg)),
        },
        "entities": entities,
    }


def _esc(x) -> str:
    return html.escape(str(x))


def _svg_graph(sub: dict, width: int = 560, height: int = 300) -> str:
    nodes = sub["nodes"][:24]
    ids = {n["id"] for n in nodes}
    cx, cy, r = width / 2, height / 2, min(width, height) / 2 - 34
    pos = {}
    others = [n for n in nodes if n["id"] != sub["center"]]
    pos[sub["center"]] = (cx, cy)
    for i, n in enumerate(others):
        a = 2 * math.pi * i / max(len(others), 1)
        pos[n["id"]] = (cx + r * math.cos(a), cy + r * math.sin(a))
    col = {"critical": "#b42318", "high": "#dc6803", "medium": "#b54708", "low": "#667085"}
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="Link graph">'
    ]
    for e in sub["edges"]:
        if e["src"] in ids and e["dst"] in ids:
            (x1, y1), (x2, y2) = pos[e["src"]], pos[e["dst"]]
            parts.append(
                f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" stroke="#98a2b3" stroke-width="1"/>'
            )
    for n in nodes:
        x, y = pos[n["id"]]
        big = n["id"] == sub["center"]
        parts.append(
            f'<circle cx="{x:.0f}" cy="{y:.0f}" r="{11 if big else 7}" fill="{col.get(n["tier"], "#667085")}" '
            f'stroke="#fff" stroke-width="1.5"/><text x="{x:.0f}" y="{y + (22 if big else 18):.0f}" '
            f'font-size="9" text-anchor="middle" fill="#344054">{_esc(n["label"])}</text>'
        )
    parts.append("</svg>")
    return "".join(parts)


def _entity_section(res, e: dict) -> str:
    sc = e["score"]
    rows = "".join(
        f"<tr><td>{_esc(c['description'])}</td><td>{_esc(c['value'])}</td><td>{_esc(c['typical_value'])}</td>"
        f"<td><div class='bar {'pos' if c['contribution'] > 0 else 'neg'}' style='width:{min(abs(c['contribution']) * 300, 100):.0f}%'></div>"
        f"{c['contribution'] * 100:+.0f} pts</td></tr>"
        for c in e["model_contributions"]
    )
    heur = (
        "".join(
            f"<li><b>{_esc(h['label'])}</b> <span class='tag'>{_esc(h['type'])}</span> - {_esc(h['detail'])}"
            f"<br><code>{_esc(', '.join(h['txids'][:3]))}</code></li>"
            for h in e["heuristic_evidence"]
        )
        or "<li>No structural laundering patterns detected.</li>"
    )
    net = e["network_evidence"]
    origins = "".join(
        f"<tr><td>{_esc(o['ip'])}</td><td>{o['count']}</td><td>{_esc(o['country'])}</td><td>AS{o['asn']}</td>"
        f"<td>{'Tor exit' if o['is_tor_exit'] else ('hosting/VPN' if o['is_hosting_or_vpn'] else '')}</td></tr>"
        for o in net.get("top_origins", [])
    )
    link = net.get("wallet_ip_link")
    link_txt = (
        (
            f"Top origin IP {_esc(link['ip'])} carried {link['share']:.0%} of sends "
            f"(95% CI {link['ci95'][0]:.0%}-{link['ci95'][1]:.0%}). {_esc(link['interpretation'])}"
        )
        if link
        else ""
    )
    txs = "".join(
        f"<tr><td><code>{_esc(t['txid'][:20])}...</code></td><td>{_esc(t['time'])}</td><td>{t['role']}</td>"
        f"<td>{t['amount_btc']}</td><td>{t['n_in']}/{t['n_out']}</td></tr>"
        for t in e["key_transactions"]
    )
    sub = entity_subgraph(res, e["entity_id"], radius=1, max_nodes=16, max_events=0)
    trail = sub.get("trail_forward")
    trail_txt = (
        (" &rarr; ".join(f"{_esc(s['label'])}" for s in trail))
        if trail
        else "no onward trail found"
    )
    return f"""
<section class="entity">
<h2>{_esc(e["label"])} <span class="tier {sc["tier"]}">{_esc(sc["tier"]).upper()}</span></h2>
<p class="summary">{_esc(e["summary"])}</p>
<table class="kv"><tr><th>Priority score</th><td>{sc["final"]:.1f} / 100</td><th>Rank</th><td>#{sc["rank"]} of {sc["of"]:,}</td></tr>
<tr><th>Illicit probability</th><td>{sc["illicit_probability"]:.1%} (90% range {sc["interval90"][0]:.1%}-{sc["interval90"][1]:.1%})</td>
<th>Anomaly percentile</th><td>{sc["anomaly_percentile"]:.1%}</td></tr>
<tr><th>Addresses</th><td>{e["n_addresses"]}</td><th>Active</th><td>{_esc(e["first_seen"])} to {_esc(e["last_seen"])}</td></tr></table>
<h3>Structural evidence</h3><ul>{heur}</ul>
<h3>Why the model scored it this way</h3>
<table><tr><th>Factor</th><th>This entity</th><th>Typical</th><th>Effect</th></tr>{rows}</table>
<h3>Network evidence</h3>
<table><tr><th>Origin IP</th><th>Sends</th><th>Country</th><th>ASN</th><th>Flag</th></tr>{origins}</table>
<p class="small">{link_txt}</p>
<h3>Key transactions</h3>
<table><tr><th>TXID</th><th>Time (UTC)</th><th>Role</th><th>BTC</th><th>in/out</th></tr>{txs}</table>
<h3>Link graph (1 hop)</h3>{_svg_graph(sub)}
<p class="small">Onward money trail: {trail_txt}</p>
<p class="small">Addresses (first 10): <code>{_esc(", ".join(e["addresses"]))}</code></p>
</section>"""


CSS = """
body{font:14px/1.5 system-ui,Segoe UI,Arial,sans-serif;color:#101828;max-width:900px;margin:24px auto;padding:0 16px}
h1{font-size:22px;margin:0}h2{font-size:18px;margin:24px 0 6px;border-bottom:2px solid #101828;padding-bottom:4px}
h3{font-size:14px;margin:16px 0 4px;color:#344054}table{border-collapse:collapse;width:100%;margin:6px 0}
th,td{border:1px solid #d0d5dd;padding:4px 8px;text-align:left;font-size:12px;vertical-align:top}th{background:#f2f4f7}
code{font-size:11px;word-break:break-all}.tag{font-size:10px;background:#eaecf0;border-radius:4px;padding:1px 5px}
.tier{font-size:11px;padding:2px 8px;border-radius:10px;color:#fff;vertical-align:middle}
.critical{background:#b42318}.high{background:#dc6803}.medium{background:#b54708}.low{background:#667085}
.bar{height:6px;border-radius:3px;margin-bottom:2px}.bar.pos{background:#b42318}.bar.neg{background:#12b76a}
.summary{background:#f9fafb;border-left:4px solid #101828;padding:8px 12px}.small{font-size:12px;color:#475467}
.custody{border:1px solid #101828;padding:10px 14px;margin:16px 0;background:#fcfcfd}.custody code{display:block}
.warn{border:1px solid #b54708;background:#fffaeb;padding:8px 12px;font-size:12px;margin:12px 0}
.entity{page-break-inside:avoid}@media print{body{margin:0}.entity{page-break-before:always}}
"""


def render_report_html(
    res,
    entity_ids: list[int],
    title: str = "ChainTrace investigation report",
    analyst: str = "",
    notes: str = "",
) -> str:
    """Render the standalone HTML report for ``entity_ids``."""
    pkg = build_evidence_package(res, entity_ids, title, analyst, notes)
    blob = canonical_json(pkg)
    digest = sha256_hex(blob)
    body = "".join(_entity_section(res, e) for e in pkg["entities"])
    synthetic = ""
    if pkg["dataset"]["synthetic_data"]:
        synthetic = "<div class='warn'><b>Synthetic data.</b> All addresses, IPs and transactions in this report are simulated.</div>"
    safe_blob = blob.replace("</", "<\\/")
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><title>{_esc(title)}</title>
<style>{CSS}</style></head><body>
<h1>{_esc(title)}</h1>
<p class="small">Generated {pkg["generated_utc"]} &middot; Analyst: {_esc(analyst) or "unspecified"} &middot; {_esc(TOOL_VERSION)}</p>
{synthetic}
<div class="warn"><b>Triage evidence, not proof.</b> Scores rank leads for human review. Heuristics are inferences; probabilities are model output;
IP correlations are not attribution. Do not act against any person on this report alone.</div>
<p>{_esc(notes)}</p>
<div class="custody"><b>Chain of custody</b>
<code>Evidence SHA-256: {digest}</code>
<code>Dataset fingerprint SHA-256: {pkg["dataset"]["fingerprint_sha256"]}</code>
<code>Model: {_esc(res.model_version)} (labels: {_esc(res.label_source)}) &middot; config SHA-256: {pkg["model"]["config_sha256"]}</code>
<span class="small">Verify offline: <code>python -m backend.forensics.report verify report.html</code></span></div>
{body}
<script type="application/json" id="evidence">{safe_blob}</script>
</body></html>"""


def verify_report(html_text: str) -> tuple[bool, str, str]:
    """Recompute the evidence digest from the embedded JSON.

    Returns (ok, recorded_digest, recomputed_digest).
    """
    m = _JSON_RE.search(html_text)
    if not m:
        return False, "", ""
    blob = m.group(1).replace("<\\/", "</")
    recomputed = sha256_hex(canonical_json(json.loads(blob)))
    rec = re.search(r"Evidence SHA-256: ([0-9a-f]{64})", html_text)
    recorded = rec.group(1) if rec else ""
    return recorded == recomputed, recorded, recomputed


if __name__ == "__main__":
    if len(sys.argv) == 3 and sys.argv[1] == "verify":
        ok, a, b = verify_report(Path(sys.argv[2]).read_text(encoding="utf-8"))
        print(
            "OK: report matches its recorded digest"
            if ok
            else f"MISMATCH\nrecorded:   {a}\nrecomputed: {b}"
        )
        sys.exit(0 if ok else 1)
    print("usage: python -m backend.forensics.report verify <report.html>")
    sys.exit(2)
