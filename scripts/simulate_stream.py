#!/usr/bin/env python3
"""CLI utility to simulate live transaction injection into ChainTrace.

Can be run alongside the web UI during live presentations to inject
transactions one-by-one or in an automated stream.

Usage:
    python scripts/simulate_stream.py --status         # Check current simulation state
    python scripts/simulate_stream.py --step           # Inject next transaction
    python scripts/simulate_stream.py --auto 2.5       # Stream next transactions every 2.5s
    python scripts/simulate_stream.py --reset          # Reset scenario to clean baseline
    python scripts/simulate_stream.py --full           # Fast-forward to full scenario
"""

import argparse
import http.client
import json
import sys
import time
from urllib.parse import urlparse


DEFAULT_URL = "http://127.0.0.1:8000"


def make_request(base_url: str, path: str, method: str = "GET", payload: dict | None = None) -> dict:
    parsed = urlparse(base_url)
    host = parsed.hostname or "127.0.0.1"
    port = parsed.port or 8000

    headers = {"Accept": "application/json"}
    body_bytes = None
    if payload is not None:
        body_bytes = json.dumps(payload).encode("utf-8")
        headers["Content-Type"] = "application/json"

    try:
        conn = http.client.HTTPConnection(host, port, timeout=5)
        conn.request(method, path, body=body_bytes, headers=headers)
        resp = conn.getresponse()
        resp_data = resp.read().decode("utf-8")
        conn.close()
        return json.loads(resp_data)
    except Exception as e:
        print(f"\033[91mError connecting to ChainTrace on {host}:{port}: {e}\033[0m")
        print("Please ensure the backend server is running: ./run_demo.sh")
        sys.exit(1)


def print_status(status: dict) -> None:
    step = status.get("current_step", 0)
    total = status.get("total_steps", 6)
    print("\n" + "=" * 68)
    print("  \033[1;36mChainTrace Live Ingestion Simulation Status\033[0m")
    print("=" * 68)
    print(f"  Step Progress:      \033[1;32m{step} / {total}\033[0m ({status.get('step_title', '')})")
    print(f"  Headline:           {status.get('step_headline', '')}")
    print(f"  Active Tx Count:    \033[1;33m{status.get('total_transactions', 0)}\033[0m")
    print(f"  Active Alerts:      \033[1;31m{status.get('total_alerts', 0)}\033[0m")
    print(f"  Can Inject Next:    {'Yes' if status.get('can_inject_next') else 'No (Scenario Complete)'}")
    if status.get("next_txid"):
        print(f"  Next Up:            \033[1;35mTX {status.get('next_txid')}\033[0m — {status.get('next_step_title')}")
    print("=" * 68 + "\n")


def inject_step(base_url: str) -> None:
    res = make_request(base_url, "/simulation/inject", method="POST")
    if not res.get("success"):
        print(f"\n\033[93m⚠ {res.get('message', 'Cannot inject next step')}\033[0m\n")
        return

    inj = res.get("injected", {})
    st = res.get("status", {})
    print("\n" + "─" * 68)
    print(f"  \033[1;32m⚡ INGESTION EVENT: Step {inj.get('step')} — TX {inj.get('txid')}\033[0m")
    print("─" * 68)
    print(f"  Title:              {inj.get('title')}")
    print(f"  Volume:             \033[1;33m{inj.get('amount_btc')} BTC\033[0m (Label: {inj.get('label')})")
    print(f"  Narrative:          {inj.get('narrative')}")
    if inj.get("added_alerts"):
        print(f"  \033[1;31mAlerts Generated:   {len(inj.get('added_alerts'))} new alert(s)\033[0m ({', '.join(inj.get('added_alerts'))})")
    print(f"  Graph Updates:      +{inj.get('edges_added_count')} edges linked into forensic topology")
    print(f"  Total Ingested:     {st.get('total_transactions')} txs, {st.get('total_alerts')} alerts")
    print("─" * 68 + "\n")


def inject_batch(base_url: str, batch_size: int = 10) -> None:
    res = make_request(base_url, f"/simulation/inject?batch={batch_size}", method="POST")
    if not res.get("success"):
        print(f"\n\033[93m⚠ {res.get('message', 'Cannot inject batch')}\033[0m\n")
        return

    st = res.get("status", {})
    count = res.get("batch_count", 0)
    print("\n" + "─" * 68)
    print(f"  \033[1;32m⚡ BATCH INGESTION EVENT: +{count} Transactions Ingested\033[0m")
    print("─" * 68)
    print(f"  Mempool Progress:   Step {st.get('current_step')} / {st.get('total_steps')}")
    print(f"  Total Ingested:     \033[1;33m{st.get('total_transactions')} txs\033[0m, \033[1;31m{st.get('total_alerts')} alerts\033[0m")
    print(f"  Node Height:        Block #{st.get('block_height'):,}")
    print("─" * 68 + "\n")


def reset_scenario(base_url: str, mode: str = "baseline") -> None:
    res = make_request(base_url, "/simulation/reset", method="POST", payload={"mode": mode})
    print(f"\n\033[1;32m✓ Scenario reset to: {mode.upper()}\033[0m")
    print_status(res)


def auto_stream(base_url: str, interval: float) -> None:
    print(f"\n\033[1;36mStarting automated mempool transaction stream (interval: {interval}s)...\033[0m")
    print("Press Ctrl+C to pause stream at any time.\n")
    try:
        while True:
            st = make_request(base_url, "/simulation/status")
            if not st.get("can_inject_next"):
                print("\033[1;32m✓ Scenario stream completed! All transactions injected.\033[0m\n")
                break
            inject_step(base_url)
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\033[93m⏸ Stream paused by user.\033[0m\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="ChainTrace Live Ingestion Simulator")
    parser.add_argument("--url", default=DEFAULT_URL, help="Base API URL")
    parser.add_argument("--status", action="store_true", help="Print current status")
    parser.add_argument("--step", action="store_true", help="Inject next transaction")
    parser.add_argument("--batch", type=int, metavar="N", help="Inject a batch of N transactions")
    parser.add_argument("--auto", type=float, metavar="SECS", help="Auto stream next transactions every SECS seconds")
    parser.add_argument("--reset", action="store_true", help="Reset scenario to clean baseline")
    parser.add_argument("--full", action="store_true", help="Fast-forward to full scenario")

    args = parser.parse_args()

    if args.reset:
        reset_scenario(args.url, mode="baseline")
    elif args.full:
        reset_scenario(args.url, mode="full")
    elif args.batch:
        inject_batch(args.url, batch_size=args.batch)
    elif args.step:
        inject_step(args.url)
    elif args.auto:
        auto_stream(args.url, args.auto)
    else:
        st = make_request(args.url, "/simulation/status")
        print_status(st)


if __name__ == "__main__":
    main()
