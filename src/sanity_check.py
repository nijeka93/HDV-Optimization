# Sanity check for the enumeration run: progress, speed, ETA
# Usage:
#   python sanity_check.py                # auto-detect CSV and use progress JSON
#   python sanity_check.py --csv enumerated_design_results.csv
#   python sanity_check.py --total 65536  # if no JSON, provide TOTAL to get %/ETA
#   python sanity_check.py --csv out.csv --total 131072
# You can also set env vars: SANITY_CSV, SANITY_TOTAL

import argparse
import json
import os
import time
from datetime import datetime
from typing import Optional

PROGRESS_JSON = "enumeration_progress.json"
# Auto-detect CSV: prefer partial if present; else main. Can be overridden via --csv or SANITY_CSV
DEFAULT_CSV_CANDIDATES = [
    "enumerated_design_results.csv",
]


def humanize_seconds(s: float) -> str:
    s = int(max(0, s))
    d, s = divmod(s, 86400)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    parts = []
    if d:
        parts.append(f"{d}d")
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    parts.append(f"{s}s")
    return " ".join(parts)


def autodetect_csv() -> Optional[str]:
    # env var takes precedence
    env_csv = os.environ.get("SANITY_CSV")
    if env_csv and os.path.exists(env_csv):
        return env_csv
    for p in DEFAULT_CSV_CANDIDATES:
        if os.path.exists(p):
            return p
    return None


def count_rows(csv_path: str) -> int:
    # Returns number of data rows (excludes header if present)
    if not os.path.exists(csv_path):
        return 0
    # Fast-ish line count
    with open(csv_path, "rb") as f:
        lines = sum(1 for _ in f)
    return max(0, lines - 1)  # minus header


def load_progress_json():
    if not os.path.exists(PROGRESS_JSON):
        return None
    try:
        with open(PROGRESS_JSON, "r") as f:
            return json.load(f)
    except Exception:
        return None


def main():
    parser = argparse.ArgumentParser(description="Check progress of full enumeration run")
    parser.add_argument("--csv", type=str, default=None, help="Path to results CSV to inspect")
    parser.add_argument("--total", type=int, default=None, help="Total number of individuals (for %/ETA if no JSON)")
    args = parser.parse_args()

    csv_path = args.csv or autodetect_csv()
    if not csv_path:
        print("No CSV found. Specify with --csv or set SANITY_CSV env var.")
        return

    total_env = os.environ.get("SANITY_TOTAL")
    total_override = args.total or (int(total_env) if total_env else None)

    data = load_progress_json()
    if data:
        current = int(data.get("next_index", 0))
        total = int(data.get("total", 0))
        started_at_epoch = data.get("started_at_epoch")
        started_at_iso = data.get("started_at_iso")
        last_update_iso = data.get("last_update_iso")
    else:
        current = None
        total = None
        started_at_epoch = None
        started_at_iso = None
        last_update_iso = None

    # Fallback: infer progress from CSV if JSON absent or current is smaller than rows
    rows_done = count_rows(csv_path)
    if current is None or rows_done > current:
        current = rows_done

    # If no JSON total, try override
    if not total:
        total = total_override

    # Print core stats
    print(f"CSV: {csv_path}")
    print(f"Rows written: {rows_done:,}")

    if total:
        pct = current / total * 100.0
        print(f"Progress: {current:,} / {total:,} ({pct:.2f}%)")
    else:
        print("Progress: unknown total (provide --total or SANITY_TOTAL if you want %/ETA)")

    # File size
    size_mb = os.path.getsize(csv_path) / (1024 * 1024)
    print(f"CSV size: {size_mb:.1f} MB")

    # Timing + ETA
    if started_at_epoch:
        elapsed = time.time() - float(started_at_epoch)
        print(f"Started: {started_at_iso} | Last update: {last_update_iso}")
        print(f"Elapsed: {humanize_seconds(elapsed)}")
        if total and current > 0:
            rate = current / elapsed
            eta_sec = (total - current) / rate if rate > 0 else None
            eta_str = humanize_seconds(eta_sec) if eta_sec is not None else "n/a"
            print(f"Speed: {rate:.1f} rows/s | ETA: {eta_str}")
    else:
        # If we don't have JSON timing, we can still give an ETA if user provided an average rate
        rate_env = os.environ.get("SANITY_RATE")  # rows per second
        if total and rate_env:
            try:
                rate = float(rate_env)
                rem = max(0, total - current)
                eta_str = humanize_seconds(rem / rate) if rate > 0 else "n/a"
                print(f"ETA (using SANITY_RATE={rate_env} rows/s): {eta_str}")
            except Exception:
                pass


if __name__ == "__main__":
    main()

# One-liner (snapshot without saving a file):
# python - <<'PY'
# import json, os, time
# PROGRESS_JSON = "enumeration_progress.json"; CANDIDATES=[
#     "enumerated_design_results_partial.csv",
#     "enumerated_design_results_candidates.csv",
#     "enumerated_design_results.csv",
# ]
# def H(s):
#     s=int(max(0,s)); d,s=divmod(s,86400); h,s=divmod(s,3600); m,s=divmod(s,60);
#     return (f"{d}d " if d else "")+(f"{h}h " if h else "")+(f"{m}m " if m else "")+f"{s}s"
# csv_path = next((p for p in CANDIDATES if os.path.exists(p)), None)
# if not csv_path: print("No CSV found"); raise SystemExit
# rows = max(0, sum(1 for _ in open(csv_path,'rb'))-1)
# print("CSV:", csv_path, "| rows:", rows)
# if os.path.exists(PROGRESS_JSON):
#     d=json.load(open(PROGRESS_JSON)); cur,tot=d.get('next_index',rows),d.get('total',None); st=d.get('started_at_epoch')
#     if tot: print(f"Progress: {cur:,}/{tot:,} ({cur/tot*100:.2f}%)")
#     if st and tot: el=time.time()-float(st); rate=cur/el if el>0 else 0; eta=(tot-cur)/rate if rate>0 else None; print('Elapsed:',H(el),'| Rate:',f"{rate:.1f}",'rows/s | ETA:',H(eta) if eta else 'n/a')
# PY
