#!/usr/bin/env python3
"""Samples GPU telemetry once a second into a CSV until it receives SIGTERM/SIGINT.
t_unix is the same clock (time.time) that the benchmark CSV uses, so rows can be joined.
wall_meter_w is left empty: fill it from the wall meter log (see make_report.py --wall-csv)."""
import argparse
import csv
import shutil
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime

FIELDS = ["power.draw", "power.draw.instant", "power.draw.average", "utilization.gpu",
          "temperature.gpu", "clocks.sm"]
COLS = ["t_unix", "t_iso", "gpu_power_w", "gpu_power_instant_w", "gpu_power_avg_w",
        "gpu_util_pct", "gpu_temp_c", "gpu_clock_mhz", "wall_meter_w"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--interval", type=float, default=1.0)
    a = ap.parse_args()
    ms = max(100, int(a.interval * 1000))
    if not shutil.which("nvidia-smi"):
        # No NVIDIA GPU (e.g. a Mac): header only, and wait to be stopped like the real logger.
        with open(a.out, "w", newline="") as f:
            csv.writer(f).writerow(COLS)
        signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
        signal.signal(signal.SIGINT, lambda *_: sys.exit(0))
        while True:
            time.sleep(3600)
    p = subprocess.Popen(["nvidia-smi", f"--query-gpu={','.join(FIELDS)}",
                          "--format=csv,noheader,nounits", "-lms", str(ms)],
                         stdout=subprocess.PIPE, text=True)
    signal.signal(signal.SIGTERM, lambda *_: p.terminate())
    signal.signal(signal.SIGINT, lambda *_: p.terminate())
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for line in p.stdout:
            t = time.time()
            vals = [v.strip() for v in line.split(",")]
            if len(vals) != len(FIELDS):
                continue
            vals = ["" if v.startswith("[") or v == "N/A" else v for v in vals]
            w.writerow([f"{t:.3f}",
                        datetime.fromtimestamp(t, UTC).isoformat(timespec="milliseconds"),
                        *vals, ""])
            f.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
