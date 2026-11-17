#!/usr/bin/env python3
"""
sample_node.py

Answers "when the server saturates, what is the bottleneck: GPU, Python (CPU)
or memory?" Run it on the server node while scripts/openai_capacity_probe.py
generates load. Stdlib only; Linux (reads /proc).

Every --interval seconds it writes one CSV row with:
  gpu_busy_pct        nvidia-smi utilization.gpu: % of the interval in which
                      at least one kernel ran. Idle gaps = GPU waiting on CPU.
  gpu_watts, sm_mhz   power draw and SM clock
  <group>_cpu_pct     CPU of the vLLM processes, summed over their threads
  <group>_main_pct    CPU of each group's busiest *main* thread. Python runs
                      the scheduler/API loop on one thread, so ~100% here
                      while the GPU idles means Python is the bottleneck.
  running, waiting    vLLM requests in the batch / queued (/metrics)
  kv_cache_pct        vLLM KV-cache usage; ~100% with requests waiting means
                      memory, not compute, limits concurrency
  gen_tps             server-side generated tokens/s (delta of the counter)

Groups: engine (VLLM::EngineCore), workers (VLLM::Worker*), api (the
`vllm serve` front-end process).

Example (Ctrl-C to stop):
  python3 scripts/sample_node.py --out docs/capacity/data/35b-a3b_chat_1node_node.csv
"""
import argparse
import csv
import os
import re
import subprocess
import time
import urllib.request
from datetime import datetime, timezone

CLK = os.sysconf("SC_CLK_TCK")
GROUPS = ("engine", "workers", "api")
METRICS = {  # column -> metric names (newer vLLM first)
    "running": ("vllm:num_requests_running",),
    "waiting": ("vllm:num_requests_waiting",),
    "kv_cache_pct": ("vllm:kv_cache_usage_perc", "vllm:gpu_cache_usage_perc"),
    "gen_tokens_total": ("vllm:generation_tokens_total",),
}


def read(path):
    try:
        with open(path, "rb") as f:
            return f.read().decode(errors="replace")
    except OSError:
        return ""


def group_of(pid):
    comm = read(f"/proc/{pid}/comm").strip()
    if comm.startswith("VLLM::EngineCor"):
        return "engine"
    if comm.startswith("VLLM::Worker"):
        return "workers"
    cmd = read(f"/proc/{pid}/cmdline").replace("\0", " ")
    if re.search(r"\bvllm(\s+|\.entrypoints\S*\s+)serve\b", cmd) or "vllm.entrypoints.openai" in cmd:
        return "api"
    return None


def cpu_ticks(stat):
    # fields after the ")" of comm: state is [0], utime [11], stime [12]
    rest = stat.rsplit(")", 1)[-1].split()
    return int(rest[11]) + int(rest[12]) if len(rest) > 12 else 0


def snapshot(owner):
    """{pid: (group, process_ticks, main_thread_ticks)} for vLLM processes of `owner`."""
    snap = {}
    for d in os.listdir("/proc"):
        if not d.isdigit():
            continue
        pid = int(d)
        try:
            if owner is not None and os.stat(f"/proc/{pid}").st_uid != owner:
                continue
        except OSError:
            continue
        g = group_of(pid)
        if g:
            snap[pid] = (g, cpu_ticks(read(f"/proc/{pid}/stat")),
                         cpu_ticks(read(f"/proc/{pid}/task/{pid}/stat")))
    return snap


def gpu():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,power.draw,clocks.sm",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.splitlines()[0]
        return [v.strip() for v in out.split(",")]
    except Exception:
        return ["", "", ""]


def scrape(url):
    vals = {}
    if not url:
        return vals
    try:
        with urllib.request.urlopen(url, timeout=3) as r:
            text = r.read().decode(errors="replace")
    except Exception:
        return vals
    for col, names in METRICS.items():
        for name in names:
            total, found = 0.0, False
            for line in text.splitlines():
                if line.startswith(name) and (len(line) == len(name) or line[len(name)] in " {"):
                    try:
                        total += float(line.rsplit(" ", 1)[1])
                        found = True
                    except ValueError:
                        pass
            if found:
                vals[col] = total
                break
    return vals


def main():
    ap = argparse.ArgumentParser(description="Sample GPU, vLLM CPU and vLLM queue metrics.")
    ap.add_argument("--metrics-url", default="http://127.0.0.1:8000/metrics",
                    help="vLLM Prometheus endpoint ('' to skip)")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--duration", type=float, default=0, help="seconds; 0 = until Ctrl-C")
    ap.add_argument("--all-users", action="store_true",
                    help="count vLLM processes of every user (default: only yours)")
    ap.add_argument("--out", default="node_samples.csv")
    args = ap.parse_args()
    owner = None if args.all_users else os.getuid()

    cols = (["timestamp_utc", "gpu_busy_pct", "gpu_watts", "sm_mhz"]
            + [f"{g}_{k}" for g in GROUPS for k in ("cpu_pct", "main_pct")]
            + ["running", "waiting", "kv_cache_pct", "gen_tps"])
    prev, prev_t = snapshot(owner), time.monotonic()
    prev_gen = scrape(args.metrics_url).get("gen_tokens_total")
    if not prev:
        print("# warning: no vLLM processes found yet (owned by you; --all-users to widen)")
    end = time.monotonic() + args.duration if args.duration else None
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        print("  gpu% | engine cpu/main% | api cpu/main% | run wait  kv% | gen tok/s")
        try:
            while end is None or time.monotonic() < end:
                time.sleep(args.interval)
                busy, watts, mhz = gpu()
                cur, now = snapshot(owner), time.monotonic()
                dt = now - prev_t
                row = {"timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       "gpu_busy_pct": busy, "gpu_watts": watts, "sm_mhz": mhz}
                for g in GROUPS:
                    procs = [(p, cur[p], prev[p]) for p in cur if p in prev and cur[p][0] == g]
                    tot = sum(c[1] - o[1] for _, c, o in procs) / CLK / dt * 100
                    main = max((c[2] - o[2] for _, c, o in procs), default=0) / CLK / dt * 100
                    row[f"{g}_cpu_pct"] = round(tot, 1) if procs else ""
                    row[f"{g}_main_pct"] = round(main, 1) if procs else ""
                m = scrape(args.metrics_url)
                for k in ("running", "waiting"):
                    row[k] = int(m[k]) if k in m else ""
                if "kv_cache_pct" in m:
                    kv = m["kv_cache_pct"]
                    row["kv_cache_pct"] = round(kv * 100 if kv <= 1 else kv, 1)
                gen = m.get("gen_tokens_total")
                if gen is not None and prev_gen is not None:
                    row["gen_tps"] = round((gen - prev_gen) / dt, 1)
                prev_gen = gen if gen is not None else prev_gen
                prev, prev_t = cur, now
                w.writerow(row)
                f.flush()
                print(f"{row['gpu_busy_pct']:>6} | {row['engine_cpu_pct']!s:>7}/{row['engine_main_pct']!s:<7}"
                      f" | {row['api_cpu_pct']!s:>6}/{row['api_main_pct']!s:<6}"
                      f" | {row['running']!s:>3} {row['waiting']!s:>4} {row.get('kv_cache_pct', '')!s:>4}"
                      f" | {row.get('gen_tps', '')}")
        except KeyboardInterrupt:
            pass
    print(f"# samples written to {args.out}")


if __name__ == "__main__":
    main()
