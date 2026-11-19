#!/usr/bin/env python3
"""'Is anyone else using the machine?' gate. Prints a report and exits 1 on STOP.
Read-only: it only looks at nvidia-smi, the process list, memory, load average and local ports.
Works on Linux (NVIDIA) and macOS (Apple Silicon)."""
import argparse
import json
import os
import platform
import re
import sys
import urllib.request

import common
import protocol

INFERENCE_PROCS = ("vllm", "mlx_lm", "ollama", "llama-server", "lms ", "LM Studio", "sglang",
                   "trtllm")


def _secs(etime):            # [[dd-]hh:]mm:ss -> seconds
    d, _, rest = etime.rpartition("-")
    parts = [int(x) for x in rest.split(":")]
    while len(parts) < 3:
        parts.insert(0, 0)
    return (int(d) if d else 0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]


def _procs():
    """[(user, pid, pcpu, etime_s, args)] for every process, on Linux and macOS."""
    out = []
    for line in common.sh("ps -axo user=,pid=,pcpu=,etime=,args=", timeout=20).splitlines():
        f = line.split(None, 4)
        if len(f) < 5:
            continue
        try:
            out.append((f[0], int(f[1]), float(f[2]), _secs(f[3]), f[4]))
        except ValueError:
            continue
    return out


def _ollama_loaded(url="http://127.0.0.1:11434"):
    try:
        with urllib.request.urlopen(url + "/api/ps", timeout=3) as r:
            return json.loads(r.read()).get("models", [])
    except Exception:  # noqa: BLE001
        return None


def memory():
    """(total_bytes, available_bytes)."""
    if platform.system() == "Darwin":
        total = int(common.sh("sysctl -n hw.memsize") or 0)
        vm = common.sh("vm_stat")
        page = int((re.search(r"page size of (\d+)", vm) or [0, 16384])[1])
        free = sum(int(m) for m in re.findall(
            r"Pages (?:free|inactive|speculative|purgeable):\s+(\d+)", vm))
        return total, free * page
    info = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k, v = line.split(":", 1)
                info[k] = int(v.split()[0]) * 1024
    except OSError:
        pass
    return info.get("MemTotal", 0), info.get("MemAvailable", 0)


def check(cfg, max_load=2.0, check_load=True):
    me = os.environ.get("USER", "")
    problems, info = [], {}
    engine = (cfg.get("backend") or {}).get("type", "vllm")
    if platform.system() != "Darwin":
        apps = common.sh("nvidia-smi --query-compute-apps=pid,process_name,used_memory "
                         "--format=csv,noheader")
        info["gpu_compute_apps"] = []
        for line in filter(None, apps.splitlines()):
            pid = line.split(",")[0].strip()
            owner = common.sh(f"ps -o user= -p {pid}") or "?"
            info["gpu_compute_apps"].append({"line": line, "owner": owner})
            if owner != me:
                problems.append(f"GPU compute process not owned by {me}: {line} (owner {owner})")
        info["gpu"] = common.sh("nvidia-smi --query-gpu=utilization.gpu,power.draw,"
                                "temperature.gpu --format=csv,noheader")
    procs = _procs()
    # Only sustained work by another person counts: skip root (system tools such as a
    # momentary `nvidia-smi -q`) and anything that has been running for under a minute.
    busy = [p for p in procs if p[2] > 50 and p[0] not in (me, "root") and p[3] >= 60
            and not p[0].startswith("_")]
    info["busy_foreign_processes"] = [f"{u} {pid} {cpu}% {args[:120]}"
                                      for u, pid, cpu, _, args in busy]
    problems += [f"busy process from another user: {x}" for x in info["busy_foreign_processes"]]
    # An idle system Ollama service is fine; whether it has a model loaded is checked below.
    servers = [p for p in procs if p[0] != me and any(s in p[4] for s in INFERENCE_PROCS)
               and "ollama serve" not in p[4]]
    info["foreign_inference_servers"] = [f"{u} {pid} {args[:120]}" for u, pid, _, _, args in servers]
    problems += [f"inference server from another user: {x}"
                 for x in info["foreign_inference_servers"]]
    sysps = _ollama_loaded()
    info["ollama_11434_loaded"] = sysps
    if sysps:
        problems.append("an Ollama server (:11434) has a model loaded: "
                        + ", ".join(m.get("name", "?") for m in sysps))
    total, avail = memory()
    info["mem_total_gib"], info["mem_available_gib"] = round(total / 2**30, 1), \
        round(avail / 2**30, 1)
    if engine == "vllm" and total:
        need = total * protocol.VLLM_GPU_MEMORY_UTILIZATION
        if avail < need:
            problems.append(f"only {avail / 2**30:.0f} GiB of {total / 2**30:.0f} GiB free; vLLM "
                            f"needs {need / 2**30:.0f} GiB ({protocol.VLLM_GPU_MEMORY_UTILIZATION}"
                            f" of total) free to start")
    load1 = os.getloadavg()[0]
    info["loadavg"] = list(os.getloadavg())
    # Load average lags by a minute and includes our own previous run, so it only counts when
    # the CPU is busy right now with something that is not ours.
    foreign = sum(p[2] for p in procs if p[0] != me)
    info["foreign_cpu_pct_now"] = round(foreign, 1)
    if check_load and load1 > max_load and foreign > 100:
        problems.append(f"1-min load average {load1:.2f} > {max_load} with other users' "
                        f"processes using {foreign:.0f}% CPU")
    info["who"] = common.sh("who").splitlines()
    return {"ok": not problems, "problems": problems, "info": info}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(common.BENCH / "configs/TEMPLATE.json"))
    ap.add_argument("--out")
    a = ap.parse_args()
    rep = check(common.load_json(a.config))
    print(json.dumps(rep, indent=2))
    if a.out:
        common.write_json(a.out, rep)
    print("PRE-FLIGHT:", "PASS" if rep["ok"] else "STOP - someone else may be using the machine")
    sys.exit(0 if rep["ok"] else 1)
