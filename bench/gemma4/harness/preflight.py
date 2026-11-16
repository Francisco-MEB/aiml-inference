#!/usr/bin/env python3
"""'Is anyone else using the machine?' gate. Prints a report and exits 1 on STOP.
Read-only: it only looks at nvidia-smi, the process list, load average and local ports."""
import argparse
import json
import os
import sys

import common


def check(cfg, max_load=2.0, check_load=True):
    me = os.environ.get("USER", "")
    problems, info = [], {}
    apps = common.sh("nvidia-smi --query-compute-apps=pid,process_name,used_memory "
                     "--format=csv,noheader")
    info["gpu_compute_apps"] = []
    for line in filter(None, apps.splitlines()):
        pid = line.split(",")[0].strip()
        owner = common.sh(f"ps -o user= -p {pid}") or "?"
        info["gpu_compute_apps"].append({"line": line, "owner": owner})
        if owner != me:
            problems.append(f"GPU compute process not owned by {me}: {line} (owner {owner})")
    # Any Ollama runner or other inference server belonging to someone else
    procs = common.sh("ps -eo user:20,pid,pcpu,etime,args --sort=-pcpu | head -40")
    def secs(etime):            # [[dd-]hh:]mm:ss -> seconds
        d, _, rest = etime.rpartition("-")
        parts = [int(x) for x in rest.split(":")]
        while len(parts) < 3:
            parts.insert(0, 0)
        return (int(d) if d else 0) * 86400 + parts[0] * 3600 + parts[1] * 60 + parts[2]

    # Only sustained work by another person counts: skip root (system tools such as a
    # momentary `nvidia-smi -q`) and anything that has been running for under a minute.
    busy = [p for p in procs.splitlines()[1:]
            if float(p.split()[2]) > 50 and p.split()[0] not in (me, "root")
            and secs(p.split()[3]) >= 60]
    info["busy_foreign_processes"] = busy
    problems += [f"busy process from another user: {p[:140]}" for p in busy]
    sysps = common.loaded_models("http://127.0.0.1:11434")
    info["system_ollama_11434_loaded"] = sysps
    if sysps:
        problems.append("system Ollama (:11434) has a model loaded: "
                        + ", ".join(m.get("name", "?") for m in sysps))
    load1 = os.getloadavg()[0]
    info["loadavg"] = list(os.getloadavg())
    # Load average lags by a minute and includes our own previous run, so it only counts when
    # the CPU is busy right now with something that is not ours.
    cpu = common.sh("ps -eo user:20,pcpu --no-headers")
    foreign = sum(float(x.split()[1]) for x in cpu.splitlines() if x.split()[0] != me)
    info["foreign_cpu_pct_now"] = round(foreign, 1)
    if check_load and load1 > max_load and foreign > 100:
        problems.append(f"1-min load average {load1:.2f} > {max_load} with other users' "
                        f"processes using {foreign:.0f}% CPU")
    info["who"] = common.sh("who").splitlines()
    info["docker_ps"] = common.sh("docker ps --format '{{.Names}} {{.Image}}' 2>&1") \
        or "(no access or none)"
    info["container_shims"] = common.sh("ps -eo user,pid,comm | grep containerd-shim").splitlines()
    info["mem"] = common.sh("free -g | sed -n 2p")
    info["gpu"] = common.sh("nvidia-smi --query-gpu=utilization.gpu,power.draw,temperature.gpu "
                            "--format=csv,noheader")
    return {"ok": not problems, "problems": problems, "info": info}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(common.BENCH / "config/gemma4_ollama_native.json"))
    ap.add_argument("--out")
    a = ap.parse_args()
    rep = check(common.load_json(a.config))
    print(json.dumps(rep, indent=2))
    if a.out:
        common.write_json(a.out, rep)
    print("PRE-FLIGHT:", "PASS" if rep["ok"] else "STOP - someone else may be using the machine")
    sys.exit(0 if rep["ok"] else 1)
