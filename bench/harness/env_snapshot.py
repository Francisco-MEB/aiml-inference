"""Collects everything needed to reproduce a run into env.json (Linux/NVIDIA and macOS).
Server facts (engine version, KV cache, the exact command) are added per model by run_bench.py
under "servers", because each model gets its own server."""
import os
import platform
import re
from datetime import UTC, datetime

import common


def machine_facts():
    if platform.system() == "Darwin":
        mem = int(common.sh("sysctl -n hw.memsize") or 0)
        return {"os": f"macOS {platform.mac_ver()[0]}", "arch": platform.machine(),
                "cpu": common.sh("sysctl -n machdep.cpu.brand_string"),
                "hw_model": common.sh("sysctl -n hw.model"),
                "mem_total_gib": round(mem / 2**30, 1),
                "gpu": "Apple Silicon (unified memory)",
                "gpu_wired_limit_mb": common.sh("sysctl -n iogpu.wired_limit_mb")}
    smi = common.sh("nvidia-smi")
    m = re.search(r"Driver Version:\s*(\S+)\s+CUDA Version:\s*(\S+)", smi)
    osr = {}
    try:
        with open("/etc/os-release") as f:
            osr = dict(line.rstrip("\n").split("=", 1) for line in f if "=" in line)
    except OSError:
        pass
    mem_kb = 0
    try:
        with open("/proc/meminfo") as f:
            mem_kb = int(re.search(r"MemTotal:\s+(\d+)", f.read()).group(1))
    except (OSError, AttributeError):
        pass
    return {"os": osr.get("PRETTY_NAME", "").strip('"'), "kernel": platform.release(),
            "arch": platform.machine(),
            "cpu": common.sh("lscpu | grep 'Model name' | sed 's/.*: *//' | paste -sd+"),
            "mem_total_gib": round(mem_kb / 2**20, 1),
            "gpu": common.sh("nvidia-smi --query-gpu=name --format=csv,noheader"),
            "nvidia_driver": m.group(1) if m else "", "cuda_version": m.group(2) if m else ""}


def snapshot(cfg, run_id, overrides, preflight, models):
    return {
        "run_id": run_id,
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "hostname": platform.node().split(".")[0],
        "user": os.environ.get("USER", ""),
        **machine_facts(),
        "backend": cfg["backend"],
        "uv_version": common.sh("uv --version"),
        "models": models,
        "config_sha256": common.canonical_hash(cfg["_pinned"]),
        "config_path": cfg["_path"],
        "cli_overrides": overrides,
        "prompts_sha256": cfg["prompts"]["sha256"],
        "git_commit": common.sh(f"git -C {common.REPO} rev-parse HEAD"),
        "git_branch": common.sh(f"git -C {common.REPO} rev-parse --abbrev-ref HEAD"),
        "git_dirty": bool(common.sh(f"git -C {common.REPO} status --porcelain --untracked-files=no")),
        "python": platform.python_version(),
        "other_gpu_processes_at_start": preflight["info"].get("gpu_compute_apps", []),
        "preflight_ok": preflight["ok"],
        "preflight_problems": preflight["problems"],
        "loadavg_at_start": preflight["info"].get("loadavg"),
    }
