"""Collects everything needed to reproduce a run into env.json."""
import os
import platform
import re
from datetime import UTC, datetime

import common


def server_binary_info(cfg):
    mode = cfg["server"]["mode"]
    if mode == "native":
        path = cfg["server"].get("_binary_path", "")
        info = {"mode": "native", "binary_path": path}
        if path and os.path.isfile(path):
            info["binary_sha256"] = common.sha256_file(path)
        info["release_asset_sha256_pinned"] = cfg["server"]["native"]["asset_sha256"]
        info["docker_image_digest"] = "not used (native mode); pinned for Docker runs: " + \
            cfg["server"]["docker"]["arm64_digest"]
        return info
    if mode == "docker":
        name = f"ollama-bench-{os.environ.get('USER', '')}-{cfg['server']['port']}"
        return {"mode": "docker", "image": cfg["server"]["docker"]["image"],
                "docker_image_digest": common.sh(
                    f"docker inspect --format '{{{{.Image}}}}' {name}")}
    return {"mode": mode}


def server_logged_config(log_path):
    """Ollama prints its full effective settings at startup; keep that line verbatim."""
    try:
        with open(log_path, errors="replace") as f:
            for line in f:
                if "server config" in line:
                    m = re.search(r'env="map\[(.*)\]"', line)
                    return m.group(1) if m else line.strip()
    except OSError:
        pass
    return ""


def snapshot(cfg, run_id, overrides, preflight, models, server_log, server_env):
    smi = common.sh("nvidia-smi")
    m = re.search(r"Driver Version:\s*(\S+)\s+CUDA Version:\s*(\S+)", smi)
    with open("/etc/os-release") as f:
        osr = dict(line.split("=", 1) for line in f if "=" in line)
    native = cfg["backend"].get("native_url")
    version = ""
    if native:
        try:
            version = common.http_json(native + "/api/version").get("version", "")
        except Exception:  # noqa: BLE001
            version = "unreachable"
    return {
        "run_id": run_id,
        "timestamp_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "hostname": platform.node(),
        "user": os.environ.get("USER", ""),
        "os": osr.get("PRETTY_NAME", "").strip().strip('"'),
        "kernel": platform.release(),
        "arch": platform.machine(),
        "cpu": common.sh("lscpu | grep 'Model name' | sed 's/.*: *//' | paste -sd+"),
        "mem_total_gib": common.sh("free -g | awk 'NR==2{print $2}'"),
        "gpu": common.sh("nvidia-smi --query-gpu=name --format=csv,noheader"),
        "nvidia_driver": m.group(1) if m else "",
        "cuda_version": m.group(2) if m else "",
        "docker_version": common.sh("docker --version"),
        "nvidia_container_toolkit": common.sh("nvidia-ctk --version | head -1"),
        "backend": cfg["backend"],
        "server": server_binary_info(cfg),
        "server_version_reported": version,
        "server_env_set_by_harness": {k: v for k, v in server_env.items()
                                      if k.startswith("OLLAMA_")},
        "server_effective_config_logged": server_logged_config(server_log),
        "models": models,
        "config_sha256": common.canonical_hash(cfg["_pinned"]),
        "config_path": cfg["_path"],
        "cli_overrides": overrides,
        "prompts_sha256": cfg["prompts"]["sha256"],
        "git_commit": common.sh(f"git -C {common.REPO} rev-parse HEAD"),
        "git_branch": common.sh(f"git -C {common.REPO} rev-parse --abbrev-ref HEAD"),
        "git_dirty": bool(common.sh(f"git -C {common.REPO} status --porcelain --untracked-files=no")),
        "python": platform.python_version(),
        "other_gpu_processes_at_start": preflight["info"]["gpu_compute_apps"],
        "preflight_ok": preflight["ok"],
        "preflight_problems": preflight["problems"],
        "loadavg_at_start": preflight["info"]["loadavg"],
    }
