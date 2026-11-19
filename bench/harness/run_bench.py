#!/usr/bin/env python3
"""One run: pre-flight -> download+verify -> per model: server -> speed sweep -> stop ->
quality eval -> report. Everything lands in bench/results/<study_id>/<run_id>/.

Protocol studies are normally driven level by level by run_protocol.py (server slots = users),
which calls this once per model and level:
  python3 harness/run_bench.py --config configs/<study>.json --models m --levels 8 \
      --num-parallel 8 --reps 5 --workloads short_chat coding --skip-quality
and once per model for the quality phase:
  python3 harness/run_bench.py --config configs/<study>.json --models m --skip-speed
"""
import argparse
import copy
import csv
import os
import platform
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime

import common
import engines
import env_snapshot
import preflight
import protocol
import pull_models
import run_quality
import run_speed

EXIT_SERVER_FAILED = 3


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="study config, e.g. configs/<study>.json")
    ap.add_argument("--study", help="results folder name (default: the config's study.id)")
    ap.add_argument("--workloads", nargs="+", help=f"subset of {protocol.WORKLOADS}")
    ap.add_argument("--level-cap-min", type=float,
                    help="stop starting new batches of a workload level after this many minutes")
    ap.add_argument("--allow-deviation", action="store_true",
                    help="run a config that breaks the protocol (tests only; recorded in env.json)")
    ap.add_argument("--no-report", action="store_true", help="skip the per-run spreadsheet")
    ap.add_argument("--models", nargs="+", help="model keys; default: all")
    ap.add_argument("--levels", type=int, nargs="+")
    ap.add_argument("--reps", type=int)
    ap.add_argument("--max-tokens", type=int, dest="max_tokens")
    ap.add_argument("--num-parallel", type=int,
                    help="server slots for the speed phase (protocol: = users)")
    ap.add_argument("--context-length", type=int, help="tokens per sequence (tests only)")
    ap.add_argument("--quality-limit", type=int, help="lm-eval --limit for every task (tests)")
    ap.add_argument("--quality-tasks", nargs="+", help="lm-eval task names (tests)")
    ap.add_argument("--skip-speed", action="store_true")
    ap.add_argument("--skip-quality", action="store_true")
    ap.add_argument("--time-budget-hours", type=float,
                    help="stop before a model that would push the run past this many hours")
    ap.add_argument("--skip-pull", action="store_true",
                    help="don't download; verify file names and sizes only")
    ap.add_argument("--ignore-preflight", action="store_true",
                    help="run even if pre-flight says STOP (only after agreeing with the others)")
    a = ap.parse_args()
    # Background jobs ignore Ctrl-C, so SIGTERM is the stop signal: exit through the finally
    # block below, which stops the benchmark server and the power logger.
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))

    cfg = common.load_json(a.config)
    pinned = copy.deepcopy(cfg)
    overrides = {}
    deviations = protocol.check_config(cfg) if cfg.get("protocol") == protocol.VERSION else \
        [f"config is not a {protocol.VERSION} study (protocol={cfg.get('protocol')!r})"]
    if deviations and not a.allow_deviation:
        raise SystemExit("config breaks the protocol:\n  " + "\n  ".join(deviations))
    if a.allow_deviation:      # a test run: never merged into results or counted as quality
        deviations.append("test run (--allow-deviation)")
    if a.workloads:
        unknown = set(a.workloads) - set(protocol.WORKLOADS)
        if unknown:
            raise SystemExit(f"unknown workloads {sorted(unknown)}; use {protocol.WORKLOADS}")
        overrides["workloads"] = a.workloads
    if a.level_cap_min:
        cfg["sweep"]["level_cap_s"] = a.level_cap_min * 60
        overrides["level_cap_min"] = a.level_cap_min
    if a.levels:
        cfg["sweep"]["concurrency_levels"] = overrides["concurrency_levels"] = a.levels
    if a.reps:
        cfg["sweep"]["reps"] = overrides["reps"] = a.reps
    if a.max_tokens:
        cfg["request"]["max_tokens"] = overrides["max_tokens"] = a.max_tokens
    slots = a.num_parallel or max(cfg["sweep"]["concurrency_levels"])
    if a.num_parallel:
        overrides["slots"] = slots
    context = a.context_length or protocol.CONTEXT_PER_USER
    if a.context_length:
        overrides["context_length"] = context
        deviations.append(f"context per sequence {context}, protocol {protocol.CONTEXT_PER_USER}")
    if a.time_budget_hours:
        cfg.setdefault("run", {})["time_budget_hours"] = overrides["time_budget_hours"] = \
            a.time_budget_hours
    budget_s = (cfg.get("run", {}).get("time_budget_hours") or 1e9) * 3600
    do_quality = not a.skip_quality and cfg["quality"].get("enabled", True)
    t_run0 = time.time()
    if a.quality_tasks:
        cfg["quality"]["tasks"] = overrides["quality_tasks"] = a.quality_tasks
    if a.quality_limit:
        cfg["quality"]["limit"] = overrides["quality_limit"] = a.quality_limit
    for k in ("quality_tasks", "quality_limit"):
        if k in overrides:
            deviations.append(f"{k} overridden to {overrides[k]} (test run)")
    keys = a.models or [m["key"] for m in cfg["models"]]
    if a.models:
        overrides["models"] = keys
    models = [m for m in cfg["models"] if m["key"] in keys]
    if not models:
        raise SystemExit(f"no models match {keys}")
    cfg["_pinned"], cfg["_path"] = pinned, os.path.relpath(a.config, common.REPO)

    run_id = (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + platform.node().split(".")[0]
              + "_" + common.canonical_hash(pinned)[:8])
    study = a.study or cfg.get("study", {}).get("id") or "unsorted"
    run_dir = common.RESULTS / study / run_id
    k = 2
    while run_dir.exists():                 # two runs started in the same second
        run_dir = common.RESULTS / study / f"{run_id}-{k}"
        k += 1
    run_id = run_dir.name
    run_dir.mkdir(parents=True)
    print(f"run_id {run_id}\nrun dir {run_dir}")
    prompts = common.load_prompts(cfg)
    if a.workloads:
        prompts = [p for p in prompts if p["id"] in a.workloads]

    pf = preflight.check(cfg)
    common.write_json(run_dir / "preflight.json", pf)
    if not pf["ok"]:
        print("PRE-FLIGHT STOP:\n  " + "\n  ".join(pf["problems"]))
        if not a.ignore_preflight:
            sys.exit(1)

    phases, model_info, quality, skipped, servers = [], [], [], [], []

    def save_run():
        common.write_json(run_dir / "run.json", {
            "run_id": run_id, "phases": phases, "models": model_info, "quality": quality,
            "models_skipped": skipped, "servers": servers,
            "total_seconds": round(time.time() - t_run0)})

    def phase(name, t0):
        phases.append({"phase": name, "start_unix": round(t0, 3), "end_unix": round(time.time(), 3),
                       "seconds": round(time.time() - t0, 1)})

    power = subprocess.Popen([sys.executable, str(common.BENCH / "harness/power_logger.py"),
                              "--out", str(run_dir / "power.csv"),
                              "--interval", str(cfg["power"]["interval_s"])])
    server = None
    exit_code = 0
    try:
        t0 = time.time()
        time.sleep(cfg["sweep"]["idle_baseline_s"])        # idle power baseline, nothing loaded
        phase("idle_baseline", t0)

        t0 = time.time()
        recs = []
        for m in models:
            if not a.skip_pull:
                pull_models.pull(m)
            rec = pull_models.verify(m, rehash=not a.skip_pull)
            print(f"  {m['repo']}: {'VERIFIED' if rec['ok'] else 'MISMATCH'}"
                  f"{'' if rec['ok'] else ' ' + str(rec.get('missing') or rec.get('note'))}")
            recs.append(rec)
        common.write_json(run_dir / "models.json", recs)
        phase("pull_verify", t0)
        if not all(r["ok"] for r in recs):
            raise SystemExit("model weights don't match the pinned checksums; see models.json")

        envj = env_snapshot.snapshot(cfg, run_id, overrides, pf, recs)
        envj.update({"protocol_version": cfg.get("protocol", "pre-1.0"), "study_id": study,
                     "study": cfg.get("study", {}), "quantization": cfg.get("quantization", ""),
                     "engine": cfg["backend"]["type"], "slots": slots if not a.skip_speed else "",
                     "context_per_sequence": context, "protocol_deviations": deviations,
                     "workloads": [p["id"] for p in prompts] if not a.skip_speed else [],
                     "servers": servers})
        common.write_json(run_dir / "env.json", envj)
        eff = {k: v for k, v in cfg.items() if not k.startswith("_")}
        eff["models"] = models
        common.write_json(run_dir / "effective_config.json", eff)

        if not a.skip_speed:
            with open(run_dir / "requests.csv", "w", newline="") as f1:
                w1 = csv.DictWriter(f1, fieldnames=run_speed.REQ_FIELDS)
                w1.writeheader()
                last = None          # (seconds, GB) of the previous model
                for i, m in enumerate(models):
                    elapsed = time.time() - t_run0
                    gb = m.get("model_blob_gb") or 1
                    est = last[0] * gb / last[1] * 1.25 if last else 0
                    stop = ""
                    if elapsed + est > budget_s:
                        stop = (f"time budget: {elapsed / 3600:.2f} h used, next model "
                                f"estimated at {est / 3600:.2f} h")
                    elif i > 0:
                        again = preflight.check(cfg, check_load=False)
                        if not again["ok"] and not a.ignore_preflight:
                            stop = "another user appeared: " + "; ".join(again["problems"])
                    if stop:
                        print(f"STOPPING before {m['tag']}: {stop}")
                        skipped += [{"key": x["key"], "reason": stop} for x in models[i:]]
                        break
                    print(f"== speed: {m['tag']} with {slots} slots "
                          f"(elapsed {elapsed / 60:.0f} min)")
                    t0 = time.time()
                    server = engines.Server(cfg, m, slots, run_dir / f"server_{m['key']}.log",
                                            context)
                    try:
                        server.start()
                    except engines.ServerFailed as e:
                        print(f"SERVER FAILED for {m['key']}: {e}")
                        model_info.append({"key": m["key"], "tag": m["tag"], "fits_gpu": False,
                                           "server_error": str(e)[-2000:]})
                        exit_code = EXIT_SERVER_FAILED
                        server = None
                        phase(f"speed:{m['key']}", t0)
                        save_run()
                        continue
                    sinfo = dict(server.info(), model=m["key"])
                    servers.append(sinfo)
                    common.write_json(run_dir / "env.json", envj)
                    row_meta = row_fields(cfg, envj, m, sinfo, study, slots)
                    info = run_speed.run_model(cfg, m, prompts, run_id, w1, row_meta=row_meta)
                    info["server"] = {k: v for k, v in sinfo.items() if k != "command"}
                    info["server_after"] = engines.vllm_metrics(engines.base_url(cfg)) \
                        if cfg["backend"]["type"] == "vllm" else {}
                    model_info.append(info)
                    f1.flush()
                    server.stop()
                    server = None
                    phase(f"speed:{m['key']}", t0)
                    last = (time.time() - t0, gb)
                    save_run()      # survive an interrupted run

        if do_quality:
            q = cfg["quality"]
            for m in models:
                print(f"== quality: {m['tag']}")
                t0 = time.time()
                qdir = run_dir / "lm_eval" / m["key"]
                qdir.mkdir(parents=True)
                server = engines.Server(cfg, m, q.get("slots", protocol.QUALITY_SLOTS),
                                        run_dir / f"server_quality_{m['key']}.log",
                                        q.get("context", protocol.QUALITY_CONTEXT))
                try:
                    server.start()
                except engines.ServerFailed as e:
                    print(f"SERVER FAILED for {m['key']} (quality): {e}")
                    quality.append({"model_key": m["key"], "model_tag": m["tag"],
                                    "server_error": str(e)[-2000:], "results": []})
                    exit_code = EXIT_SERVER_FAILED
                    server = None
                    continue
                sinfo = dict(server.info(), model=m["key"], phase="quality")
                servers.append(sinfo)
                rec = run_quality.run(cfg, m, qdir, cfg["quality"].get("limit"))
                rec["server"] = {k: v for k, v in sinfo.items() if k != "command"}
                quality.append(rec)
                server.stop()
                server = None
                phase(f"quality:{m['key']}", t0)
                save_run()
            common.write_json(run_dir / "env.json", envj)
    finally:
        if server is not None:
            server.stop()
        power.send_signal(signal.SIGTERM)
        power.wait(timeout=10)
        save_run()
    if not a.no_report:
        subprocess.run([sys.executable, str(common.BENCH / "harness/make_report.py"),
                        str(run_dir)], check=False)
    print(f"done: {run_dir}")
    sys.exit(exit_code)


def row_fields(cfg, envj, m, sinfo, study, slots):
    """The protocol columns (protocol.ROW_FIELDS) every request row of this model carries."""
    engine = cfg["backend"]["type"]
    return {
        "protocol_version": envj["protocol_version"], "study_id": study,
        "machine": cfg.get("study", {}).get("machine") or platform.node(),
        "backend": f"{engine}/native", "engine": engine,
        "engine_version": sinfo.get("engine_version", ""),
        "server_version": sinfo.get("engine_version", ""),
        "quantization": envj["quantization"], "slots": slots,
        "model_repo": m["repo"], "model_revision": m["revision"],
        "weight_format": m.get("weight_format", ""),
        "bits_per_weight": m.get("bits_per_weight", ""),
        "kv_cache_dtype": sinfo.get("kv_cache_dtype", "" if engine == "vllm" else "n/a (mlx)"),
        "kv_cache_tokens": sinfo.get("kv_cache_tokens") or sinfo.get("kv_cache_tokens_logged", ""),
        "thinking": m.get("thinking_off", ""),
    }


if __name__ == "__main__":
    main()
