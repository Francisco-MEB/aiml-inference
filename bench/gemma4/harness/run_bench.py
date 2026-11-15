#!/usr/bin/env python3
"""One command for a whole run: pre-flight -> server -> pull+verify -> speed sweep ->
quality eval -> report. Everything lands in results/<run_id>/.

  python3 harness/run_bench.py                                   # full sweep, pinned config
  python3 harness/run_bench.py --models e4b --levels 1 2 --reps 1 \
      --max-tokens 32 --num-parallel 2 --quality-limit 5         # smoke test
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
import env_snapshot
import preflight
import pull_models
import run_quality
import run_speed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(common.BENCH / "config/gemma4_ollama_native.json"))
    ap.add_argument("--models", nargs="+", help="model keys; default: all non-optional")
    ap.add_argument("--levels", type=int, nargs="+")
    ap.add_argument("--reps", type=int)
    ap.add_argument("--max-tokens", "--num-predict", type=int, dest="max_tokens")
    ap.add_argument("--num-parallel", type=int, help="OLLAMA_NUM_PARALLEL for the speed phase")
    ap.add_argument("--context-length", type=int, help="OLLAMA_CONTEXT_LENGTH, speed phase")
    ap.add_argument("--quality-limit", type=int, help="lm-eval --limit")
    ap.add_argument("--skip-speed", action="store_true")
    ap.add_argument("--skip-quality", action="store_true")
    ap.add_argument("--with-quality", action="store_true",
                    help="run lm-eval even if quality.enabled is false in the config")
    ap.add_argument("--time-budget-hours", type=float,
                    help="stop before a model that would push the run past this many hours")
    ap.add_argument("--skip-pull", action="store_true", help="verify digests but do not pull")
    ap.add_argument("--ignore-preflight", action="store_true",
                    help="run even if pre-flight says STOP (only after agreeing with the others)")
    a = ap.parse_args()

    cfg = common.load_json(a.config)
    pinned = copy.deepcopy(cfg)
    overrides = {}
    if a.levels:
        cfg["sweep"]["concurrency_levels"] = overrides["concurrency_levels"] = a.levels
    if a.reps:
        cfg["sweep"]["reps"] = overrides["reps"] = a.reps
    if a.max_tokens:
        cfg["request"]["max_tokens"] = overrides["max_tokens"] = a.max_tokens
    if a.num_parallel:
        cfg["server"]["env"]["OLLAMA_NUM_PARALLEL"] = overrides["OLLAMA_NUM_PARALLEL"] = \
            str(a.num_parallel)
    if a.context_length:
        cfg["server"]["env"]["OLLAMA_CONTEXT_LENGTH"] = overrides["OLLAMA_CONTEXT_LENGTH"] = \
            str(a.context_length)
    if a.time_budget_hours:
        cfg.setdefault("run", {})["time_budget_hours"] = overrides["time_budget_hours"] = \
            a.time_budget_hours
    budget_s = (cfg.get("run", {}).get("time_budget_hours") or 1e9) * 3600
    do_quality = not a.skip_quality and (cfg["quality"].get("enabled", True) or a.with_quality)
    t_run0 = time.time()
    if a.quality_limit:
        cfg["quality"]["limit"] = overrides["quality_limit"] = a.quality_limit
    keys = a.models or [m["key"] for m in cfg["models"] if not m.get("optional")]
    if a.models:
        overrides["models"] = keys
    models = [m for m in cfg["models"] if m["key"] in keys]
    if not models:
        raise SystemExit(f"no models match {keys}")
    cfg["_pinned"], cfg["_path"] = pinned, os.path.relpath(a.config, common.REPO)
    is_ollama = cfg["backend"]["type"] == "ollama"
    if cfg["server"]["mode"] == "native":
        cfg["server"]["_binary_path"] = cfg["server"]["native"].get("binary_path") or str(
            common.BENCH / f".tools/ollama-{cfg['server']['ollama_version']}/bin/ollama")

        want = cfg["server"]["native"].get("binary_sha256")
        if cfg["server"]["native"].get("binary_path") and want:
            got = common.sha256_file(cfg["server"]["_binary_path"])
            if got != want:
                raise SystemExit(f"Ollama binary sha256 mismatch: pinned {want}, found {got}")

    run_id = (datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "_" + platform.node()
              + "_" + common.canonical_hash(pinned)[:8])
    run_dir = common.BENCH / "results" / run_id
    run_dir.mkdir(parents=True)
    print(f"run_id {run_id}\nrun dir {run_dir}")
    prompts = common.load_prompts(cfg)

    pf = preflight.check(cfg)
    common.write_json(run_dir / "preflight.json", pf)
    if not pf["ok"]:
        print("PRE-FLIGHT STOP:\n  " + "\n  ".join(pf["problems"]))
        if not a.ignore_preflight:
            sys.exit(1)

    phases, model_info, quality, skipped = [], [], [], []

    def save_run():
        common.write_json(run_dir / "run.json", {
            "run_id": run_id, "phases": phases, "models": model_info, "quality": quality,
            "models_skipped": skipped, "total_seconds": round(time.time() - t_run0)})

    def phase(name, t0):
        phases.append({"phase": name, "start_unix": round(t0, 3), "end_unix": round(time.time(), 3),
                       "seconds": round(time.time() - t0, 1)})

    power = subprocess.Popen([sys.executable, str(common.BENCH / "harness/power_logger.py"),
                              "--out", str(run_dir / "power.csv"),
                              "--interval", str(cfg["power"]["interval_s"])])
    env = common.server_env(cfg)
    if is_ollama and cfg["server"]["mode"] == "native":
        env["OLLAMA_BENCH_BIN"] = cfg["server"]["_binary_path"]
    started = False
    try:
        t0 = time.time()
        time.sleep(cfg["sweep"]["idle_baseline_s"])        # idle power baseline, nothing loaded
        phase("idle_baseline", t0)

        if is_ollama:
            common.start_server(cfg, env, run_dir / "server_speed.log")
            started = True
            t0 = time.time()
            recs = []
            for m in models:
                if not a.skip_pull:
                    print(f"pulling {m['tag']} ...")
                    pull_models.pull(cfg["backend"]["native_url"], m["tag"])
                rec = pull_models.verify(cfg, m)
                print(f"  {m['tag']}: {'VERIFIED' if rec['ok'] else 'MISMATCH'} {rec['checks']}")
                recs.append(rec)
            common.write_json(run_dir / "models.json", recs)
            phase("pull_verify", t0)
            if not all(r["ok"] for r in recs):
                raise SystemExit("model digest mismatch; see models.json")
        else:
            recs = [{"key": m["key"], "tag": m["tag"], "note": "not verified by harness"}
                    for m in models]

        envj = env_snapshot.snapshot(cfg, run_id, overrides, pf, recs,
                                     run_dir / "server_speed.log", env)
        common.write_json(run_dir / "env.json", envj)
        eff = {k: v for k, v in cfg.items() if not k.startswith("_")}
        eff["models"] = models
        common.write_json(run_dir / "effective_config.json", eff)

        if not a.skip_speed:
            with open(run_dir / "requests.csv", "w", newline="") as f1, \
                    open(run_dir / "native_crosscheck.csv", "w", newline="") as f2:
                w1 = csv.DictWriter(f1, fieldnames=run_speed.REQ_FIELDS)
                w2 = csv.DictWriter(f2, fieldnames=run_speed.XCHECK_FIELDS)
                w1.writeheader()
                w2.writeheader()
                size = {r["key"]: r.get("size_bytes") or 1 for r in recs}
                last = None          # (seconds, size) of the previous model
                for i, m in enumerate(models):
                    elapsed = time.time() - t_run0
                    est = last[0] * size[m["key"]] / last[1] * 1.25 if last else 0
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
                    print(f"== speed: {m['tag']} (elapsed {elapsed / 60:.0f} min)")
                    t0 = time.time()
                    model_info.append(run_speed.run_model(cfg, m, prompts, run_id, w1, w2))
                    f1.flush()
                    f2.flush()
                    phase(f"speed:{m['key']}", t0)
                    last = (time.time() - t0, size[m["key"]])
                    save_run()      # survive an interrupted run

        if do_quality:
            qenv = common.server_env(cfg, cfg["quality"].get("server_env"))
            if "OLLAMA_BENCH_BIN" in env:
                qenv["OLLAMA_BENCH_BIN"] = env["OLLAMA_BENCH_BIN"]
            if is_ollama and qenv != env:
                common.stop_server(cfg)
                started = False
                common.start_server(cfg, qenv, run_dir / "server_quality.log")
                started = True
            for m in models:
                print(f"== quality: {m['tag']}")
                t0 = time.time()
                qdir = run_dir / "lm_eval" / m["key"]
                qdir.mkdir(parents=True)
                rec = run_quality.run(cfg, m, qdir, cfg["quality"].get("limit"))
                rec["server_env"] = {k: v for k, v in qenv.items() if k.startswith("OLLAMA_")}
                quality.append(rec)
                if is_ollama:
                    run_speed.unload(cfg, m)
                phase(f"quality:{m['key']}", t0)
    finally:
        if started:
            common.stop_server(cfg)
        power.send_signal(signal.SIGTERM)
        power.wait(timeout=10)
        save_run()
    subprocess.run([sys.executable, str(common.BENCH / "harness/make_report.py"), str(run_dir)],
                   check=False)
    print(f"done: {run_dir}")


if __name__ == "__main__":
    main()
