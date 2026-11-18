#!/usr/bin/env python3
"""Runs a protocol study level by level, unattended. Normally started through
bench/run_protocol.sh, which detaches it from your session.

  python3 harness/run_protocol.py --config configs/<study>.json --calibrate
  python3 harness/run_protocol.py --config configs/<study>.json --mode stop --cap-hours 6

Each level is one run_bench.py call with server slots = users, so the server restarts between
levels. Already-complete levels are skipped, so the same command resumes after an interruption.

Modes for levels above the calibration levels (protocol section "High levels"):
  stop       per model and workload, stop after the first level that fails the relaxed
             scenario (p95 first token > 5 s, median < 5 tok/s, or errors) or does not fit
             fully on the GPU. Higher levels cannot change any capacity figure.
  all        every level up to 128 users.
  cap:<min>  every level, but each workload level stops starting new batches after <min> minutes.

Progress goes to bench/.run/<study>.status.json and bench/.run/<study>.log (see status.sh)."""
import argparse
import signal
import subprocess
import sys
import time
from datetime import UTC, datetime

import common
import estimate_time
import preflight
import protocol
import study_data

RUN_DIR = common.BENCH / ".run"
STOP = {"requested": False, "child": None}


def _on_term(*_):
    """bench/stop_protocol.sh sends SIGTERM: pass it to the running level (which stops its
    server cleanly), then leave after it ends. Completed levels are kept for a resume."""
    STOP["requested"] = True
    if STOP["child"] and STOP["child"].poll() is None:
        STOP["child"].send_signal(signal.SIGTERM)


def log(msg):
    print(f"[{datetime.now(UTC).strftime('%H:%M:%SZ')}] {msg}", flush=True)


class Status:
    def __init__(self, study, cfg_path, mode, cap_hours):
        self.path = RUN_DIR / f"{study}.status.json"
        self.d = {"study_id": study, "config": cfg_path, "mode": mode, "cap_hours": cap_hours,
                  "started_utc": datetime.now(UTC).isoformat(timespec="seconds"),
                  "state": "starting", "current": None, "done": [], "skipped": [],
                  "estimate_remaining_min": None, "updated_utc": None}

    def set(self, **kw):
        self.d.update(kw)
        self.d["updated_utc"] = datetime.now(UTC).isoformat(timespec="seconds")
        RUN_DIR.mkdir(exist_ok=True)
        common.write_json(self.path, self.d)


def wait_for_free_gpu(cfg, free_min, max_wait_min, status):
    """The GPU must be free of other users for free_min minutes in a row (first step only)."""
    t0, since = time.time(), None
    while True:
        pf = preflight.check(cfg)
        if pf["ok"]:
            since = since or time.time()
            if time.time() - since >= free_min * 60:
                return True
        else:
            if since:
                log("someone started using the machine; waiting again")
            since = None
            status.set(state="waiting for a free GPU", waiting_reason=pf["problems"][:3])
        if time.time() - t0 >= max_wait_min * 60 or STOP["requested"]:
            return False
        time.sleep(30)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs="+", help="model keys (default: all in the config)")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--calibrate", action="store_true",
                   help=f"run only the calibration levels {protocol.CALIBRATION_LEVELS}")
    g.add_argument("--mode", help="stop | all | cap:<minutes per level>")
    ap.add_argument("--cap-hours", type=float, default=None,
                    help="no new level starts after this many hours of running")
    ap.add_argument("--free-min", type=float, default=10,
                    help="minutes the GPU must be free before the first step")
    ap.add_argument("--max-wait-min", type=float, default=240)
    ap.add_argument("--levels", type=int, nargs="+", help="override levels (tests only)")
    ap.add_argument("--test-overrides", nargs=argparse.REMAINDER, default=[],
                    help="extra run_bench flags for a smoke test, e.g. --reps 1 --max-tokens 32 "
                         "--allow-deviation (results are marked as deviating)")
    a = ap.parse_args()
    signal.signal(signal.SIGTERM, _on_term)

    cfg = common.load_json(a.config)
    study = cfg["study"]["id"]
    bad = protocol.check_config(cfg)
    if bad and "--allow-deviation" not in a.test_overrides:
        raise SystemExit("config breaks the protocol:\n  " + "\n  ".join(bad))
    mode, level_cap_min = ("calibrate", None) if a.calibrate else (a.mode.split(":")[0], None)
    if mode == "cap":
        level_cap_min = float(a.mode.split(":")[1])
    if mode not in ("calibrate", *protocol.MODES):
        raise SystemExit(f"unknown mode {a.mode!r}")
    levels = a.levels or (protocol.CALIBRATION_LEVELS if a.calibrate else protocol.LEVELS)
    models = [m for m in cfg["models"] if not a.models or m["key"] in a.models]
    status = Status(study, a.config, a.mode or "calibrate", a.cap_hours)
    log(f"study {study}: mode {status.d['mode']}, models {[m['key'] for m in models]}, "
        f"levels {levels}, cap {a.cap_hours} h")

    status.set(state="waiting for a free GPU")
    if not wait_for_free_gpu(cfg, a.free_min, a.max_wait_min, status):
        status.set(state="gave up: GPU never free")
        log("GPU not free long enough; giving up")
        sys.exit(1)

    t_start, pulled = time.time(), set()
    for m in models:
        stopped = {}                        # workload -> reason (stop mode)
        for n in levels:
            have = study_data.completed(study, m["key"], n)
            todo = [w for w in protocol.WORKLOADS if w not in have and w not in stopped]
            # In stop mode a workload that already failed at a lower level stays stopped.
            if mode == "stop":
                for w in list(todo):
                    prev = study_data.failed_below(study, m["key"], w, n)
                    if prev:
                        stopped[w] = prev
                        todo.remove(w)
            if not todo:
                continue
            if a.cap_hours and time.time() - t_start >= a.cap_hours * 3600:
                status.d["skipped"].append({"model": m["key"], "users": n, "reason": "time cap"})
                log(f"CAP reached; skipping {m['key']} at {n} users")
                continue
            pf = preflight.check(cfg, check_load=False)
            if not pf["ok"]:
                status.set(state="paused: another user on the GPU")
                log("another user is on the GPU; waiting")
                if not wait_for_free_gpu(cfg, 2, a.max_wait_min, status):
                    status.set(state="gave up: GPU busy")
                    sys.exit(1)
            est = estimate_time.remaining(study, cfg, models, levels, mode, level_cap_min)
            status.set(state="running", current={"model": m["key"], "users": n, "workloads": todo},
                       estimate_remaining_min=est)
            log(f"START {m['key']} {n} users, workloads {todo}; remaining estimate {est} min")
            cmd = [sys.executable, "-u", str(common.BENCH / "harness/run_bench.py"),
                   "--config", a.config, "--models", m["key"], "--levels", str(n),
                   "--num-parallel", str(n), "--reps", str(protocol.reps_for(n)),
                   "--workloads", *todo, "--skip-quality", "--no-report"]
            if m["key"] in pulled or study_data.any_run(study, m["key"]):
                cmd.append("--skip-pull")
            if level_cap_min:
                cmd += ["--level-cap-min", str(level_cap_min)]
            cmd += a.test_overrides
            t0 = time.time()
            STOP["child"] = subprocess.Popen(cmd)
            rc = STOP["child"].wait()
            if STOP["requested"]:
                status.set(state="stopped by request (resume with the same command)", current=None)
                log(f"STOPPED by request during {m['key']} {n} users; completed levels are kept")
                sys.exit(0)
            pulled.add(m["key"])
            mins = round((time.time() - t0) / 60, 1)
            if rc != 0:
                log(f"FAILED {m['key']} {n} users (exit {rc}, {mins} min)")
                status.d["skipped"].append({"model": m["key"], "users": n, "reason": f"exit {rc}"})
                if mode == "stop":
                    stopped.update({w: f"run failed at {n} users" for w in todo})
                continue
            status.d["done"].append({"model": m["key"], "users": n, "workloads": todo,
                                     "minutes": mins})
            log(f"END {m['key']} {n} users ({mins} min)")
            if mode == "stop":
                for w in todo:
                    why = study_data.level_failure(study, m["key"], w, n)
                    if why:
                        stopped[w] = why
                        log(f"  stop rule: {m['key']} {w} stops after {n} users ({why})")
        if stopped:
            status.d["skipped"] += [{"model": m["key"], "workload": w, "reason": r}
                                    for w, r in stopped.items()]

    status.set(state="building report", current=None, estimate_remaining_min=0)
    merge = [sys.executable, str(common.BENCH / "harness/merge_study.py"), study]
    if "--allow-deviation" in a.test_overrides:
        merge.append("--include-deviations")
    subprocess.run(merge, check=False)
    if a.calibrate:
        subprocess.run([sys.executable, str(common.BENCH / "harness/estimate_time.py"),
                        "--config", a.config], check=False)
    status.set(state="done", finished_utc=datetime.now(UTC).isoformat(timespec="seconds"),
               total_min=round((time.time() - t_start) / 60, 1))
    log(f"PROTOCOL_DONE {study} ({status.d['total_min']} min)")


if __name__ == "__main__":
    main()
