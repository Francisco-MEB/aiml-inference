#!/usr/bin/env python3
"""Combines every protocol run of a study into bench/results/<study_id>/combined/ and builds
findings.xlsx there, plus timing.csv (projected against actual minutes per model and level).

  python3 harness/merge_study.py <study_id>

Each model appears as "<model key> (<quantization>)", so the quantization is visible in every
sheet and chart. Only slots = users runs are merged; deviating test runs are left out unless
--include-deviations is given."""
import argparse
import csv
import subprocess
import sys

import common
import study_data


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("study")
    ap.add_argument("--include-deviations", action="store_true")
    a = ap.parse_args()
    root = common.RESULTS / a.study
    by_model, timing, quant = {}, [], {}
    keep = study_data.chosen_runs(a.study)       # newest complete run per level only
    for d in sorted(root.iterdir()):
        if d.name not in keep:
            continue
        env_p = d / "env.json"
        if d.name == "combined" or not env_p.is_file() or not (d / "requests.csv").is_file():
            continue
        env = common.load_json(env_p)
        ov = env.get("cli_overrides") or {}
        levels = ov.get("concurrency_levels") or []
        slots = str(env.get("server_env_set_by_harness", {}).get("OLLAMA_NUM_PARALLEL", ""))
        if len(levels) != 1 or slots != str(levels[0]):
            continue
        if env.get("protocol_deviations") and not a.include_deviations \
                and env.get("protocol_version", "pre-1.0") != "pre-1.0":
            continue
        run = common.load_json(d / "run.json") if (d / "run.json").is_file() else {}
        for mk in ov.get("models") or []:
            by_model.setdefault(mk, []).append(str(d))
            quant[mk] = env.get("quantization") or "quantization not recorded"
            timing.append({"model": mk, "users": levels[0],
                           "workloads": " ".join(env.get("workloads") or []),
                           "actual_min": round((run.get("total_seconds") or 0) / 60, 2),
                           "run_id": d.name})
    if not by_model:
        raise SystemExit(f"no slots = users runs in {root}")
    est = root / "time_estimate.json"
    if est.is_file():
        e = common.load_json(est)["modes"]
        for t in timing:
            for mode, res in e.items():
                lv = next((x for x in res["models"].get(t["model"], {}).get("levels", [])
                           if x["users"] == t["users"]), None)
                t[f"projected_min_{mode}"] = round(lv["seconds"] / 60, 2) if lv else ""
    out = root / "combined"
    cmd = [sys.executable, str(common.BENCH / "harness/merge_runs.py"), "--out", str(out)]
    for mk, dirs in by_model.items():
        cmd += ["--add", quant[mk], mk, *dirs]
    subprocess.run(cmd, check=True)
    timing.sort(key=lambda t: (t["model"], t["users"]))
    fields = sorted({k for t in timing for k in t}, key=lambda k: list(timing[0]).index(k)
                    if k in timing[0] else 99)
    with open(out / "timing.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(timing)
    subprocess.run([sys.executable, str(common.BENCH / "harness/make_report.py"), str(out)],
                   check=True)


if __name__ == "__main__":
    main()
