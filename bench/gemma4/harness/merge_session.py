#!/usr/bin/env python3
"""Merge every finished run of the 2026-11-15 Gemma 4 session (and later runs of the same kind)
into results/combined_gemma4_gb10, then build findings.xlsx.

  python3 harness/merge_session.py                   # merge + report
  python3 harness/merge_session.py missing-quality   # print model keys still lacking quality
  python3 harness/merge_session.py missing-speed     # print model keys lacking a full
                                                     # slots=users sweep (levels 1 2 4 8 16)

slots=users runs: one level per run, OLLAMA_NUM_PARALLEL equal to that level.
slots=16 runs: the first full sweep (16 fixed slots), kept for comparison."""
import json
import pathlib
import subprocess

import common
import sys

BENCH = pathlib.Path(__file__).resolve().parents[1]
R = BENCH / "results"
FIXED = R / "20261115T195725Z_promaxgb10-2c9c_8ad16415"
KEYS = ("e4b", "12b", "31b", "26b-a4b")
LEVELS = {1, 2, 4, 8, 16}


def scan():
    matched, quality = {}, []
    for d in sorted(R.iterdir()):
        if d.name < "20261115T2136" or d.name.startswith("combined") \
                or not (d / "run.json").is_file():
            continue
        env = common.load_json(d / "env.json")
        run = common.load_json(d / "run.json")
        ov = env["cli_overrides"] or {}
        if any(q.get("results") for q in run.get("quality", [])):
            quality.append(d)
        elif (d / "requests.csv").is_file() and run.get("models") \
                and len(ov.get("concurrency_levels", [])) == 1 \
                and ov.get("OLLAMA_NUM_PARALLEL") == str(ov["concurrency_levels"][0]):
            matched.setdefault(ov["models"][0], {})[ov["concurrency_levels"][0]] = d
    return matched, quality


def main():
    matched, quality = scan()
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "missing-speed":
        print(" ".join(k for k in ("26b-a4b", "12b", "31b", "e4b")
                       if set(matched.get(k, {})) < LEVELS or k not in matched))
        return
    if mode == "missing-quality":
        done = set()
        for d in quality:
            for q in common.load_json(d / "run.json")["quality"]:
                if q.get("results"):
                    done.add((q["model_key"], q["tasks_requested"][0]))
        for task in ("mmlu_pro", "humaneval_chat"):
            print(task, " ".join(k for k in KEYS if (k, task) not in done))
        return
    cmd = [sys.executable, str(BENCH / "harness/merge_runs.py"),
           "--out", str(R / "combined_gemma4_gb10")]
    for k in KEYS:
        if k in matched:
            cmd += ["--add", "slots=users", k, *map(str, sorted(matched[k].values()))]
    for k in ("e4b", "12b"):
        cmd += ["--add", "slots=16", k, str(FIXED)]
    cmd += ["--quality-from", *map(str, quality)]
    subprocess.run(cmd, check=True)
    subprocess.run([sys.executable, str(BENCH / "harness/make_report.py"),
                    str(R / "combined_gemma4_gb10")], check=True)


if __name__ == "__main__":
    main()
