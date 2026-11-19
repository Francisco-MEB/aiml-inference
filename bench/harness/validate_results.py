#!/usr/bin/env python3
"""Checks that a study's results follow the protocol, so they line up with other members'.

  python3 harness/validate_results.py <study_id>

Exit code 0 = every check passed (warnings allowed), 1 = at least one FAIL."""
import argparse
import csv
import sys

import common
import protocol
import run_speed


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("study")
    a = ap.parse_args()
    root = common.RESULTS / a.study
    fails, warns, quants, runs = [], [], set(), 0
    for d in sorted(root.iterdir()):
        if d.name == "combined" or not (d / "env.json").is_file():
            continue
        env = common.load_json(d / "env.json")
        ver = env.get("protocol_version", "pre-1.0")
        ov = env.get("cli_overrides") or {}
        if not (d / "requests.csv").is_file():
            warns.append(f"{d.name}: no requests (stopped before running)")
            continue
        runs += 1
        if ver != protocol.VERSION:
            warns.append(f"{d.name}: protocol {ver}, not {protocol.VERSION} (kept for reference)")
            continue
        if env.get("protocol_deviations"):
            fails.append(f"{d.name}: deviates: {'; '.join(env['protocol_deviations'])}")
        if env.get("prompts_sha256") != protocol.PROMPTS_SHA256:
            fails.append(f"{d.name}: prompt file sha256 differs")
        levels = ov.get("concurrency_levels") or []
        slots = str(env.get("server_env_set_by_harness", {}).get("OLLAMA_NUM_PARALLEL"))
        if len(levels) != 1 or slots != str(levels[0]):
            fails.append(f"{d.name}: server slots {slots} do not equal users {levels}")
        for k in ("reps", "max_tokens"):
            if k in ov:
                fails.append(f"{d.name}: {k} overridden to {ov[k]}")
        quants.add(env.get("quantization"))
        with open(d / "requests.csv", newline="") as f:
            reader = csv.DictReader(f)
            missing = [c for c in run_speed.REQ_FIELDS if c not in (reader.fieldnames or [])]
            rows = list(reader)
        if missing:
            fails.append(f"{d.name}: missing columns {missing}")
        n = levels[0] if levels else 0
        run = common.load_json(d / "run.json") if (d / "run.json").is_file() else {}
        capped = {(c["workload"]) for m in run.get("models", []) for c in m.get("capped", [])}
        for w in env.get("workloads") or protocol.WORKLOADS:
            got = sum(1 for r in rows if r["prompt_id"] == w)
            want = n * protocol.reps_for(n)
            if got < want and w not in capped:
                fails.append(f"{d.name}: {w} has {got} requests, protocol needs {want}")
        bad = [r for r in rows if r.get("error") or not r.get("output_tokens")]
        if bad:
            fails.append(f"{d.name}: {len(bad)} requests with an error or no token count")
        think = sum(int(common.num(r.get("reasoning_chars")) or 0) for r in rows)
        if think:
            warns.append(f"{d.name}: {think} reasoning characters returned (thinking was on)")
        for c in ("protocol_version", "quantization", "study_id"):
            vals = {r.get(c) for r in rows}
            if len(vals) != 1 or not next(iter(vals)):
                fails.append(f"{d.name}: column {c} empty or mixed: {sorted(map(str, vals))}")
    if len(quants) > 1:
        fails.append(f"study mixes quantizations {sorted(map(str, quants))}")
    for w in warns:
        print("WARN ", w)
    for f in fails:
        print("FAIL ", f)
    print(f"{runs} runs checked in {root}: {len(fails)} failures, {len(warns)} warnings")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
