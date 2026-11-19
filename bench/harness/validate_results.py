#!/usr/bin/env python3
"""Checks that a study's results follow the protocol, so they line up with other members'.

  python3 harness/validate_results.py <study_id>

Exit code 0 = every check passed (warnings allowed), 1 = at least one FAIL. Runs from older
protocol versions (1.0, pre-1.0) are reported as warnings and kept for reference."""
import argparse
import csv
import sys

import common
import protocol
import run_speed


def check_quality_run(d, env, run, fails, warns, quality_by_model):
    for q in run.get("quality", []):
        mk = q["model_key"]
        if q.get("server_error"):
            fails.append(f"{d.name}: quality server for {mk} failed to start")
            continue
        if q.get("returncode") not in (0, None):
            fails.append(f"{d.name}: lm-eval for {mk} exited with {q['returncode']}")
            continue
        done = {x["task"] for x in q.get("results", [])}
        for s in q.get("tasks_skipped", []):
            warns.append(f"{d.name}: {mk} skipped {s['task']}: {s['reason']}")
        if not done:
            fails.append(f"{d.name}: lm-eval for {mk} produced no results")
            continue
        if env.get("protocol_deviations"):
            warns.append(f"{d.name}: quality for {mk} deviates (test run): "
                         f"{'; '.join(env['protocol_deviations'])}")
            continue
        quality_by_model.setdefault(mk, set()).update(done)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("study")
    a = ap.parse_args()
    root = common.RESULTS / a.study
    fails, warns, quants, runs = [], [], set(), 0
    models_seen, quality_by_model, thinking_exempt = set(), {}, set()
    for d in sorted(root.iterdir()):
        if d.name == "combined" or not (d / "env.json").is_file():
            continue
        env = common.load_json(d / "env.json")
        ver = env.get("protocol_version", "pre-1.0")
        ov = env.get("cli_overrides") or {}
        run = common.load_json(d / "run.json") if (d / "run.json").is_file() else {}
        cfg = common.load_json(d / "effective_config.json") \
            if (d / "effective_config.json").is_file() else {}
        if ver != protocol.VERSION:
            warns.append(f"{d.name}: protocol {ver}, not {protocol.VERSION} (kept for reference)")
            continue
        for m in cfg.get("models", []):
            if m.get("thinking_cannot_disable"):
                thinking_exempt.add(m["key"])
        for m in env.get("models", []):
            if not m.get("ok"):
                fails.append(f"{d.name}: {m.get('repo')} weights not verified against the pins")
            if not m.get("revision"):
                fails.append(f"{d.name}: {m.get('tag')} has no pinned revision")
        if run.get("quality"):
            runs += 1
            check_quality_run(d, env, run, fails, warns, quality_by_model)
        if not (d / "requests.csv").is_file():
            if not run.get("quality"):
                warns.append(f"{d.name}: no requests (stopped before running)")
            continue
        runs += 1
        if env.get("protocol_deviations"):
            fails.append(f"{d.name}: deviates: {'; '.join(env['protocol_deviations'])}")
        if env.get("prompts_sha256") != protocol.PROMPTS_SHA256:
            fails.append(f"{d.name}: prompt file sha256 differs")
        levels = ov.get("concurrency_levels") or []
        slots = common.slots_of(env)
        if len(levels) != 1 or slots != str(levels[0]):
            fails.append(f"{d.name}: server slots {slots} do not equal users {levels}")
        for k in ("reps", "max_tokens", "context_length"):
            if k in ov and not (k == "reps" and levels and ov[k] == protocol.reps_for(levels[0])):
                fails.append(f"{d.name}: {k} overridden to {ov[k]}")
        quants.add(env.get("quantization"))
        for x in run.get("models", []):
            if x.get("fits_gpu") is False:
                warns.append(f"{d.name}: {x['key']} server did not start at {slots} slots "
                             f"(recorded as not fitting)")
        with open(d / "requests.csv", newline="") as f:
            reader = csv.DictReader(f)
            missing = [c for c in run_speed.REQ_FIELDS if c not in (reader.fieldnames or [])]
            rows = list(reader)
        if missing:
            fails.append(f"{d.name}: missing columns {missing}")
        n = levels[0] if levels else 0
        capped = {(c["workload"]) for m in run.get("models", []) for c in m.get("capped", [])}
        started = {x["key"] for x in run.get("models", []) if x.get("fits_gpu") is not False}
        if started:
            for w in env.get("workloads") or protocol.WORKLOADS:
                got = sum(1 for r in rows if r["prompt_id"] == w)
                want = n * protocol.reps_for(n) * len(started)
                if got < want and w not in capped:
                    fails.append(f"{d.name}: {w} has {got} requests, protocol needs {want}")
        bad = [r for r in rows if r.get("error") or not r.get("output_tokens")]
        if bad:
            fails.append(f"{d.name}: {len(bad)} requests with an error or no token count")
        for mk in {r["model_key"] for r in rows}:
            models_seen.add(mk)
            think = sum(int(common.num(r.get("reasoning_chars")) or 0) for r in rows
                        if r["model_key"] == mk)
            if think and mk in thinking_exempt:
                warns.append(f"{d.name}: {mk} returned {think} reasoning characters (this model "
                             f"cannot turn thinking off; recorded at its lowest setting)")
            elif think:
                fails.append(f"{d.name}: {mk} returned {think} reasoning characters: thinking "
                             f"was not off")
        for c in ("protocol_version", "quantization", "study_id", "engine", "model_revision"):
            vals = {r.get(c) for r in rows}
            if not rows or not all(vals) or (c != "model_revision" and len(vals) != 1):
                fails.append(f"{d.name}: column {c} empty or mixed: {sorted(map(str, vals))}")
    if len(quants) > 1:
        fails.append(f"study mixes quantizations {sorted(map(str, quants))}")
    for mk in sorted(models_seen):
        if mk not in quality_by_model:
            warns.append(f"{mk}: no quality result in this study (run the quality phase, or "
                         f"note where the same build's quality was measured)")
    for w in warns:
        print("WARN ", w)
    for f in fails:
        print("FAIL ", f)
    print(f"{runs} runs checked in {root}: {len(fails)} failures, {len(warns)} warnings")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
