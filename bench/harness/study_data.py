"""Reads what a study has measured so far: one entry per (model, users, workload) from the
matched-slot runs in bench/results/<study_id>/. Used for resuming, the stop rule and time
estimates. Works for pre-1.0 runs too (the Gemma study), which lack the new row columns."""
import csv
import statistics

import common
import protocol


def _runs(study):
    root = common.RESULTS / study
    if not root.is_dir():
        return
    for d in sorted(root.iterdir()):
        env_p, req_p = d / "env.json", d / "requests.csv"
        if not (env_p.is_file() and req_p.is_file()):
            continue
        env = common.load_json(env_p)
        ov = env.get("cli_overrides") or {}
        if not isinstance(ov, dict):
            continue
        levels = ov.get("concurrency_levels") or []
        slots = common.slots_of(env)
        if len(levels) != 1 or slots != str(levels[0]):
            continue                       # only slots = users runs count for the protocol
        run = common.load_json(d / "run.json") if (d / "run.json").is_file() else {}
        with open(req_p, newline="") as f:
            rows = list(csv.DictReader(f))
        yield {"dir": d, "env": env, "run": run, "users": levels[0], "rows": rows,
               "reps": ov.get("reps") or protocol.reps_for(levels[0]),
               "models": ov.get("models") or [m["key"] for m in env.get("models", [])]}


def levels(study):
    """{(model, users, workload): {...}} from the newest complete run of each."""
    out = {}
    for r in _runs(study):
        capped = {(c["workload"], c["users"]) for m in r["run"].get("models", [])
                  for c in m.get("capped", [])}
        fits = {m["key"]: m.get("fits_gpu") for m in r["run"].get("models", [])}
        overhead = r["run"].get("total_seconds")
        by = {}
        for row in r["rows"]:
            by.setdefault((row["model_key"], row["prompt_id"]), []).append(row)
        batch_total = 0.0
        for (mk, w), rows in by.items():
            n = r["users"]
            complete = len(rows) >= n * r["reps"] or (w, n) in capped
            batches = {row["rep"]: common.num(row["batch_wall_s"]) for row in rows}
            batch_total += sum(v for v in batches.values() if v)
            st = protocol.level_stats(rows)
            old = out.get((mk, n, w))
            if old and old["complete"] and not complete:
                continue                   # keep the complete run over a later partial one
            out[(mk, n, w)] = {
                "complete": complete, "capped": (w, n) in capped, "fits_gpu": fits.get(mk),
                "batch_s": statistics.mean([v for v in batches.values() if v] or [0]),
                "reps": len(batches), "run_id": r["dir"].name, **st}
        for (mk, n, w), v in out.items():
            if v["run_id"] == r["dir"].name and overhead is not None:
                v["run_overhead_s"] = max(0.0, overhead - batch_total)
    return out


def chosen_runs(study):
    """Run folder names that hold the data used for each complete (model, users, workload)."""
    return {v["run_id"] for v in levels(study).values() if v["complete"]}


def completed(study, model, users):
    return {w for (mk, n, w), v in levels(study).items()
            if mk == model and n == users and v["complete"]}


def any_run(study, model):
    return any(model in r["models"] for r in _runs(study))


def level_failure(study, model, workload, users):
    """Why this level ends the workload in stop mode, or '' if it passes the relaxed rule."""
    v = levels(study).get((model, users, workload))
    if not v:
        return ""
    if v.get("fits_gpu") is False:
        return "model no longer fits fully on the GPU"
    if not protocol.passes(v, protocol.scenario(protocol.STOP_SCENARIO)):
        return (f"fails the {protocol.STOP_SCENARIO} target: p95 first token "
                f"{v['ttft_p95_s'] and round(v['ttft_p95_s'], 2)} s, median "
                f"{v['decode_median'] and round(v['decode_median'], 1)} tok/s, "
                f"{v['errors']} errors")
    return ""


def failed_below(study, model, workload, users):
    for n in protocol.LEVELS:
        if n >= users:
            break
        why = level_failure(study, model, workload, n)
        if why:
            return f"{why} (at {n} users)"
    return ""


def quality_runs(study):
    """{model_key: run folder name} of runs whose quality phase produced results without
    deviations (newest wins)."""
    out = {}
    root = common.RESULTS / study
    if not root.is_dir():
        return out
    for d in sorted(root.iterdir()):
        run_p, env_p = d / "run.json", d / "env.json"
        if not (run_p.is_file() and env_p.is_file()):
            continue
        env = common.load_json(env_p)
        if env.get("protocol_deviations"):
            continue
        for q in common.load_json(run_p).get("quality", []):
            if q.get("results") and q.get("returncode") in (0, None):
                out[q["model_key"]] = d.name
    return out


def quality_elsewhere(repo, revision, exclude_study=None):
    """(study, run) of another study that measured quality for this exact build, or None."""
    if not common.RESULTS.is_dir():
        return None
    for sd in sorted(common.RESULTS.iterdir()):
        if not sd.is_dir() or sd.name == exclude_study:
            continue
        runs = quality_runs(sd.name)
        for mk, run in runs.items():
            eff = sd / run / "effective_config.json"
            if not eff.is_file():
                continue
            for m in common.load_json(eff).get("models", []):
                if m["key"] == mk and m.get("repo") == repo and m.get("revision") == revision:
                    return sd.name, run
    return None
