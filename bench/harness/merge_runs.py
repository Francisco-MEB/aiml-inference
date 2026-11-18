#!/usr/bin/env python3
"""Combines several run directories into one, so make_report.py can put them in one spreadsheet.

Each --add names a model variant: a label, a model key, and the run directories that hold its
data (several when each concurrency level was a separate run). The variant shows up in the
report as "<model> (<label>)".

  python3 harness/merge_runs.py --out results/combined \
      --add "slots=16" e4b results/RUN_A \
      --add "slots=users" e4b results/RUN_B1 results/RUN_B2 results/RUN_B3

Only complete data should be added: rows of a model that was interrupted are left out simply
by not naming that model for that run."""
import argparse
import csv
import pathlib

import common
import run_speed


def read_csv(path):
    if not pathlib.Path(path).is_file():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--add", action="append", nargs="+", required=True,
                    metavar=("LABEL MODEL_KEY", "RUN_DIR"))
    ap.add_argument("--quality-from", nargs="*", default=[], metavar="RUN_DIR",
                    help="run directories whose lm-eval results go into the Quality sheet")
    a = ap.parse_args()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    reqs, xcheck, power, phases, minfo, mver, cfg_models, sources = [], [], {}, [], [], [], [], []
    levels, env0, cfg0, total = set(), None, None, 0.0
    seen_dirs = set()
    for label, key, *dirs in a.add:
        variant = f"{key} ({label})"
        infos = []
        for d in map(pathlib.Path, dirs):
            env = common.load_json(d / "env.json")
            cfg = common.load_json(d / "effective_config.json")
            run = common.load_json(d / "run.json") if (d / "run.json").is_file() else {}
            env0, cfg0 = env0 or env, cfg0 or cfg
            slots = env["server_env_set_by_harness"].get("OLLAMA_NUM_PARALLEL")
            rows = [r for r in read_csv(d / "requests.csv") if r["model_key"] == key]
            for r in rows:
                r["model_key"] = variant
                levels.add(int(r["concurrency"]))
            reqs += rows
            for x in read_csv(d / "native_crosscheck.csv"):
                if x["model_key"] == key:
                    x["model_key"] = variant
                    xcheck.append(x)
            infos += [dict(m, slots=slots) for m in run.get("models", []) if m["key"] == key]
            sources.append({"variant": variant, "run_id": env["run_id"],
                            "OLLAMA_NUM_PARALLEL": slots,
                            "levels": sorted({int(r["concurrency"]) for r in rows}),
                            "requests": len(rows), "git_commit": env["git_commit"],
                            "config_sha256": env["config_sha256"],
                            "cli_overrides": env["cli_overrides"],
                            "complete_run": bool(run)})
            if d not in seen_dirs:
                seen_dirs.add(d)
                p = read_csv(d / "power.csv")
                for row in p:
                    power[row["t_unix"]] = row
                if p:
                    total += float(p[-1]["t_unix"]) - float(p[0]["t_unix"])
                if not phases:
                    phases += [x for x in run.get("phases", []) if x["phase"] == "idle_baseline"]
        base = next(m for m in cfg["models"] if m["key"] == key)
        cfg_models.append(dict(base, key=variant, label=f"{base.get('label', key)} ({label})"))
        ver = next((m for m in env["models"] if m["key"] == key), {"tag": base["tag"]})
        mver.append(dict(ver, key=variant))
        # Model facts (memory, load time) from the run with the most parallel slots.
        infos.sort(key=lambda m: int(m.get("slots") or 0))
        info = dict(infos[-1]) if infos else {"tag": base["tag"]}
        info["key"] = variant
        info["memory_by_slots"] = {m["slots"]: (m.get("ps_after_load") or [{}])[0].get("size")
                                   for m in infos}
        minfo.append(info)

    quality, qcfg = [], None
    for d in map(pathlib.Path, a.quality_from):
        quality += common.load_json(d / "run.json").get("quality", [])
        qcfg = common.load_json(d / "effective_config.json")["quality"]
        p = read_csv(d / "power.csv")
        if p:
            total += float(p[-1]["t_unix"]) - float(p[0]["t_unix"])

    write_csv(out / "requests.csv", reqs, run_speed.REQ_FIELDS)
    write_csv(out / "native_crosscheck.csv", xcheck, run_speed.XCHECK_FIELDS)
    prow = [power[k] for k in sorted(power, key=float)]
    if prow:
        write_csv(out / "power.csv", prow, list(prow[0]))
    env0 = dict(env0, run_id=out.name, models=mver, merged_from=sources)
    env0["cli_overrides"] = "see merged_from: one entry per source run"
    env0["server_env_set_by_harness"] = dict(
        env0["server_env_set_by_harness"],
        OLLAMA_NUM_PARALLEL="varies by run: see merged_from (protocol studies use slots equal "
                            "to the number of simultaneous users at each level)")
    common.write_json(out / "env.json", env0)
    cfg0 = dict(cfg0, models=cfg_models)
    if qcfg:
        cfg0["quality"] = dict(qcfg, tasks=sorted({t for q in quality
                                                   for t in q["tasks_requested"]}),
                               limit="per run: " + ", ".join(sorted(
                                   {f"{'+'.join(q['tasks_requested'])}={q['limit']}"
                                    for q in quality})))
    cfg0["sweep"] = dict(cfg0["sweep"], concurrency_levels=sorted(levels))
    common.write_json(out / "effective_config.json", cfg0)
    common.write_json(out / "run.json", {"run_id": out.name, "phases": phases, "models": minfo,
                                         "quality": quality, "models_skipped": [],
                                         "total_seconds": round(total), "merged_from": sources})
    print(f"merged {len(reqs)} requests from {len(seen_dirs)} runs into {out}")


if __name__ == "__main__":
    main()
