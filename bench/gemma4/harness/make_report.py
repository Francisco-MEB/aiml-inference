#!/usr/bin/env python3
"""Turns a run directory into findings.xlsx (plus one CSV per sheet in report/).

  python3 harness/make_report.py results/<run_id> [--wall-csv meter.csv]

--wall-csv: a CSV with columns t_unix,watts from the wall power meter. Its readings fill the
wall_meter_w column and add wall-based watts and joules-per-token to the Summary sheet."""
import argparse
import bisect
import csv
import pathlib

import common


def num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def read_csv(path):
    if not pathlib.Path(path).is_file():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


class Series:
    """Piecewise-linear power signal; energy(t0, t1) integrates it in joules."""

    def __init__(self, pts):
        pts = sorted((t, w) for t, w in pts if t is not None and w is not None)
        self.t = [p[0] for p in pts]
        self.w = [p[1] for p in pts]

    def at(self, x):
        if not self.t:
            return None
        i = bisect.bisect_left(self.t, x)
        if i == 0:
            return self.w[0]
        if i == len(self.t):
            return self.w[-1]
        t0, t1 = self.t[i - 1], self.t[i]
        return self.w[i - 1] + (self.w[i] - self.w[i - 1]) * (x - t0) / (t1 - t0)

    def energy(self, a, b):
        if not self.t or b <= a:
            return None
        xs = [a] + [t for t in self.t if a < t < b] + [b]
        ys = [self.at(x) for x in xs]
        return sum((xs[i + 1] - xs[i]) * (ys[i] + ys[i + 1]) / 2 for i in range(len(xs) - 1))

    def mean(self, a, b):
        e = self.energy(a, b)
        return e / (b - a) if e is not None else None


def r(v, nd=2):
    return round(v, nd) if isinstance(v, float) else v


def build(run_dir, wall_csv=None):
    run_dir = pathlib.Path(run_dir)
    env = common.load_json(run_dir / "env.json")
    cfg = common.load_json(run_dir / "effective_config.json")
    run = common.load_json(run_dir / "run.json")
    reqs = read_csv(run_dir / "requests.csv")
    power = read_csv(run_dir / "power.csv")
    xcheck = read_csv(run_dir / "native_crosscheck.csv")
    gpu = Series((num(p["t_unix"]), num(p["gpu_power_w"])) for p in power)
    wall = Series([])
    if wall_csv:
        wall = Series((num(x["t_unix"]), num(x["watts"])) for x in read_csv(wall_csv))
        for p in power:
            v = wall.at(num(p["t_unix"]))
            p["wall_meter_w"] = "" if v is None else round(v, 2)
    idle = next((p for p in run["phases"] if p["phase"] == "idle_baseline"), None)
    idle_w = gpu.mean(idle["start_unix"], idle["end_unix"]) if idle else None
    thr = cfg["thresholds"]

    # ---- Summary -------------------------------------------------------------------------
    groups = {}
    for q in reqs:
        groups.setdefault((q["model_key"], q["prompt_class"], int(q["concurrency"])), []).append(q)
    summary = []
    for (mk, pc, n), rows in groups.items():
        batches = {}
        for q in rows:
            batches[q["rep"]] = (num(q["batch_start_unix"]), num(q["batch_end_unix"]),
                                 num(q["batch_agg_tps"]))
        toks = sum(int(num(q["output_tokens"]) or 0) for q in rows)
        wall_s = sum(b[1] - b[0] for b in batches.values())
        e_gpu = sum(gpu.energy(b[0], b[1]) or 0 for b in batches.values()) if gpu.t else None
        e_wall = sum(wall.energy(b[0], b[1]) or 0 for b in batches.values()) if wall.t else None
        ttft = [num(q["ttft_s"]) for q in rows]
        dec = [num(q["decode_tps"]) for q in rows]
        errs = sum(1 for q in rows if q["error"])
        p95 = common.percentile(ttft, 95)
        dstat = common.percentile(dec, 50) if thr["decode_stat"] == "median" else common.mean(dec)
        ok = (errs == 0 and p95 is not None and dstat is not None
              and p95 <= thr["p95_ttft_s_max"] and dstat >= thr["decode_tps_min"])
        summary.append({
            "model": mk, "prompt_class": pc, "concurrency": n, "n_requests": len(rows),
            "errors": errs,
            "prompt_tokens_mean": r(common.mean([num(q["prompt_tokens"]) for q in rows]), 0),
            "output_tokens_mean": r(common.mean([num(q["output_tokens"]) for q in rows]), 1),
            "ttft_mean_s": r(common.mean(ttft), 3), "ttft_p95_s": r(p95, 3),
            "decode_tps_per_req_mean": r(common.mean(dec)),
            "decode_tps_per_req_median": r(common.percentile(dec, 50)),
            "decode_tps_per_req_min": r(min([d for d in dec if d is not None], default=None)),
            "prefill_tps_est_mean": r(common.mean([num(q["prefill_tps_est"]) for q in rows])),
            "prefill_note": "valid (no queueing)" if n == 1 else "includes queueing; use N=1",
            "aggregate_tps_mean": r(common.mean([b[2] for b in batches.values()])),
            "gpu_power_avg_w": r(e_gpu / wall_s) if e_gpu and wall_s else None,
            "gpu_joules_per_output_token": r(e_gpu / toks, 3) if e_gpu and toks else None,
            "wall_power_avg_w": r(e_wall / wall_s) if e_wall and wall_s else None,
            "wall_joules_per_output_token": r(e_wall / toks, 3) if e_wall and toks else None,
            "reasoning_chars": sum(int(num(q["reasoning_chars"]) or 0) for q in rows),
            "meets_thresholds": "yes" if ok else "no",
        })
    summary.sort(key=lambda s: (s["model"], s["prompt_class"], s["concurrency"]))

    # ---- Capacity ------------------------------------------------------------------------
    capacity = []
    for mk, pc in sorted({(s["model"], s["prompt_class"]) for s in summary}):
        lv = [s for s in summary if s["model"] == mk and s["prompt_class"] == pc]
        passing = [s["concurrency"] for s in lv if s["meets_thresholds"] == "yes"]
        tested = [s["concurrency"] for s in lv]
        cap = max(passing) if passing else 0
        note = ""
        if passing and cap == max(tested):
            note = "passes at the highest level tested; true capacity may be higher"
        if any(c < cap and c not in passing for c in tested):
            note = "a lower level failed while a higher one passed; check the raw rows"
        capacity.append({"model": mk, "prompt_class": pc, "user_capacity": cap,
                         "levels_tested": " ".join(map(str, tested)),
                         "levels_passing": " ".join(map(str, passing)),
                         "rule": f"p95 TTFT <= {thr['p95_ttft_s_max']} s and "
                                 f"{thr['decode_stat']} per-request decode >= "
                                 f"{thr['decode_tps_min']} tok/s, 0 errors", "note": note})

    # ---- Parameters ----------------------------------------------------------------------
    srv = env["server"]
    P = []

    def add(section, name, value, note=""):
        P.append({"section": section, "parameter": name, "value": value, "note": note})

    if cfg["server"]["mode"] != "docker":
        add("DEVIATION", "Not run in Docker", cfg["server"]["deviation_note"],
            "Same Ollama version as the pinned image; see the Server rows for exact values")
    add("Run", "run_id", env["run_id"])
    add("Run", "timestamp_utc", env["timestamp_utc"])
    add("Run", "git_commit", env["git_commit"], f"branch {env['git_branch']}, "
        f"uncommitted changes: {env['git_dirty']}")
    add("Run", "config file", env["config_path"])
    add("Run", "config_sha256", env["config_sha256"], "hash of the pinned config before overrides")
    add("Run", "command-line overrides", str(env["cli_overrides"] or "none"),
        "anything listed here differs from the pinned config")
    add("Server", "backend type", cfg["backend"]["type"])
    add("Server", "base_url (OpenAI-compatible)", cfg["backend"]["base_url"])
    add("Server", "mode", srv.get("mode"))
    add("Server", "Ollama version (reported by server)", env["server_version_reported"])
    for k in ("binary_path", "binary_sha256", "release_asset_sha256_pinned", "image",
              "docker_image_digest"):
        if srv.get(k):
            add("Server", k, srv[k])
    for k, v in sorted(env["server_env_set_by_harness"].items()):
        add("Server env (speed phase)", k, v, "set explicitly by the harness")
    add("Server env (speed phase)", "full effective settings (from server log)",
        env["server_effective_config_logged"], "everything not listed above was left at default")
    for k, v in cfg["request"].items():
        add("Request", k, str(v))
    add("Sweep", "concurrency_levels", " ".join(map(str, cfg["sweep"]["concurrency_levels"])))
    add("Sweep", "reps", cfg["sweep"]["reps"])
    add("Sweep", "warmup_max_tokens", cfg["sweep"]["warmup_max_tokens"], "warm-up is discarded")
    add("Sweep", "order", "per model: load, warm-up, each prompt x each level x reps, unload")
    add("Prompts", "file", cfg["prompts"]["path"])
    add("Prompts", "sha256", cfg["prompts"]["sha256"])
    for k, v in thr.items():
        add("Thresholds", k, v)
    add("Quantization", "all models", cfg["quantization"])
    for m in env["models"]:
        add("Model", m["tag"], "manifest sha256:" + str(m.get("ollama_digest", "")),
            "verified against pin: " + str(m.get("ok", "n/a")))
        if m.get("model_blob_sha256_on_disk"):
            add("Model", m["tag"] + " weights file", "sha256:" + m["model_blob_sha256_on_disk"],
                "hashed on disk")
    for m in run.get("models", []):
        for ps in m.get("ps_after_load", []):
            add("Model", m["tag"] + " loaded", f"size {ps['size']} bytes, on GPU "
                f"{ps['size_vram']} bytes, context {ps['context_length']}",
                f"load + warm-up {m.get('load_plus_warmup_s')} s")
    for m in run.get("models", []):
        for name, v in (m.get("thinking_check") or {}).items():
            add("Thinking check", f"{m['tag']}: {name}",
                f"reasoning chars {v['reasoning_chars']}, answer chars {v['content_chars']}"
                + (f", error {v['error']}" if v["error"] else ""),
                "discarded 64-token probe; reasoning chars > 0 means thinking was on")
    add("Power", "source", cfg["power"]["source"])
    add("Power", "interval_s", cfg["power"]["interval_s"])
    add("Power", "idle GPU power (W)", r(idle_w), "mean over the idle baseline before loading")
    add("Power", "wall meter", "merged" if wall.t else "not provided",
        "wall_meter_w column is empty until --wall-csv is given")
    q = cfg["quality"]
    add("Quality", "tasks", ", ".join(q["tasks"]))
    add("Quality", "limit per task", q.get("limit") or "none (full)")
    add("Quality", "gen_kwargs", q["gen_kwargs"])
    add("Quality", "server env for quality phase", str(q.get("server_env")),
        "server is restarted with these before lm-eval")
    add("Quality", "lm-eval version", common.sh(
        f"{common.REPO}/.venv/bin/python -c 'import lm_eval; print(lm_eval.__version__)'"))

    # ---- Quality / Environment / cross-check -----------------------------------------------
    quality = []
    for qr in run.get("quality", []):
        for x in qr["results"]:
            quality.append({"model": qr["model_key"], "task": x["task"], "metric": x["metric"],
                            "value": r(x["value"], 4), "stderr": r(x["stderr"], 4)
                            if isinstance(x["stderr"], float) else x["stderr"],
                            "n_samples": x["n"], "limit": qr["limit"], "status": "ran"})
        for s in qr["tasks_skipped"]:
            quality.append({"model": qr["model_key"], "task": s["task"], "metric": "", "value": "",
                            "stderr": "", "n_samples": "", "limit": qr["limit"],
                            "status": "SKIPPED: " + s["reason"]})
        if qr.get("returncode") not in (0, None):
            quality.append({"model": qr["model_key"], "task": "(lm-eval)", "metric": "",
                            "value": "", "stderr": "", "n_samples": "", "limit": qr["limit"],
                            "status": f"lm-eval exited with code {qr['returncode']}; see log"})
    environment = []
    for k, v in env.items():
        if k in ("models", "server_env_set_by_harness"):
            continue
        if isinstance(v, dict):
            for k2, v2 in v.items():
                environment.append({"key": f"{k}.{k2}", "value": str(v2)})
        else:
            environment.append({"key": k, "value": str(v)})
    for p in run["phases"]:
        environment.append({"key": f"phase_seconds.{p['phase']}", "value": p["seconds"]})
    for x in xcheck:
        for k in list(x):
            if k.endswith(("_tps", "_mean", "_s")):
                x[k] = r(num(x[k]))

    sheets = [("Parameters", P), ("Summary", summary), ("Capacity", capacity),
              ("Quality", quality), ("Environment", environment),
              ("ServerTimingCheck", xcheck), ("Requests", reqs), ("Power", power)]
    out = run_dir / "report"
    out.mkdir(exist_ok=True)
    for name, rows in sheets:
        if rows:
            with open(out / f"{name}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    try:
        import openpyxl
        from openpyxl.styles import Alignment, Font, PatternFill
    except ImportError:
        print("openpyxl not installed: wrote CSVs only. Run with `uv run` to get findings.xlsx")
        return
    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, rows in sheets:
        ws = wb.create_sheet(name)
        if not rows:
            ws.append(["(no data in this run)"])
            continue
        cols = list(rows[0])
        ws.append(cols)
        for c in ws[1]:
            c.font = Font(bold=True)
            c.fill = PatternFill("solid", fgColor="DDDDDD")
        for row in rows:
            ws.append([num(row[c]) if name in ("Requests", "Power") and num(row[c]) is not None
                       else row[c] for c in cols])
        ws.freeze_panes = "A2"
        for i, c in enumerate(cols):
            width = max([len(str(c))] + [len(str(row[c])) for row in rows[:200]])
            ws.column_dimensions[openpyxl.utils.get_column_letter(i + 1)].width = \
                min(max(10, width + 2), 70)
        if name == "Parameters":
            for row in ws.iter_rows(min_row=2):
                for c in row:
                    c.alignment = Alignment(wrap_text=True, vertical="top")
                if row[0].value == "DEVIATION":
                    for c in row:
                        c.font = Font(bold=True)
                        c.fill = PatternFill("solid", fgColor="FFE699")
    wb.save(run_dir / "findings.xlsx")
    print(f"wrote {run_dir / 'findings.xlsx'} and {out}/*.csv")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--wall-csv")
    a = ap.parse_args()
    build(a.run_dir, a.wall_csv)
