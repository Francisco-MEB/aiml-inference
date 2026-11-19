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

SCALING_NOTE = ("Limits: one machine, one model loaded at a time, GPU-only power, concurrency "
                "above the highest tested level not measured. Multiplying by a number of "
                "machines assumes independent nodes with no shared bottleneck.")
METRICS = [("aggregate_tps_mean", "Aggregate output tok/s (all users together)"),
           ("decode_tps_per_req_median", "Per-user decode tok/s (median request)"),
           ("ttft_p95_s", "p95 time to first token (s)")]


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


def decode_stat(s, stat):
    return s[{"median": "decode_tps_per_req_median", "mean": "decode_tps_per_req_mean",
              "p05": "decode_tps_per_req_p05", "min": "decode_tps_per_req_min"}[stat]]


def capacity_for(levels, ttft_max, decode_min, stat):
    """levels: Summary rows of one model+prompt, sorted by concurrency. Capacity is the highest
    level that passes with every lower level passing too."""
    cap, limit, higher_pass = 0, "", False
    failed = False
    for s in levels:
        why = []
        if s["errors"]:
            why.append("errors")
        if s["ttft_p95_s"] is None or s["ttft_p95_s"] > ttft_max:
            why.append("TTFT")
        d = decode_stat(s, stat)
        if d is None or d < decode_min:
            why.append("decode speed")
        if not why and not failed:
            cap = s["concurrency"]
        elif not why:
            higher_pass = True
        elif not failed:
            failed, limit = True, f"{' and '.join(why)} at {s['concurrency']} users"
    if not failed:
        limit = "not reached: passes at the highest level tested"
    return cap, limit, higher_pass


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
    order = [m["key"] for m in cfg["models"]]
    prompt_order = list(dict.fromkeys(q["prompt_class"] for q in reqs))

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
        aggs = [b[2] for b in batches.values() if b[2] is not None]
        toks = sum(int(num(q["output_tokens"]) or 0) for q in rows)
        wall_s = sum(b[1] - b[0] for b in batches.values())
        e_gpu = sum(gpu.energy(b[0], b[1]) or 0 for b in batches.values()) if gpu.t else None
        e_wall = sum(wall.energy(b[0], b[1]) or 0 for b in batches.values()) if wall.t else None
        ttft = [num(q["ttft_s"]) for q in rows]
        dec = [num(q["decode_tps"]) for q in rows]
        decv = [d for d in dec if d is not None]
        summary.append({
            "model": mk, "prompt_class": pc, "concurrency": n, "n_requests": len(rows),
            "n_batches": len(batches), "errors": sum(1 for q in rows if q["error"]),
            "prompt_tokens_mean": r(common.mean([num(q["prompt_tokens"]) for q in rows]), 0),
            "output_tokens_mean": r(common.mean([num(q["output_tokens"]) for q in rows]), 1),
            "ttft_mean_s": r(common.mean(ttft), 3), "ttft_p50_s": r(common.percentile(ttft, 50), 3),
            "ttft_p95_s": r(common.percentile(ttft, 95), 3),
            "ttft_max_s": r(max([t for t in ttft if t is not None], default=None), 3),
            "e2e_mean_s": r(common.mean([num(q["e2e_s"]) for q in rows]), 2),
            "decode_tps_per_req_mean": r(common.mean(dec)),
            "decode_tps_per_req_median": r(common.percentile(dec, 50)),
            "decode_tps_per_req_p05": r(common.percentile(dec, 5)),
            "decode_tps_per_req_min": r(min(decv, default=None)),
            "prefill_tps_est_mean": r(common.mean([num(q["prefill_tps_est"]) for q in rows])),
            "prefill_note": "valid (no queueing)" if n == 1 else "includes queueing; use N=1",
            "aggregate_tps_mean": r(common.mean(aggs)),
            "aggregate_tps_min": r(min(aggs, default=None)),
            "aggregate_tps_max": r(max(aggs, default=None)),
            "scaling_efficiency": None,
            "output_tokens_per_hour": None,
            "gpu_power_avg_w": r(e_gpu / wall_s) if e_gpu and wall_s else None,
            "gpu_joules_per_output_token": r(e_gpu / toks, 3) if e_gpu and toks else None,
            "wall_power_avg_w": r(e_wall / wall_s) if e_wall and wall_s else None,
            "wall_joules_per_output_token": r(e_wall / toks, 3) if e_wall and toks else None,
            "reasoning_chars": sum(int(num(q["reasoning_chars"]) or 0) for q in rows),
        })
    summary.sort(key=lambda s: (order.index(s["model"]) if s["model"] in order else 99,
                                prompt_order.index(s["prompt_class"]), s["concurrency"]))
    base = {(s["model"], s["prompt_class"]): s["aggregate_tps_mean"]
            for s in summary if s["concurrency"] == 1}
    for s in summary:
        b = base.get((s["model"], s["prompt_class"]))
        if b and s["aggregate_tps_mean"]:
            # 1.0 = N users each get what a single user gets
            s["scaling_efficiency"] = r(s["aggregate_tps_mean"] / (s["concurrency"] * b), 3)
            s["output_tokens_per_hour"] = round(s["aggregate_tps_mean"] * 3600)
    models_run = list(dict.fromkeys(s["model"] for s in summary))

    def levels_of(mk, pc):
        return [s for s in summary if s["model"] == mk and s["prompt_class"] == pc]

    # ---- Capacity (long) + pivot ---------------------------------------------------------
    scenarios = thr.get("scenarios") or [{"name": "pinned", "p95_ttft_s_max": thr["p95_ttft_s_max"],
                                          "decode_tps_min": thr["decode_tps_min"]}]
    stats = thr.get("decode_stats") or [thr["decode_stat"]]
    capacity = []
    for mk in models_run:
        for pc in prompt_order:
            lv = levels_of(mk, pc)
            if not lv:
                continue
            for sc in scenarios:
                for st in stats:
                    cap, limit, higher = capacity_for(lv, sc["p95_ttft_s_max"],
                                                      sc["decode_tps_min"], st)
                    capacity.append({
                        "model": mk, "prompt_class": pc, "scenario": sc["name"],
                        "p95_ttft_s_max": sc["p95_ttft_s_max"],
                        "decode_tps_min": sc["decode_tps_min"], "decode_statistic": st,
                        "user_capacity": cap, "limited_by": limit,
                        "levels_tested": " ".join(str(s["concurrency"]) for s in lv),
                        "note": "a higher level passed after a lower one failed" if higher else ""})
    pinned = scenarios[0]["name"]
    title = (f"User capacity per machine: {pinned}, p95 TTFT <= {scenarios[0]['p95_ttft_s_max']} "
             f"s and {thr['decode_stat']} per-user decode >= {scenarios[0]['decode_tps_min']} "
             f"tok/s. Levels tested: {' '.join(map(str, cfg['sweep']['concurrency_levels']))}.")
    pivot = [[title], ["model", *prompt_order]]
    for mk in models_run:
        row = [mk]
        for pc in prompt_order:
            c = next((c for c in capacity if c["model"] == mk and c["prompt_class"] == pc
                      and c["scenario"] == pinned and c["decode_statistic"] == thr["decode_stat"]),
                     None)
            top = c and c["limited_by"].startswith("not reached")
            row.append("" if c is None else (f">={c['user_capacity']}" if top
                                             else c["user_capacity"]))
        pivot.append(row)
    pivot += [[], ["All scenarios and statistics (long format):"]]

    # ---- Models --------------------------------------------------------------------------
    info = {m["key"]: m for m in run.get("models", [])}
    pulled = {m["key"]: m for m in env["models"]}
    models_sheet = []
    for m in cfg["models"]:
        mk = m["key"]
        if mk not in models_run:
            continue
        rows = [s for s in summary if s["model"] == mk]
        n1 = [s for s in rows if s["concurrency"] == 1]
        peak = max(rows, key=lambda s: s["aggregate_tps_mean"] or 0)
        longest = max(n1, key=lambda s: s["prompt_tokens_mean"] or 0, default={})
        ps = (info.get(mk, {}).get("ps_after_load") or [{}])[0]      # 1.0 (Ollama) runs
        sv = info.get(mk, {}).get("server") or {}                     # 2.0 runs
        loaded = sv.get("weights_loaded_gib")
        row = {
            "model": mk, "label": m.get("label"), "architecture": m.get("arch"),
            "params_billion": m.get("params_b"),
            "active_params_billion": m.get("active_params_b", m.get("params_b")),
            "tag": m["tag"], "revision": m.get("revision", ""),
            "quantization": m.get("quantization") or cfg["quantization"],
            "bits_per_weight": m.get("bits_per_weight"),
            "download_size_gb": r((pulled.get(mk, {}).get("size_bytes") or 0) / 1e9),
            "memory_loaded_gb": r(loaded * 2**30 / 1e9) if loaded else
            r(ps["size"] / 1e9) if ps.get("size") else "not recorded",
            "kv_cache_tokens": sv.get("kv_cache_tokens") or sv.get("kv_cache_tokens_logged"),
            "context_per_request": sv.get("context") or ps.get("context_length"),
            "server_startup_s": sv.get("startup_s"),
            "load_plus_warmup_s": info.get(mk, {}).get("load_plus_warmup_s"),
            "single_user_decode_tps": r(common.mean([s["decode_tps_per_req_mean"] for s in n1])),
            "single_user_prefill_tps": longest.get("prefill_tps_est_mean"),
            "single_user_prefill_prompt_tokens": longest.get("prompt_tokens_mean"),
            "peak_aggregate_tps": peak["aggregate_tps_mean"],
            "peak_at_users": peak["concurrency"], "peak_prompt_class": peak["prompt_class"],
            "peak_output_tokens_per_hour": peak["output_tokens_per_hour"],
            "gpu_power_at_peak_w": peak["gpu_power_avg_w"],
            "gpu_joules_per_token_at_peak": peak["gpu_joules_per_output_token"],
        }
        for pc in prompt_order:
            row[f"capacity_{pc}"] = next((p[1 + prompt_order.index(pc)] for p in pivot[2:]
                                          if p and p[0] == mk), "")
        models_sheet.append(row)

    # ---- Scaling (blocks: one per prompt class and metric; rows = users, cols = models) ---
    scaling = []
    for pc in prompt_order:
        for key, title in METRICS:
            for n in sorted({s["concurrency"] for s in summary if s["prompt_class"] == pc}):
                row = {"prompt_class": pc, "metric": title, "concurrent_users": n}
                for mk in models_run:
                    row[mk] = next((s[key] for s in levels_of(mk, pc) if s["concurrency"] == n),
                                   None)
                scaling.append(row)

    # ---- Parameters ----------------------------------------------------------------------
    srv = env.get("server") or {}
    P = []

    def add(section, name, value, note=""):
        P.append({"section": section, "parameter": name, "value": value, "note": note})

    if cfg["server"].get("mode") not in (None, "docker"):        # 1.0 (Ollama) runs
        add("DEVIATION", "Not run in Docker", cfg["server"].get("deviation_note", ""),
            "Same Ollama version as the pinned image; see the Server rows for exact values")
    if cfg.get("quantization_deviation"):
        add("DEVIATION", "Quantization", cfg["quantization_deviation"],
            f"build publisher: {cfg.get('quantization_publisher', '')}")
    add("Protocol", "version", env.get("protocol_version", "pre-1.0"),
        "bench/benchmark-protocol.md; results line up only within the same version")
    if env.get("study"):
        st = env["study"]
        add("Protocol", "study", env.get("study_id", ""), st.get("notes", ""))
        add("Protocol", "member", st.get("member", ""))
        add("Protocol", "machine", st.get("machine", ""), st.get("hardware", ""))
    add("Protocol", "QUANTIZATION (study build format)", env.get("quantization") or cfg["quantization"],
        "compare studies with different quantizations only with this in mind; publisher "
        + str(cfg.get("quantization_publisher", "")))
    if env.get("protocol_deviations"):
        add("Protocol", "DEVIATIONS", "; ".join(env["protocol_deviations"]),
            "this run does not follow the protocol")
    add("Run", "run_id", env["run_id"])
    add("Run", "timestamp_utc", env["timestamp_utc"])
    add("Run", "total run time (min)", r(run.get("total_seconds", 0) / 60, 1))
    add("Run", "git_commit", env["git_commit"], f"branch {env['git_branch']}, "
        f"uncommitted changes: {env['git_dirty']}")
    add("Run", "config file", env["config_path"])
    add("Run", "config_sha256", env["config_sha256"], "hash of the pinned config before overrides")
    add("Run", "command-line overrides", str(env["cli_overrides"] or "none"),
        "anything listed here differs from the pinned config")
    add("Run", "models run", " ".join(models_run))
    for s in run.get("models_skipped", []):
        add("Run", f"model NOT run: {s['key']}", s["reason"])
    add("Run", "other GPU processes at start", str(env.get("other_gpu_processes_at_start") or "none"))
    add("Server", "engine", env.get("engine") or cfg["backend"]["type"])
    if env.get("slots") not in (None, ""):
        add("Server", "parallel slots (= users)", env["slots"], "server restarted per level")
    for sv in env.get("servers") or []:
        tag = f"{sv.get('model')} {sv.get('phase', 'speed')}"
        add("Server", f"{tag}: version", sv.get("engine_version", ""),
            f"ready in {sv.get('startup_s')} s")
        add("Server", f"{tag}: command", " ".join(sv.get("command") or []),
            "every server flag; the rest are engine defaults")
        for k in ("kv_cache_dtype", "kv_cache_tokens", "kv_cache_tokens_logged",
                  "weights_loaded_gib", "kv_cache_available_gib", "max_concurrency_logged",
                  "prefix_caching"):
            if sv.get(k) not in (None, ""):
                add("Server", f"{tag}: {k}", sv[k])
    if env.get("server_version_reported"):                       # 1.0 (Ollama) runs
        add("Server", "Ollama version (reported by server)", env["server_version_reported"])
    for k in ("binary_path", "binary_sha256", "release_asset_sha256_pinned", "image",
              "docker_image_digest"):
        if srv.get(k):
            add("Server", k, srv[k])
    for k, v in sorted((env.get("server_env_set_by_harness") or {}).items()):
        add("Server env (speed phase)", k, v, "set explicitly by the harness")
    for k, v in cfg["request"].items():
        add("Request", k, str(v))
    add("Sweep", "concurrency_levels", " ".join(map(str, cfg["sweep"]["concurrency_levels"])))
    add("Sweep", "reps", cfg["sweep"]["reps"], "batches per level; a batch is N simultaneous requests")
    add("Sweep", "warmup_max_tokens", cfg["sweep"]["warmup_max_tokens"], "warm-up is discarded")
    add("Sweep", "warmup_all_slots", str(cfg["sweep"].get("warmup_all_slots")),
        "one discarded batch at the highest level")
    add("Sweep", "order", "per model: load, warm-up, each prompt x each level x reps, unload")
    add("Prompts", "file", cfg["prompts"]["path"])
    add("Prompts", "sha256", cfg["prompts"]["sha256"])
    add("Thresholds", "pinned rule", f"p95 TTFT <= {thr['p95_ttft_s_max']} s and "
        f"{thr['decode_stat']} per-request decode >= {thr['decode_tps_min']} tok/s, 0 errors")
    for sc in scenarios:
        add("Thresholds", f"scenario: {sc['name']}",
            f"p95 TTFT <= {sc['p95_ttft_s_max']} s, decode >= {sc['decode_tps_min']} tok/s")
    add("Thresholds", "decode statistics reported", ", ".join(stats),
        "p05 = the slowest 5% of requests are below this")
    add("Thresholds", "capacity definition",
        "highest tested level that passes, with every lower level passing too")
    add("Quantization", "all models", cfg["quantization"])
    for m in env["models"]:
        if m.get("revision"):
            add("Model", m["tag"], f"revision {m['revision']}",
                f"{m.get('pinned_weight_files')} weight files verified against pinned sha256: "
                f"{m.get('ok')} (rehashed: {m.get('rehashed')})")
        else:                                                    # 1.0 (Ollama) runs
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
    if run.get("quality"):
        add("Quality", "tasks", ", ".join(q["tasks"]))
        add("Quality", "limit per task", q.get("limit") or str(q.get("limits") or "none (full)"))
        add("Quality", "gen_kwargs", q["gen_kwargs"])
        add("Quality", "quality server", f"{q.get('slots')} slots, context {q.get('context')}, "
            f"max_gen_toks {q.get('max_gen_toks')}", "server restarted for the quality phase")
        add("Quality", "lm-eval version", common.sh(
            f"{common.REPO}/.venv/bin/python -c 'import lm_eval; print(lm_eval.__version__)'"))
    else:
        add("Quality", "output-quality evaluation", "not run", "speed and capacity only")

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
        if k in ("models", "server_env_set_by_harness", "servers"):
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
        c, n = num(x.get("client_decode_tps_n1_mean")), num(x.get("native_decode_tps"))
        x["client_vs_server_decode_pct"] = r(100 * (c - n) / n, 1) if c and n else None

    timing = read_csv(run_dir / "timing.csv")
    sheets = [("Parameters", P), ("Models", models_sheet), ("Summary", summary),
              ("Capacity", capacity), ("Scaling", scaling), ("Timing", timing),
              ("Quality", quality),
              ("ServerTimingCheck", xcheck), ("Environment", environment),
              ("Requests", reqs), ("Power", power)]
    sheets = [(n, rows) for n, rows in sheets if rows or n not in ("Quality", "Timing")]
    out = run_dir / "report"
    out.mkdir(exist_ok=True)
    for name, rows in sheets:
        if rows:
            with open(out / f"{name}.csv", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    with open(out / "CapacityPivot.csv", "w", newline="") as f:
        csv.writer(f).writerows(pivot[1:2 + len(models_run)])
    try:
        import openpyxl
        from openpyxl.chart import LineChart, Reference
        from openpyxl.styles import Alignment, Font, PatternFill
    except ImportError:
        print("openpyxl not installed: wrote CSVs only. Run with `uv run` to get findings.xlsx")
        return
    bold, grey = Font(bold=True), PatternFill("solid", fgColor="DDDDDD")

    def header(ws, row):
        for c in ws[row]:
            c.font, c.fill = bold, grey

    def widths(ws, cols, rows):
        for i, c in enumerate(cols):
            width = max([len(str(c))] + [len(str(row[c])) for row in rows[:200]])
            ws.column_dimensions[openpyxl.utils.get_column_letter(i + 1)].width = \
                min(max(10, width + 2), 70)

    wb = openpyxl.Workbook()
    wb.remove(wb.active)
    for name, rows in sheets:
        ws = wb.create_sheet(name)
        if not rows:
            ws.append(["(no data in this run)"])
            continue
        cols = list(rows[0])
        if name == "Scaling":
            ws.append([SCALING_NOTE])
            ws["A1"].font = bold
            ws.append([])
            anchor_row = 3
            for pc in prompt_order:
                for _key, title in METRICS:
                    block = [x for x in rows if x["prompt_class"] == pc and x["metric"] == title]
                    ws.append([f"{title}: {pc}"])
                    ws.cell(ws.max_row, 1).font = bold
                    ws.append(["concurrent users", *models_run])
                    hdr = ws.max_row
                    header(ws, hdr)
                    for x in block:
                        ws.append([x["concurrent_users"], *[x[m] for m in models_run]])
                    last = ws.max_row
                    ws.append([])
                    ch = LineChart()
                    ch.title = f"{title}: {pc}"
                    ch.x_axis.title, ch.y_axis.title = "Concurrent users", title.split(" (")[0]
                    ch.x_axis.delete = ch.y_axis.delete = False
                    ch.height, ch.width = 7.5, 15
                    ch.add_data(Reference(ws, min_col=2, max_col=1 + len(models_run),
                                          min_row=hdr, max_row=last), titles_from_data=True)
                    ch.set_categories(Reference(ws, min_col=1, min_row=hdr + 1, max_row=last))
                    for s in ch.series:
                        s.marker.symbol = "circle"
                    col = "H" if title == METRICS[0][1] else ("R" if title == METRICS[1][1]
                                                              else "AB")
                    ws.add_chart(ch, f"{col}{anchor_row}")
                anchor_row += 16
            ws.column_dimensions["A"].width = 18
            continue
        if name == "Capacity":
            for prow in pivot:
                ws.append(prow)
            ws["A1"].font = bold
            header(ws, 2)
            ws.cell(ws.max_row, 1).font = bold
        ws.append(cols)
        hdr = ws.max_row
        header(ws, hdr)
        for row in rows:
            ws.append([num(row[c]) if name in ("Requests", "Power") and num(row[c]) is not None
                       else row[c] for c in cols])
        ws.freeze_panes = ws.cell(hdr + 1, 1) if name != "Capacity" else None
        widths(ws, cols, rows)
        if name == "Parameters":
            for row in ws.iter_rows(min_row=2):
                for c in row:
                    c.alignment = Alignment(wrap_text=True, vertical="top")
                if row[0].value == "DEVIATION":
                    for c in row:
                        c.font = bold
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
