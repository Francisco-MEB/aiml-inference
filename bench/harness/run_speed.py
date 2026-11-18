"""Speed / concurrency sweep for one model. Talks only to the OpenAI-compatible base URL;
the Ollama-specific parts (ps, unload, native cross-check) are skipped for other backends.
Design follows scripts/ollama_concurrency_probe.py: N requests released together, one CSV row
per request, aggregate tok/s per batch."""
import threading
import time

import common
import protocol

REQ_FIELDS = [*protocol.ROW_FIELDS, "run_id", "model_key", "model_tag", "prompt_id", "prompt_class", "concurrency",
              "rep", "req", "batch_start_unix", "batch_end_unix", "batch_wall_s", "batch_agg_tps",
              "req_start_unix", "ttft_s", "e2e_s", "prompt_tokens", "output_tokens", "decode_tps",
              "prefill_tps_est", "reasoning_chars", "content_chars", "finish_reason", "error"]
XCHECK_FIELDS = ["run_id", "model_key", "prompt_id", "prompt_tokens", "output_tokens",
                 "native_prefill_tps", "native_decode_tps", "native_load_s",
                 "client_decode_tps_n1_mean", "client_prefill_tps_est_n1_mean", "thinking_chars"]


def run_batch(base_url, model, messages, req, n, tag):
    results = [None] * n
    go = threading.Event()

    def worker(i):
        msgs = common.with_nonce(messages, common.nonce(tag, i))
        go.wait()
        results[i] = common.chat_stream(base_url, model, msgs, req)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(n)]
    for t in threads:
        t.start()
    t0u, t0 = time.time(), time.perf_counter()
    go.set()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    total = sum(r["output_tokens"] or 0 for r in results)
    return results, t0u, t0u + wall, wall, (total / wall if wall > 0 else None)


def native_crosscheck(cfg, model, prompt, client_rows):
    """Ollama only: one non-streamed native request, so the client-side rates can be compared
    with the server's own eval_duration / prompt_eval_duration timings."""
    req = cfg["request"]
    body = {"model": model["tag"], "stream": False, "think": False,
            "messages": common.with_nonce(prompt["messages"],
                                          common.nonce(model["key"], prompt["id"], "xcheck")),
            "options": {"temperature": req["temperature"], "seed": req["seed"],
                        "num_predict": req["max_tokens"]}}
    r = common.http_json(cfg["backend"]["native_url"] + "/api/chat", body,
                         timeout=req.get("timeout_s", 900))

    def rate(c, d):
        return r[c] / (r[d] / 1e9) if r.get(c) and r.get(d) else None

    n1 = [x for x in client_rows if x["concurrency"] == 1 and x["prompt_id"] == prompt["id"]]
    return {"model_key": model["key"], "prompt_id": prompt["id"],
            "prompt_tokens": r.get("prompt_eval_count"), "output_tokens": r.get("eval_count"),
            "native_prefill_tps": rate("prompt_eval_count", "prompt_eval_duration"),
            "native_decode_tps": rate("eval_count", "eval_duration"),
            "native_load_s": (r.get("load_duration") or 0) / 1e9,
            "client_decode_tps_n1_mean": common.mean([x["decode_tps"] for x in n1]),
            "client_prefill_tps_est_n1_mean": common.mean([x["prefill_tps_est"] for x in n1]),
            "thinking_chars": len((r.get("message") or {}).get("thinking") or "")}


def unload(cfg, model):
    native = cfg["backend"]["native_url"]
    common.http_json(native + "/api/generate", {"model": model["tag"], "keep_alive": 0},
                     timeout=120)
    for _ in range(60):
        if not common.loaded_models(native):
            return True
        time.sleep(0.5)
    return False


def run_model(cfg, model, prompts, run_id, req_writer, xcheck_writer, log=print, row_meta=None):
    """row_meta: protocol columns written into every request row (see protocol.ROW_FIELDS).
    sweep.level_cap_s, if set, stops starting new batches of a workload level after that long;
    info["capped"] lists the levels that were cut short."""
    base, req, sweep = cfg["backend"]["base_url"], cfg["request"], cfg["sweep"]
    if "extra_body" in model:          # per-model way to turn thinking off
        req = dict(req, extra_body=model["extra_body"])
    row_meta = row_meta or {}
    cap_s = sweep.get("level_cap_s")
    is_ollama = cfg["backend"]["type"] == "ollama"
    info = {"key": model["key"], "tag": model["tag"]}

    # Load + discarded warm-up request
    warm = dict(req, max_tokens=sweep["warmup_max_tokens"])
    t0 = time.time()
    res, *_ = run_batch(base, model["tag"], prompts[0]["messages"], warm, 1,
                        common.nonce(model["key"], "warmup"))
    info["load_plus_warmup_s"] = round(time.time() - t0, 2)
    if res[0]["error"]:
        raise SystemExit(f"warm-up failed for {model['tag']}: {res[0]['error']}")
    if is_ollama:
        ps = common.loaded_models(cfg["backend"]["native_url"]) or []
        info["ps_after_load"] = [{k: m.get(k) for k in
                                  ("name", "size", "size_vram", "context_length")} for m in ps]
        for m in ps:
            pct = 100 * (m.get("size_vram") or 0) / (m.get("size") or 1)
            log(f"  loaded {m.get('name')} {m.get('size', 0) / 1e9:.1f} GB, "
                f"{pct:.0f}% on GPU, ctx {m.get('context_length')}")

    if is_ollama:
        info["fits_gpu"] = all((m.get("size_vram") or 0) >= (m.get("size") or 1)
                               for m in info["ps_after_load"]) and bool(info["ps_after_load"])
    if sweep.get("warmup_all_slots") and max(sweep["concurrency_levels"]) > 1:
        # Discarded: the first use of each parallel slot has a one-off cost.
        run_batch(base, model["tag"], prompts[0]["messages"], warm,
                  max(sweep["concurrency_levels"]), common.nonce(model["key"], "warmup-all"))

    # Thinking switch check (discarded): shows what the pinned setting changes on this server.
    info["thinking_check"] = {}
    for name, extra in (("pinned_setting", req.get("extra_body") or {}),
                        ("server_default_no_setting", {}),
                        ("reasoning_effort_high", {"reasoning_effort": "high"})):
        probe = dict(req, max_tokens=64, extra_body=extra)
        r0, *_ = run_batch(base, model["tag"], prompts[0]["messages"], probe, 1,
                           common.nonce(model["key"], "think", name))
        info["thinking_check"][name] = {"reasoning_chars": r0[0]["reasoning_chars"],
                                        "content_chars": r0[0]["content_chars"],
                                        "error": r0[0]["error"]}
    log(f"  thinking check: {info['thinking_check']}")

    rows, info["capped"] = [], []
    for p in prompts:
        for n in sweep["concurrency_levels"]:
            t_level = time.time()
            for rep in range(sweep["reps"]):
                if cap_s and rep > 0 and time.time() - t_level > cap_s:
                    info["capped"].append({"workload": p["id"], "users": n, "reps_done": rep})
                    log(f"  {p['id']} N={n}: level time cap reached after {rep} batches")
                    break
                tag = common.nonce(run_id, model["key"], p["id"], n, rep)
                res, t0u, t1u, wall, agg = run_batch(base, model["tag"], p["messages"], req, n,
                                                     tag)
                for i, r in enumerate(res):
                    row = {**row_meta, "run_id": run_id, "model_key": model["key"], "model_tag": model["tag"],
                           "prompt_id": p["id"], "prompt_class": p["class"], "concurrency": n,
                           "rep": rep, "req": i, "batch_start_unix": round(t0u, 3),
                           "batch_end_unix": round(t1u, 3), "batch_wall_s": wall,
                           "batch_agg_tps": agg, **r}
                    row["req_start_unix"] = round(row["req_start_unix"], 3)
                    req_writer.writerow({k: row.get(k) for k in REQ_FIELDS})
                    rows.append(row)
            lvl = [r for r in rows if r["prompt_id"] == p["id"] and r["concurrency"] == n]
            errs = sum(1 for r in lvl if r["error"])
            ttft = common.percentile([r["ttft_s"] for r in lvl], 95)
            dec = common.mean([r["decode_tps"] for r in lvl])
            log(f"  {p['id']:<13} N={n:<3} p95 TTFT {ttft if ttft is None else round(ttft, 2)} s, "
                f"decode {dec if dec is None else round(dec, 1)} tok/s/req, errors {errs}")
    info["reasoning_chars_total"] = sum(r["reasoning_chars"] for r in rows)
    info["errors"] = sum(1 for r in rows if r["error"])

    if is_ollama:
        for p in prompts:
            try:
                x = native_crosscheck(cfg, model, p, rows)
                xcheck_writer.writerow({"run_id": run_id, **x})
            except Exception as e:  # noqa: BLE001
                log(f"  native cross-check failed for {p['id']}: {e!r}")
        info["unloaded"] = unload(cfg, model)
        log(f"  unloaded: {info['unloaded']}")
    return info
