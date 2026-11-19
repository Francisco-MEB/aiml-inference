"""Speed / concurrency sweep for one model on an already-running server (engines.Server).
Talks only to the OpenAI-compatible API, so it is the same for vLLM and MLX. Design follows
scripts/ollama_concurrency_probe.py: N requests released together, one CSV row per request,
aggregate tok/s per batch."""
import threading
import time

import common
import engines
import protocol

REQ_FIELDS = [*protocol.ROW_FIELDS, "run_id", "model_key", "model_tag", "prompt_id", "prompt_class", "concurrency",
              "rep", "req", "batch_start_unix", "batch_end_unix", "batch_wall_s", "batch_agg_tps",
              "req_start_unix", "ttft_s", "e2e_s", "prompt_tokens", "output_tokens", "decode_tps",
              "prefill_tps_est", "reasoning_chars", "content_chars", "finish_reason", "error"]
# Ollama-only server-timing cross-check; kept so 1.0 studies still merge and report.
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


def run_model(cfg, model, prompts, run_id, req_writer, log=print, row_meta=None):
    """row_meta: protocol columns written into every request row (see protocol.ROW_FIELDS).
    sweep.level_cap_s, if set, stops starting new batches of a workload level after that long;
    info["capped"] lists the levels that were cut short."""
    base = engines.base_url(cfg) + "/v1"
    name = engines.request_model(cfg, model)
    req, sweep = dict(cfg["request"]), cfg["sweep"]
    req["extra_body"] = dict(model.get("extra_body") or {})   # per-model thinking setting
    row_meta = row_meta or {}
    cap_s = sweep.get("level_cap_s")
    info = {"key": model["key"], "tag": model["tag"], "fits_gpu": True}

    # Discarded warm-up request (vLLM compiles kernels on the first request)
    warm = dict(req, max_tokens=sweep["warmup_max_tokens"])
    t0 = time.time()
    res, *_ = run_batch(base, name, prompts[0]["messages"], warm, 1,
                        common.nonce(model["key"], "warmup"))
    info["load_plus_warmup_s"] = round(time.time() - t0, 2)
    if res[0]["error"]:
        raise SystemExit(f"warm-up failed for {model['tag']}: {res[0]['error']}")
    if sweep.get("warmup_all_slots") and max(sweep["concurrency_levels"]) > 1:
        # Discarded: the first use of each parallel slot has a one-off cost.
        run_batch(base, name, prompts[0]["messages"], warm,
                  max(sweep["concurrency_levels"]), common.nonce(model["key"], "warmup-all"))

    # Thinking check (discarded): does the pinned setting really keep reasoning text out?
    probe = dict(req, max_tokens=64)
    r0, *_ = run_batch(base, name, prompts[0]["messages"], probe, 1,
                       common.nonce(model["key"], "think", "pinned"))
    info["thinking_check"] = {"pinned_setting": {"reasoning_chars": r0[0]["reasoning_chars"],
                                                 "content_chars": r0[0]["content_chars"],
                                                 "error": r0[0]["error"]}}
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
                res, t0u, t1u, wall, agg = run_batch(base, name, p["messages"], req, n, tag)
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

    return info
