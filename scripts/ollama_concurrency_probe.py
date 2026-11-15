#!/usr/bin/env python3
"""
ollama_concurrency_probe.py

Measures how a single Ollama server behaves as the number of simultaneous
users grows. Stdlib only (no pip install needed), so it runs as-is on any node.

For each concurrency level N, it fires N identical requests at the same time,
repeats that R times, and records per request:
  - ttft_s        time to first token (client-side, includes queueing)
  - e2e_s         end-to-end latency (client-side)
  - decode_tps    output tokens/s from Ollama's own eval_count / eval_duration
  - prefill_tps   prompt tokens/s from prompt_eval_count / prompt_eval_duration
  - load_s        model load time reported by Ollama (should be ~0 after warmup)
and per batch:
  - agg_tps       total output tokens across the batch / batch wall time

Results go to a CSV (one row per request) plus a printed summary.

IMPORTANT: server-side concurrency is set by env vars on the *server*
(OLLAMA_NUM_PARALLEL, OLLAMA_MAX_LOADED_MODELS, OLLAMA_CONTEXT_LENGTH).
Record them with --server-note so the run is reproducible.

Example:
  python3 ollama_concurrency_probe.py --model qwen2.5:14b \
      --levels 1 2 4 8 --reps 5 --num-predict 256 \
      --server-note "NUM_PARALLEL=4 CTX=4096 ollama=0.x.y" \
      --out results_dell1.csv
"""
import argparse
import csv
import json
import platform
import statistics
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

DEFAULT_PROMPT = (
    "Write a detailed explanation of how a hash table handles collisions, "
    "covering chaining and open addressing, with their time complexities."
)


def get_json(url):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.loads(r.read())


def one_request(host, model, prompt, num_predict, seed, results, idx, start_evt):
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0,        # deterministic-ish output length
            "seed": seed,
            "num_predict": num_predict,
        },
    }).encode()
    req = urllib.request.Request(
        f"{host}/api/generate", data=body,
        headers={"Content-Type": "application/json"},
    )
    start_evt.wait()  # all threads start together
    t0 = time.perf_counter()
    ttft = None
    final = {}
    err = ""
    try:
        with urllib.request.urlopen(req, timeout=600) as r:
            for line in r:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                if ttft is None and chunk.get("response"):
                    ttft = time.perf_counter() - t0
                if chunk.get("done"):
                    final = chunk
    except urllib.error.HTTPError as e:  # e.g. 404 model not found, 503 queue full
        try:
            detail = e.read().decode(errors="replace").strip()
        except Exception:
            detail = ""
        err = f"HTTP {e.code}: {detail}"
    except Exception as e:
        err = repr(e)
    t1 = time.perf_counter()

    def rate(count_key, dur_key):
        c, d = final.get(count_key), final.get(dur_key)
        return c / (d / 1e9) if c and d else None

    results[idx] = {
        "t_start": t0,
        "t_end": t1,
        "ttft_s": ttft,
        "e2e_s": t1 - t0,
        "prompt_tokens": final.get("prompt_eval_count"),
        "output_tokens": final.get("eval_count"),
        "decode_tps": rate("eval_count", "eval_duration"),
        "prefill_tps": rate("prompt_eval_count", "prompt_eval_duration"),
        "load_s": (final.get("load_duration") or 0) / 1e9,
        "error": err,
    }


def run_batch(host, model, prompt, num_predict, n, seed):
    results = [None] * n
    start_evt = threading.Event()
    threads = [
        threading.Thread(target=one_request,
                         args=(host, model, prompt, num_predict, seed, results, i, start_evt))
        for i in range(n)
    ]
    for t in threads:
        t.start()
    start_evt.set()
    for t in threads:
        t.join()
    wall = max(r["t_end"] for r in results) - min(r["t_start"] for r in results)
    total_out = sum(r["output_tokens"] or 0 for r in results)
    return results, wall, total_out / wall if wall > 0 else None


def fmt(xs):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "n/a"
    med = statistics.median(xs)
    return f"{med:7.2f} (min {min(xs):.2f}, max {max(xs):.2f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--model", required=True)
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--num-predict", type=int, default=256)
    ap.add_argument("--prompt", default=DEFAULT_PROMPT)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--server-note", default="",
                    help="server env/config, e.g. 'NUM_PARALLEL=4 CTX=4096'")
    ap.add_argument("--out", default="ollama_probe.csv")
    args = ap.parse_args()

    version = get_json(f"{args.host}/api/version").get("version", "?")
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "client_host": platform.node(),
        "server": args.host,
        "ollama_version": version,
        "model": args.model,
        "num_predict": args.num_predict,
        "server_note": args.server_note,
    }
    print("# run metadata:", json.dumps(meta))

    # Warmup: load the model so load time doesn't pollute level 1.
    print("# warming up ...")
    warm, _, _ = run_batch(args.host, args.model, args.prompt, 16, 1, args.seed)
    if warm[0]["error"]:
        raise SystemExit(f"warmup failed, aborting: {warm[0]['error']}\n"
                         f"hint: does `ollama list` (against this server) show {args.model}?")
    try:
        ps = get_json(f"{args.host}/api/ps")
        for m in ps.get("models", []):
            print(f"# loaded: {m.get('name')} size={m.get('size')} "
                  f"size_vram={m.get('size_vram')} ctx={m.get('context_length', '?')}")
    except Exception:
        pass

    fields = list(meta) + ["concurrency", "rep", "req", "batch_wall_s", "batch_agg_tps",
                           "ttft_s", "e2e_s", "prompt_tokens", "output_tokens",
                           "decode_tps", "prefill_tps", "load_s", "error"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        print(f"\n{'N':>3} | {'per-request decode tok/s (median)':<36} | "
              f"{'TTFT s (median)':<28} | {'aggregate tok/s (median)':<28} | errors")
        for n in args.levels:
            all_reqs, aggs = [], []
            for rep in range(args.reps):
                res, wall, agg = run_batch(args.host, args.model, args.prompt,
                                           args.num_predict, n, args.seed)
                aggs.append(agg)
                for i, r in enumerate(res):
                    row = {**meta, "concurrency": n, "rep": rep, "req": i,
                           "batch_wall_s": wall, "batch_agg_tps": agg}
                    row.update({k: r[k] for k in fields if k in r})
                    w.writerow(row)
                    all_reqs.append(r)
            errs = sum(1 for r in all_reqs if r["error"])
            print(f"{n:>3} | {fmt([r['decode_tps'] for r in all_reqs]):<36} | "
                  f"{fmt([r['ttft_s'] for r in all_reqs]):<28} | {fmt(aggs):<28} | {errs}")
    print(f"\n# rows written to {args.out}")


if __name__ == "__main__":
    main()
