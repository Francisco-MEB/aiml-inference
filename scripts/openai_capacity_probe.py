#!/usr/bin/env python3
"""
openai_capacity_probe.py

How many simultaneous users can an OpenAI-compatible server (vLLM, Ollama,
SGLang, either cluster) carry, and how fast does each user's answer stream?
Stdlib only, so it runs as-is on any node or laptop.

For each concurrency level N, it fires N requests at the same instant, repeats
that R times, and records per request (all measured on the client):
  - ttft_s          time to first token (includes queueing: what a user feels)
  - e2e_s           send to last token
  - stream_tps      this user's tokens/s after the first token
  - itl_p50/p95_ms  gaps between streamed chunks (choppiness)
and per batch:
  - batch_agg_tps   output tokens across the batch / batch wall time
                    = the node's throughput at that load

--base-url can be given more than once: requests round-robin across servers
(two replicas, two nodes) with no load balancer in between.

By default every request generates exactly --max-tokens tokens (vLLM's
`ignore_eos`), so runs are comparable. Each prompt starts with a unique tag so
the server can't reuse another request's cached prefix (different students
send different text). Use --no-unique to measure the cached case.

Run scripts/sample_node.py alongside to see whether the GPU or the CPU
(Python) is what saturates.

Example:
  python3 scripts/openai_capacity_probe.py --levels 1 2 4 8 16 32 64 \
      --prompt-set chat --server-note "serve.sh GPU_UTIL=0.50 MAX_LEN=131072" \
      --out docs/capacity/data/35b-a3b_chat_1node.csv
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
import uuid
from datetime import datetime, timezone

CHAT_PROMPTS = [
    "Explain how a hash table handles collisions, covering chaining and open "
    "addressing, with their time complexities.",
    "I'm studying for a thermodynamics exam. Explain the difference between "
    "entropy and enthalpy with an everyday example of each.",
    "Write a short Python function that checks whether a string is a palindrome, "
    "then explain how it works line by line.",
    "Summarize the causes of the French Revolution for a first-year history class.",
    "What is the difference between a t-test and a chi-squared test, and when "
    "would I use each?",
    "Help me outline a five-paragraph essay arguing for later school start times.",
]

PARAGRAPH = (
    "The university library extended its hours during finals week. Students used "
    "the study rooms late into the night, and staff added extra printers and a "
    "coffee cart near the entrance. Feedback forms showed most students wanted the "
    "change to be permanent, though some asked for quieter areas on the upper "
    "floors. "
)
# ~65 tokens per paragraph; 60 copies is roughly a 4k-token retrieved context.
RAG_PROMPT = (
    "Use only the course material below to answer.\n\n" + PARAGRAPH * 60 +
    "\n\nQuestion: What changes did students ask for, and what did staff add?"
)


def prompt_for(prompt_set, i):
    if prompt_set == "rag":
        return RAG_PROMPT
    return CHAT_PROMPTS[i % len(CHAT_PROMPTS)]


def get_json(url, api_key=None, timeout=10):
    req = urllib.request.Request(url, headers=_headers(api_key))
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _headers(api_key):
    h = {"Content-Type": "application/json"}
    if api_key:
        h["Authorization"] = f"Bearer {api_key}"
    return h


def server_info(base_url, api_key):
    """Engine version and served model (best effort; never fails the run)."""
    root = base_url[:-3] if base_url.endswith("/v1") else base_url
    version = "?"
    for path, key in (("/version", "version"), ("/api/version", "version")):
        try:
            version = get_json(root + path, api_key).get(key, "?")
            break
        except Exception:
            continue
    models = []
    try:
        models = [m.get("id") for m in get_json(base_url + "/models", api_key).get("data", [])]
    except Exception:
        pass
    return version, models


def percentile(xs, p):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def one_request(base_url, args, prompt, results, idx, start_evt):
    if args.unique:
        prompt = f"[req {uuid.uuid4().hex[:12]}] {prompt}"
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if args.fixed_length:
        payload["ignore_eos"] = True  # vLLM/SGLang extension; others ignore it
    req = urllib.request.Request(
        base_url + "/chat/completions", data=json.dumps(payload).encode(),
        headers=_headers(args.api_key),
    )
    start_evt.wait()  # all threads start together
    t0 = time.perf_counter()
    arrivals = []  # time of each chunk that carried generated text
    usage = {}
    err = ""
    try:
        with urllib.request.urlopen(req, timeout=args.timeout) as r:
            for raw in r:
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for ch in chunk.get("choices") or []:
                    d = ch.get("delta") or {}
                    # reasoning models stream thinking separately; it still costs tokens
                    if d.get("content") or d.get("reasoning_content") or d.get("reasoning"):
                        arrivals.append(time.perf_counter())
                        break
    except urllib.error.HTTPError as e:
        try:
            detail = e.read().decode(errors="replace").strip()[:300]
        except Exception:
            detail = ""
        err = f"HTTP {e.code}: {detail}"
    except Exception as e:
        err = repr(e)[:300]
    t1 = time.perf_counter()

    out_tokens = usage.get("completion_tokens")
    if out_tokens is None and arrivals:
        out_tokens = len(arrivals)  # fallback: ~1 token per chunk
    ttft = arrivals[0] - t0 if arrivals else None
    gaps = [(b - a) * 1000 for a, b in zip(arrivals, arrivals[1:])]
    stream_tps = None
    if arrivals and out_tokens and out_tokens > 1 and arrivals[-1] > arrivals[0]:
        stream_tps = (out_tokens - 1) / (arrivals[-1] - arrivals[0])
    if not err and not arrivals:
        err = "no tokens received"
    results[idx] = {
        "server": base_url,
        "t_start": t0,
        "t_end": t1,
        "ttft_s": ttft,
        "e2e_s": t1 - t0,
        "prompt_tokens": usage.get("prompt_tokens"),
        "output_tokens": out_tokens,
        "stream_tps": stream_tps,
        "itl_p50_ms": percentile(gaps, 50),
        "itl_p95_ms": percentile(gaps, 95),
        "error": err,
    }


def run_batch(args, n, prompt_set, offset=0):
    results = [None] * n
    start_evt = threading.Event()
    threads = []
    for i in range(n):
        url = args.base_url[(offset + i) % len(args.base_url)]
        t = threading.Thread(target=one_request,
                             args=(url, args, prompt_for(prompt_set, i), results, i, start_evt))
        threads.append(t)
        t.start()
    start_evt.set()
    for t in threads:
        t.join()
    wall = max(r["t_end"] for r in results) - min(r["t_start"] for r in results)
    total_out = sum(r["output_tokens"] or 0 for r in results if not r["error"])
    return results, wall, (total_out / wall if wall > 0 else None)


def fmt(x, nd=2):
    return "n/a" if x is None else f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[1],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base-url", action="append",
                    help="OpenAI-compatible base URL ending in /v1; repeat for round-robin "
                         "(default http://127.0.0.1:8000/v1)")
    ap.add_argument("--model", default="chat", help="model name the server answers to")
    ap.add_argument("--revision", default="",
                    help="model revision, recorded in the CSV (e.g. 9d1823d2)")
    ap.add_argument("--levels", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=400)
    ap.add_argument("--prompt-set", choices=["chat", "rag"], default="chat",
                    help="chat: short varied questions; rag: ~4k-token context + question")
    ap.add_argument("--no-unique", dest="unique", action="store_false",
                    help="don't tag prompts (lets the server reuse cached prefixes)")
    ap.add_argument("--no-fixed-length", dest="fixed_length", action="store_false",
                    help="don't send ignore_eos; outputs stop naturally")
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--api-key", default=None)
    ap.add_argument("--server-note", default="",
                    help="server config, e.g. 'serve.sh GPU_UTIL=0.50 MAX_LEN=131072'")
    ap.add_argument("--out", default="capacity_probe.csv")
    args = ap.parse_args()
    args.base_url = [u.rstrip("/") for u in (args.base_url or ["http://127.0.0.1:8000/v1"])]

    infos = [server_info(u, args.api_key) for u in args.base_url]
    for u, (ver, models) in zip(args.base_url, infos):
        print(f"# server {u}: engine version {ver}, models {models or '?'}")
        if models and args.model not in models:
            raise SystemExit(f"model '{args.model}' not served at {u} (has {models}); "
                             f"pass --model")
    meta = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "client_host": platform.node(),
        "servers": " ".join(args.base_url),
        "engine_version": " ".join(sorted({v for v, _ in infos})),
        "model": args.model,
        "revision": args.revision,
        "prompt_set": args.prompt_set,
        "max_tokens": args.max_tokens,
        "fixed_length": args.fixed_length,
        "unique_prompts": args.unique,
        "server_note": args.server_note,
    }
    print("# run metadata:", json.dumps(meta))

    # Warmup: compile/caches/first-request costs shouldn't land in level 1.
    print("# warming up ...")
    warm, _, _ = run_batch(args, len(args.base_url), args.prompt_set)
    bad = [r for r in warm if r["error"]]
    if bad:
        raise SystemExit(f"warmup failed, aborting: {bad[0]['server']}: {bad[0]['error']}")

    fields = list(meta) + ["concurrency", "rep", "req", "server", "batch_wall_s",
                           "batch_agg_tps", "ttft_s", "e2e_s", "prompt_tokens",
                           "output_tokens", "stream_tps", "itl_p50_ms", "itl_p95_ms", "error"]
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        print(f"\n{'N':>4} | {'agg tok/s':>9} | {'per-user tok/s p50':>18} | "
              f"{'TTFT p50 s':>10} | {'TTFT p95 s':>10} | {'ITL p95 ms':>10} | errors")
        for n in args.levels:
            reqs, aggs = [], []
            for rep in range(args.reps):
                res, wall, agg = run_batch(args, n, args.prompt_set, offset=rep)
                aggs.append(agg)
                for i, r in enumerate(res):
                    row = {**meta, "concurrency": n, "rep": rep, "req": i,
                           "batch_wall_s": wall, "batch_agg_tps": agg}
                    row.update({k: r[k] for k in fields if k in r})
                    w.writerow(row)
                    reqs.append(r)
                f.flush()
            ok = [r for r in reqs if not r["error"]]
            errs = len(reqs) - len(ok)
            agg_med = statistics.median([a for a in aggs if a is not None]) if any(aggs) else None
            print(f"{n:>4} | {fmt(agg_med, 1):>9} | "
                  f"{fmt(percentile([r['stream_tps'] for r in ok], 50), 1):>18} | "
                  f"{fmt(percentile([r['ttft_s'] for r in ok], 50)):>10} | "
                  f"{fmt(percentile([r['ttft_s'] for r in ok], 95)):>10} | "
                  f"{fmt(percentile([r['itl_p95_ms'] for r in ok], 50), 0):>10} | {errs}")
            if errs:
                print(f"     first error: {next(r['error'] for r in reqs if r['error'])}")
    print(f"\n# rows written to {args.out}")


if __name__ == "__main__":
    main()
