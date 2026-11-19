"""Shared helpers for the club benchmark harness (bench/). Standard library only."""
import hashlib
import json
import os
import pathlib
import subprocess
import time
import urllib.error
import urllib.request

BENCH = pathlib.Path(__file__).resolve().parents[1]   # bench/
REPO = BENCH.parent
RESULTS = pathlib.Path(os.environ.get("BENCH_RESULTS") or BENCH / "results")   # env: tests


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def canonical_hash(obj):
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_json(path):
    with open(path) as f:
        return json.load(f)


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, sort_keys=False)
        f.write("\n")


def sh(cmd, timeout=20):
    """Run a command, return stripped stdout ('' on any failure). Never raises."""
    try:
        r = subprocess.run(cmd, shell=isinstance(cmd, str), capture_output=True, text=True,
                           timeout=timeout, check=False)
        return r.stdout.strip()
    except Exception:  # noqa: BLE001
        return ""


def http_json(url, body=None, timeout=30):
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read() or b"{}")


def percentile(xs, p):
    """Nearest-rank percentile (p in 0..100). None for an empty list."""
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    k = max(1, -(-len(xs) * p // 100))   # ceil
    return xs[int(k) - 1]


def num(v):
    """float(v), or None for blanks and non-numbers (CSV cells are text)."""
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def nonce(*parts):
    return hashlib.sha256("/".join(str(p) for p in parts).encode()).hexdigest()[:8]


def with_nonce(messages, tag):
    """Copy of messages with '[req <tag>] ' in front of the first user message, so the
    server's prompt (KV) cache cannot reuse a previous request's prefill."""
    out, done = [], False
    for m in messages:
        m = dict(m)
        if not done and m["role"] == "user":
            m["content"] = f"[req {tag}] " + m["content"]
            done = True
        out.append(m)
    return out


def load_prompts(cfg):
    path = BENCH / cfg["prompts"]["path"]
    got = sha256_file(path)
    if got != cfg["prompts"]["sha256"]:
        raise SystemExit(f"prompt file sha256 mismatch: config {cfg['prompts']['sha256']} "
                         f"file {got}")
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def chat_stream(base_url, model, messages, req):
    """One streaming request against an OpenAI-compatible /chat/completions endpoint.
    Timing is client-side, so it means the same thing for vLLM, MLX or anything else."""
    body = {
        "model": model, "messages": messages, "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": req["temperature"], "max_tokens": req["max_tokens"],
    }
    if "seed" in req:              # 1.0 configs only; 2.0 sends no seed (protocol.REQUEST)
        body["seed"] = req["seed"]
    body.update(req.get("extra_body") or {})
    r = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                               data=json.dumps(body).encode(),
                               headers={"Content-Type": "application/json",
                                        "Authorization": "Bearer " + os.environ.get(
                                            "OPENAI_API_KEY", "none")})
    out = {"ttft_s": None, "e2e_s": None, "prompt_tokens": None, "output_tokens": None,
           "decode_tps": None, "prefill_tps_est": None, "reasoning_chars": 0,
           "content_chars": 0, "finish_reason": "", "error": ""}
    usage, t_first, t_last = {}, None, None
    out["req_start_unix"] = time.time()
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(r, timeout=req.get("timeout_s", 900)) as resp:
            for raw in resp:
                line = raw.strip()
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    break
                chunk = json.loads(payload)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for ch in chunk.get("choices") or []:
                    d = ch.get("delta") or {}
                    rs = d.get("reasoning") or d.get("reasoning_content") or ""
                    out["reasoning_chars"] += len(rs)
                    if d.get("content"):
                        now = time.perf_counter()
                        t_first = t_first or now
                        t_last = now
                        out["content_chars"] += len(d["content"])
                    if ch.get("finish_reason"):
                        out["finish_reason"] = ch["finish_reason"]
    except urllib.error.HTTPError as e:
        out["error"] = f"HTTP {e.code}: {e.read().decode(errors='replace').strip()[:300]}"
    except Exception as e:  # noqa: BLE001 - recorded in the CSV, never fatal for the batch
        out["error"] = repr(e)
    t1 = time.perf_counter()
    out["e2e_s"] = t1 - t0
    out["prompt_tokens"] = usage.get("prompt_tokens")
    out["output_tokens"] = usage.get("completion_tokens")
    if t_first is not None:
        out["ttft_s"] = t_first - t0
        n = out["output_tokens"]
        if n and n > 1 and t_last > t_first:
            out["decode_tps"] = (n - 1) / (t_last - t_first)
        if out["prompt_tokens"] and out["ttft_s"] > 0:
            # includes queueing and network; only a fair prefill estimate at concurrency 1
            out["prefill_tps_est"] = out["prompt_tokens"] / out["ttft_s"]
    return out


# ---- run bookkeeping ----------------------------------------------------------------------

def slots_of(env):
    """Parallel server slots a run used: env.json["slots"] (2.0), or OLLAMA_NUM_PARALLEL for
    1.0 and pre-1.0 runs. '' if unknown."""
    if env.get("slots") not in (None, ""):
        return str(env["slots"])
    return str((env.get("server_env_set_by_harness") or {}).get("OLLAMA_NUM_PARALLEL", ""))
