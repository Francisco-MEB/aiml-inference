"""Shared helpers for the Gemma 4 benchmark harness. Standard library only."""
import hashlib
import json
import os
import pathlib
import subprocess
import time
import urllib.error
import urllib.request

BENCH = pathlib.Path(__file__).resolve().parents[1]   # bench/gemma4
REPO = BENCH.parents[1]


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
    Timing is client-side, so it means the same thing for Ollama, vLLM or anything else."""
    body = {
        "model": model, "messages": messages, "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": req["temperature"], "seed": req["seed"],
        "max_tokens": req["max_tokens"],
    }
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


# ---- server control (only used for backend.type == "ollama") ---------------------------------

def server_env(cfg, overrides=None):
    env = dict(cfg["server"]["env"])
    env.update(overrides or {})
    env["OLLAMA_BENCH_VERSION"] = cfg["server"]["ollama_version"]
    env["OLLAMA_BENCH_PORT"] = str(cfg["server"]["port"])
    env["OLLAMA_MODELS"] = os.path.expanduser(cfg["server"]["models_dir"])
    return env


def _server_script(cfg):
    mode = cfg["server"]["mode"]
    if mode == "external":
        return None
    return str(BENCH / "server" / ("serve_docker.sh" if mode == "docker" else "serve_native.sh"))


def start_server(cfg, env, log_path):
    script = _server_script(cfg)
    if script is None:
        return
    full = dict(os.environ)
    full.update(env)
    full["OLLAMA_BENCH_LOG"] = str(log_path)
    r = subprocess.run([script, "start"], env=full, capture_output=True, text=True,
                       check=False)
    print(r.stdout.strip() or r.stderr.strip())
    if r.returncode != 0:
        raise SystemExit(f"server start failed: {r.stderr.strip()}")


def stop_server(cfg):
    script = _server_script(cfg)
    if script is None:
        return
    env = dict(os.environ, OLLAMA_BENCH_VERSION=cfg["server"]["ollama_version"],
               OLLAMA_BENCH_PORT=str(cfg["server"]["port"]))
    subprocess.run([script, "stop"], env=env, check=False)


def loaded_models(native_url):
    try:
        return http_json(native_url + "/api/ps", timeout=5).get("models", [])
    except Exception:  # noqa: BLE001
        return None   # not reachable
