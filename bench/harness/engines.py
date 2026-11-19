#!/usr/bin/env python3
"""Private benchmark servers: vLLM on NVIDIA, MLX (mlx_lm.server) on Apple Silicon.

One server serves one model at one level: the protocol restarts it with slots = users
(protocol.server_args). It runs from the pinned local snapshot (hf.snapshot_dir), on a
private port, in its own process group, and is always stopped by the harness. It never
touches anyone else's server.

  python3 bench/harness/engines.py --config bench/configs/<study>.json --model qwen3-8b \
      --slots 4 --dry-run                     # print the exact server command"""
import argparse
import json
import os
import pathlib
import re
import shlex
import signal
import subprocess
import time
import urllib.error
import urllib.request

import common
import hf
import protocol

PID_FILE = common.BENCH / ".run" / "bench_server.pid"   # process group of the running server


def project_dir(cfg):
    engine = cfg["backend"]["type"]
    key = "vllm_project" if engine == "vllm" else "mlx_project"
    rel = os.environ.get(f"BENCH_{engine.upper()}_PROJECT") or cfg["server"][key]
    p = pathlib.Path(rel)
    return p if p.is_absolute() else common.REPO / p


def base_url(cfg):
    b = cfg["backend"]
    return f"http://{b.get('host', '127.0.0.1')}:{b.get('port', protocol.DEFAULT_PORT)}"


def request_model(cfg, model):
    """The model name requests must carry. vLLM serves it under the repo id; mlx_lm.server
    maps "default_model" to the model it was started with (any other name makes it load
    that name from the Hub)."""
    return model["repo"] if cfg["backend"]["type"] == "vllm" else "default_model"


def model_path(model):
    return str(hf.snapshot_dir(model["repo"], model["revision"]))


def command(cfg, model, slots, context=protocol.CONTEXT_PER_USER):
    """argv for the server of one model with `slots` parallel sequences."""
    engine = cfg["backend"]["type"]
    b = cfg["backend"]
    fixed = [x for pair in protocol.server_args(engine, slots, context,
                                                model.get("chat_template_kwargs"))
             for x in pair]
    fake = os.environ.get("BENCH_FAKE_SERVER")      # tests only: bench/tests/fake_server.py
    if fake:
        import sys
        return [sys.executable, fake, "--engine", engine, "--port", str(b.get("port", 8100)),
                *fixed]
    uv = ["uv", "run", "--project", str(project_dir(cfg)), "--frozen"]
    if engine == "vllm":
        return [*uv, "vllm", "serve", model_path(model),
                "--served-model-name", request_model(cfg, model),
                "--host", b.get("host", "127.0.0.1"), "--port", str(b.get("port", 8100)),
                *fixed, *model.get("server_args", [])]
    return [*uv, "mlx_lm.server", "--model", model_path(model),
            "--host", b.get("host", "127.0.0.1"), "--port", str(b.get("port", 8100)),
            "--max-tokens", str(protocol.QUALITY_MAX_GEN_TOKS), *fixed,
            *model.get("server_args", [])]


def server_env():
    env = dict(os.environ)
    env["HF_HUB_OFFLINE"] = "1"         # weights are pinned and already downloaded
    env.setdefault("VLLM_NO_USAGE_STATS", "1")
    env.setdefault("DO_NOT_TRACK", "1")
    return env


class Server:
    """A running benchmark server. Use as a context manager, or call stop()."""

    def __init__(self, cfg, model, slots, log_path, context=protocol.CONTEXT_PER_USER):
        self.cfg, self.model, self.slots, self.context = cfg, model, slots, context
        self.log_path = pathlib.Path(log_path)
        self.argv = command(cfg, model, slots, context)
        self.proc = None
        self.started_s = None

    def start(self, timeout_s=None, log=print):
        timeout_s = timeout_s or self.cfg["server"].get("startup_timeout_s", 1800)
        url = base_url(self.cfg)
        if _reachable(url):
            raise SystemExit(f"something is already listening on {url}; pick another "
                             f"backend.port or stop your old server")
        log("  server: " + " ".join(shlex.quote(a) for a in self.argv))
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with open(self.log_path, "ab") as logf:      # the child keeps its own copy of the fd
            self.proc = subprocess.Popen(self.argv, stdout=logf, stderr=subprocess.STDOUT,
                                         stdin=subprocess.DEVNULL, env=server_env(),
                                         start_new_session=True)
        PID_FILE.parent.mkdir(exist_ok=True)
        PID_FILE.write_text(f"{self.proc.pid}\n")   # lets stop_protocol.sh clean up a hung run
        while time.time() - t0 < timeout_s:
            if self.proc.poll() is not None:
                raise ServerFailed(f"server exited with {self.proc.returncode} during startup: "
                                   + tail(self.log_path))
            if _reachable(url + "/v1/models"):
                self.started_s = round(time.time() - t0, 1)
                log(f"  server ready in {self.started_s} s")
                return self
            time.sleep(2)
        self.stop()
        raise ServerFailed(f"server not ready after {timeout_s} s: " + tail(self.log_path))

    def stop(self, grace_s=90):
        if not self.proc or self.proc.poll() is not None:
            PID_FILE.unlink(missing_ok=True)
            return
        try:
            os.killpg(self.proc.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            self.proc.wait(timeout=grace_s)
        except subprocess.TimeoutExpired:
            os.killpg(self.proc.pid, signal.SIGKILL)
            self.proc.wait(timeout=30)
        PID_FILE.unlink(missing_ok=True)

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.stop()

    def info(self):
        """What the server reports about itself: version, KV cache, startup log facts."""
        engine = self.cfg["backend"]["type"]
        url = base_url(self.cfg)
        out = {"engine": engine, "startup_s": self.started_s, "slots": self.slots,
               "context": self.context, "command": self.argv}
        if engine == "vllm":
            try:
                out["engine_version"] = common.http_json(url + "/version", timeout=10)\
                    .get("version", "")
            except Exception:  # noqa: BLE001
                out["engine_version"] = ""
            out.update(vllm_metrics(url))
        else:
            out["engine_version"] = mlx_versions(self.cfg)
        out.update(log_facts(self.log_path))
        return out


class ServerFailed(RuntimeError):
    pass


def _reachable(url):
    try:
        with urllib.request.urlopen(url, timeout=3):
            return True
    except urllib.error.HTTPError:
        return True                    # something answered
    except Exception:  # noqa: BLE001
        return False


def tail(path, n=25):
    try:
        lines = pathlib.Path(path).read_text(errors="replace").splitlines()
    except OSError:
        return "(no log)"
    return "\n".join(lines[-n:])


def vllm_metrics(url):
    """KV-cache size and dtype, prefix-cache counters and preemptions from /metrics."""
    try:
        with urllib.request.urlopen(url + "/metrics", timeout=10) as r:
            text = r.read().decode(errors="replace")
    except Exception:  # noqa: BLE001
        return {}
    return parse_vllm_metrics(text)


def parse_vllm_metrics(text):
    out = {}
    m = re.search(r"^vllm:cache_config_info\{([^}]*)\}", text, re.M)
    if m:
        labels = dict(re.findall(r'(\w+)="([^"]*)"', m.group(1)))
        out["kv_cache_dtype"] = labels.get("cache_dtype", "")
        try:
            out["kv_cache_tokens"] = int(labels["num_gpu_blocks"]) * int(labels["block_size"])
        except (KeyError, ValueError):
            pass
        out["prefix_caching"] = labels.get("enable_prefix_caching", "")
    for name, key in (("vllm:num_preemptions_total", "preemptions"),
                      ("vllm:prefix_cache_hits_total", "prefix_cache_hits"),
                      ("vllm:prefix_cache_queries_total", "prefix_cache_queries")):
        vals = re.findall(rf"^{re.escape(name)}(?:\{{[^}}]*\}})? ([0-9.eE+-]+)$", text, re.M)
        if vals:
            out[key] = sum(float(v) for v in vals)
    return out


def log_facts(path):
    """Facts vLLM prints at startup (weights memory, KV cache size, max concurrency)."""
    out = {}
    try:
        text = pathlib.Path(path).read_text(errors="replace")
    except OSError:
        return out
    pats = {"weights_loaded_gib": r"Model loading took ([0-9.]+) GiB",
            "kv_cache_available_gib": r"Available KV cache memory: ([0-9.]+) GiB",
            "kv_cache_tokens_logged": r"KV cache size: ([0-9,]+) tokens",
            "max_concurrency_logged": r"Maximum concurrency for [0-9,]+ tokens per request: "
                                      r"([0-9.]+)x"}
    for k, p in pats.items():
        found = re.findall(p, text)
        if found:
            v = found[-1].replace(",", "")
            out[k] = float(v) if "." in v else int(v)
    return out


def mlx_versions(cfg):
    code = "import mlx.core as mx, mlx_lm; print(f'mlx-lm {mlx_lm.__version__}, mlx {mx.__version__}')"
    return common.sh(["uv", "run", "--project", str(project_dir(cfg)), "--frozen", "python",
                      "-c", code], timeout=120)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--model", required=True, help="model key in the config")
    ap.add_argument("--slots", type=int, default=1)
    ap.add_argument("--context", type=int, default=protocol.CONTEXT_PER_USER)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    cfg = common.load_json(a.config)
    m = next(x for x in cfg["models"] if x["key"] == a.model)
    argv = command(cfg, m, a.slots, a.context)
    if a.dry_run:
        print(" ".join(shlex.quote(x) for x in argv))
        return
    srv = Server(cfg, m, a.slots, common.BENCH / ".run" / f"server-{a.model}.log", a.context)
    with srv:
        print(json.dumps(srv.info(), indent=2))
        print("serving; Ctrl-C to stop")
        try:
            while True:
                time.sleep(3600)
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
