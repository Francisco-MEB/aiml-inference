"""The club benchmark protocol (aiml-bench/1.0) in one place.

Everything here is FIXED by the protocol: changing a value means a new protocol version, and
results from different versions are not comparable. Member choices (models, quantization,
backend, machine, high-level mode) live in each study's config and are recorded with every run.
The human-readable version is bench/benchmark-protocol.md.
"""
import common

VERSION = "aiml-bench/1.0"

PROMPTS_PATH = "prompts/prompts_v1.jsonl"
PROMPTS_SHA256 = "5c7772860192c2ae65e360358407585aca613810ddfa05687ff67b5b7a8e199d"
WORKLOADS = ["short_chat", "long_prefill", "coding"]          # prompt ids, in run order

LEVELS = [1, 2, 4, 8, 16, 32, 64, 128]                         # simultaneous users
CALIBRATION_LEVELS = [1, 2, 4]
CONTEXT_PER_USER = 4096                                        # tokens, OLLAMA_CONTEXT_LENGTH

REQUEST = {"temperature": 0, "seed": 42, "max_tokens": 256}
WARMUP_MAX_TOKENS = 16
IDLE_BASELINE_S = 10
POWER_INTERVAL_S = 1

SCENARIOS = [
    {"name": "interactive", "p95_ttft_s_max": 2.0, "decode_tps_min": 10.0},
    {"name": "relaxed", "p95_ttft_s_max": 5.0, "decode_tps_min": 5.0},
    {"name": "strict", "p95_ttft_s_max": 1.0, "decode_tps_min": 20.0},
]
CAPACITY_DECODE_STAT = "median"
DECODE_STATS = ["median", "mean", "p05", "min"]
STOP_SCENARIO = "relaxed"      # stop mode: a level failing this can't raise any capacity

MODES = ("stop", "all", "cap")

# Extra columns every request row carries (added in 1.0; Gemma pre-1.0 rows lack them).
ROW_FIELDS = ["protocol_version", "study_id", "machine", "backend", "server_version",
              "quantization", "slots"]


def reps_for(users):
    """Batches per level: 5 up to 16 users, 3 from 32 (already 96+ requests per level)."""
    return 5 if users <= 16 else 3


def scenario(name):
    return next(s for s in SCENARIOS if s["name"] == name)


def check_config(cfg):
    """List of ways a study config breaks the protocol (empty list = compliant)."""
    import json
    bad = []
    if "FILL_ME" in json.dumps(cfg):
        bad.append("some fields still say FILL_ME")
    if cfg.get("protocol") != VERSION:
        bad.append(f"protocol must be {VERSION!r} (found {cfg.get('protocol')!r})")
    p = cfg.get("prompts", {})
    if p.get("path") != PROMPTS_PATH or p.get("sha256") != PROMPTS_SHA256:
        bad.append(f"prompts must be {PROMPTS_PATH} with sha256 {PROMPTS_SHA256[:12]}...")
    r = cfg.get("request", {})
    for k, v in REQUEST.items():
        if r.get(k) != v:
            bad.append(f"request.{k} must be {v} (found {r.get(k)!r})")
    env = cfg.get("server", {}).get("env", {})
    if cfg.get("backend", {}).get("type") == "ollama" \
            and str(env.get("OLLAMA_CONTEXT_LENGTH")) != str(CONTEXT_PER_USER):
        bad.append(f"server.env.OLLAMA_CONTEXT_LENGTH must be {CONTEXT_PER_USER}")
    st = cfg.get("study", {})
    for k in ("id", "member", "machine"):
        if not st.get(k):
            bad.append(f"study.{k} is required")
    if not cfg.get("quantization"):
        bad.append("quantization is required (member's choice, e.g. q4_K_M)")
    for m in cfg.get("models", []):
        if m.get("quantization") != cfg.get("quantization"):
            bad.append(f"model {m.get('key')}: quantization {m.get('quantization')!r} differs "
                       f"from the study's {cfg.get('quantization')!r}")
        if not m.get("thinking_off"):
            bad.append(f"model {m.get('key')}: thinking_off must say how thinking is disabled")
    if not cfg.get("models"):
        bad.append("models list is empty")
    return bad


def level_stats(rows):
    """Pass/fail inputs for one model x workload x level, from its request rows."""
    ttft = [common.num(r.get("ttft_s")) for r in rows]
    dec = [common.num(r.get("decode_tps")) for r in rows]
    return {"n": len(rows), "errors": sum(1 for r in rows if r.get("error")),
            "ttft_p95_s": common.percentile(ttft, 95),
            "decode_median": common.percentile(dec, 50)}


def passes(stats, scen):
    return (stats["n"] > 0 and stats["errors"] == 0 and stats["ttft_p95_s"] is not None
            and stats["decode_median"] is not None
            and stats["ttft_p95_s"] <= scen["p95_ttft_s_max"]
            and stats["decode_median"] >= scen["decode_tps_min"])
