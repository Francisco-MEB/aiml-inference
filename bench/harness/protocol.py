"""The club benchmark protocol (aiml-bench/2.0) in one place.

Everything here is FIXED by the protocol: changing a value means a new protocol version, and
results from different versions are not comparable. Member choices are limited to which
catalog family or model they run and how far up the high levels go; the catalog and the
machine decide the rest (bench/catalog/models.json). The human-readable version is
bench/benchmark-protocol.md. 1.0 (Ollama) results are kept in bench/results as legacy.
"""
import common

VERSION = "aiml-bench/2.0"
LEGACY_VERSIONS = ("aiml-bench/1.0", "pre-1.0")

PROMPTS_PATH = "prompts/prompts_v1.jsonl"
PROMPTS_SHA256 = "5c7772860192c2ae65e360358407585aca613810ddfa05687ff67b5b7a8e199d"
WORKLOADS = ["short_chat", "long_prefill", "coding"]          # prompt ids, in run order

LEVELS = [1, 2, 4, 8, 16, 32, 64, 128]                         # simultaneous users
CALIBRATION_LEVELS = [1, 2, 4]
CONTEXT_PER_USER = 4096                                        # tokens (vLLM --max-model-len)

# No seed: on MLX a request with a seed is served alone (no batching), and temperature 0 is
# already greedy, so a seed changes nothing on either engine.
REQUEST = {"temperature": 0, "max_tokens": 256}
WARMUP_MAX_TOKENS = 16
IDLE_BASELINE_S = 10
POWER_INTERVAL_S = 1

ENGINES = ("vllm", "mlx")
DEFAULT_PORT = 8100
VLLM_GPU_MEMORY_UTILIZATION = 0.85

# Quality phase (core suite, thinking off): lm-evaluation-harness tasks and per-task limits.
QUALITY_TASKS = ["mmlu_pro", "gpqa_diamond_cot_zeroshot", "ifeval", "gsm8k_cot_zeroshot",
                 "humaneval_instruct", "aime25"]
QUALITY_LIMITS = {"mmlu_pro": 100}          # per MMLU-Pro subject: 14 x 100 = 1,400 questions
QUALITY_GATED = ["gpqa_diamond_cot_zeroshot"]
QUALITY_CODE_EXEC = ["humaneval_instruct"]
QUALITY_SLOTS = 32
QUALITY_CONTEXT = 16384
QUALITY_MAX_GEN_TOKS = 4096

SCENARIOS = [
    {"name": "interactive", "p95_ttft_s_max": 2.0, "decode_tps_min": 10.0},
    {"name": "relaxed", "p95_ttft_s_max": 5.0, "decode_tps_min": 5.0},
    {"name": "strict", "p95_ttft_s_max": 1.0, "decode_tps_min": 20.0},
]
CAPACITY_DECODE_STAT = "median"
DECODE_STATS = ["median", "mean", "p05", "min"]
STOP_SCENARIO = "relaxed"      # stop mode: a level failing this can't raise any capacity

MODES = ("stop", "all", "cap")

# Extra columns every request row carries.
ROW_FIELDS = ["protocol_version", "study_id", "machine", "backend", "engine", "engine_version",
              "server_version", "quantization", "slots", "model_repo", "model_revision",
              "weight_format", "bits_per_weight", "kv_cache_dtype", "kv_cache_tokens",
              "thinking"]


def reps_for(users):
    """Batches per level: 5 up to 16 users, 3 from 32 (already 96+ requests per level)."""
    return 5 if users <= 16 else 3


def scenario(name):
    return next(s for s in SCENARIOS if s["name"] == name)


def server_args(engine, slots, context=CONTEXT_PER_USER, chat_template_kwargs=None):
    """The fixed server settings for one level, as (flag, value) pairs. Model-specific flags
    (parsers, --language-model-only) come from the catalog and are added by the launcher."""
    import json
    kw = json.dumps(chat_template_kwargs, separators=(",", ":")) if chat_template_kwargs else ""
    if engine == "vllm":
        a = [("--max-num-seqs", str(slots)), ("--max-model-len", str(context)),
             ("--gpu-memory-utilization", str(VLLM_GPU_MEMORY_UTILIZATION))]
        if kw:
            a.append(("--default-chat-template-kwargs", kw))
        return a
    if engine == "mlx":
        a = [("--decode-concurrency", str(slots)), ("--prompt-concurrency", str(slots))]
        if kw:
            a.append(("--chat-template-args", kw))
        return a
    raise ValueError(f"unknown engine {engine!r}")


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
    if "seed" in r:
        bad.append("request.seed must not be set (it disables batching on MLX)")
    engine = cfg.get("backend", {}).get("type")
    if engine not in ENGINES:
        bad.append(f"backend.type must be one of {ENGINES} (found {engine!r})")
    st = cfg.get("study", {})
    for k in ("id", "member", "machine"):
        if not st.get(k):
            bad.append(f"study.{k} is required")
    if not cfg.get("quantization"):
        bad.append("quantization is required (the study's build format, from the catalog)")
    for m in cfg.get("models", []):
        if m.get("quantization") != cfg.get("quantization"):
            bad.append(f"model {m.get('key')}: quantization {m.get('quantization')!r} differs "
                       f"from the study's {cfg.get('quantization')!r}")
        if not m.get("thinking_off"):
            bad.append(f"model {m.get('key')}: thinking_off must say how thinking is disabled")
        for k in ("repo", "revision"):
            if not m.get(k):
                bad.append(f"model {m.get('key')}: {k} is required (pinned Hugging Face build)")
        if m.get("revision") and not all(c in "0123456789abcdef" for c in m["revision"]):
            bad.append(f"model {m.get('key')}: revision must be a commit SHA, not a branch")
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
