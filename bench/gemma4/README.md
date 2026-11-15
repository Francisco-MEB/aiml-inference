# Gemma 4 benchmark harness (Dell Pro Max GB10)

Reproducible speed, concurrency, power and quality runs for the Gemma 4 QAT Q4_0 family.
One command produces a run directory with raw data, `env.json` and a spreadsheet
(`findings.xlsx`) whose first sheet lists every parameter used.

> **Not Docker (yet).** The club stack is Ollama in Docker. On `promaxgb10-2c9c` the benchmark
> user is not in the `docker` group and has no sudo, so this config runs the **same Ollama
> version (0.40.13) as a native binary** on a private port. Every run records that in `env.json`
> and at the top of the spreadsheet. `server/serve_docker.sh` holds the pinned image for when
> access exists; it has not been tested on this host.

## Pre-flight: is anyone else using the machine?

Run this before anything that loads a model. It only reads state.

```sh
python3 bench/gemma4/harness/preflight.py
```

It prints `PRE-FLIGHT: PASS` or `STOP`, and checks:

- [ ] `nvidia-smi`: no compute processes owned by someone else
- [ ] system Ollama (`:11434`) has no model loaded (`ollama ps`)
- [ ] no other user's process above 50% CPU; 1-minute load average below 2
- [ ] `who`: note who is logged in and ask in the club channel if unsure
- [ ] containers: `docker ps` (needs docker access) or the `containerd-shim` process list

`run_bench.py` runs the same check first and refuses to start on `STOP`.

## Reproduce a run from scratch

```sh
git clone https://github.com/Francisco-MEB/aiml-inference && cd aiml-inference
git switch bench/gemma4-groundwork
uv sync                                    # lm-eval 0.4.13, openpyxl 3.1.5 (uv.lock)

# Smoke test: E4B only, about 1.5 minutes after the model is downloaded
uv run python bench/gemma4/harness/run_bench.py --models e4b --levels 1 2 --reps 1 \
    --max-tokens 32 --num-parallel 2 --quality-limit 5

# Full speed + capacity sweep (E4B, 12B, 31B, then the 26B A4B MoE), capped at 4.5 hours
uv run python bench/gemma4/harness/run_bench.py --models e4b 12b 31b 26b-a4b --time-budget-hours 4.5
```

The pinned config runs speed and capacity only (`quality.enabled` is false). Add
`--with-quality` to also run lm-eval. Between models the runner repeats the shared-machine
check and stops if someone else has started using the GPU. If the next model would not fit in
the time budget it stops cleanly and reports on the models already done.

Results land in `bench/gemma4/results/<UTC time>_<host>_<config hash>/` (not committed).

The server binary is `/usr/local/bin/ollama` (sha256 pinned in the config; the run aborts if it
differs). On a host without it, delete `server.native.binary_path` from the config and run
`bench/gemma4/server/install_ollama_native.sh`, which downloads the pinned v0.40.13 release
and checks its sha256.

## What is pinned

Everything is in [config/gemma4_ollama_native.json](config/gemma4_ollama_native.json).

| Item | Value |
|---|---|
| Ollama | 0.40.13 (native binary sha256 `81f23079…`; Docker image `ollama/ollama:0.40.13`, arm64 `sha256:44236782…`) |
| Server | `127.0.0.1:11435`, models in `~/ollama-bench-models`, `OLLAMA_NUM_PARALLEL=16`, `OLLAMA_CONTEXT_LENGTH=4096`, `OLLAMA_MAX_LOADED_MODELS=1`, `OLLAMA_KEEP_ALIVE=30m` |
| Models | `gemma4:e4b-it-qat`, `gemma4:12b-it-qat`, `gemma4:31b-it-qat`, optional `gemma4:26b-a4b-it-qat` (MoE); all Q4_0 QAT |
| Digests | manifest sha256 and weights-file sha256 per model; the weights file is re-hashed on disk every run |
| Request | temperature 0, seed 42, `max_tokens` 256, thinking off |
| Sweep | concurrency 1 2 4 8 16, 5 reps, 3 prompts ([prompts/prompts_v1.jsonl](prompts/prompts_v1.jsonl), sha256 pinned) |
| User capacity | highest level with p95 TTFT ≤ 2 s and median per-request decode ≥ 10 tok/s, no errors |

The system `ollama.service` on `:11434` is never used or changed. The server's other settings
are left at their defaults, and the full list it logs at startup is copied into `env.json`.

**Thinking mode.** On this server Gemma 4 thinks by default when nothing is specified. The
harness sends `reasoning_effort: "none"` on `/v1/chat/completions` (the native API equivalent is
`think: false`). Each run probes all three states (pinned, unset, high) and records the
reasoning text length, so the spreadsheet shows the switch worked.

**Google's own files.** The Ollama tags are the same size as Google's official GGUFs
(`google/gemma-4-*-it-qat-q4_0-gguf`) but not byte-identical. Google's sha256 values are kept in
the config for reference.

## The spreadsheet

| Sheet | Contents |
|---|---|
| Parameters | every setting used, with the not-in-Docker note first |
| Models | one row per model: size, memory, single-user speed, peak throughput, capacity |
| Summary | per model, prompt and concurrency: TTFT, decode speed, aggregate speed, power |
| Capacity | user capacity under three threshold scenarios and four decode statistics |
| Scaling | throughput and per-user speed against concurrent users, with charts |
| ServerTimingCheck | client-side speed against Ollama's own timings |
| Environment, Requests, Power | run metadata and raw rows |

Capacity is the highest tested concurrency that passes with every lower level passing too.

## What gets measured

All speed numbers come from the client side of the OpenAI-compatible streaming API, so they
mean the same thing on any backend:

- **TTFT**: request sent to first answer text
- **decode tok/s per request**: (output tokens − 1) / (last token time − first token time)
- **prefill tok/s**: prompt tokens / TTFT. Only meaningful at concurrency 1.
- **aggregate tok/s**: all output tokens in a batch / batch wall time

Every request starts with a short deterministic tag (`[req 1a2b3c4d]`) so the server cannot
skip prefill by reusing its prompt cache. For Ollama, `ServerTimingCheck` compares these numbers
with the server's own timings (they agreed within about 3% in the smoke test).

Per model: load, discarded warm-up (one request, then one batch at the highest level),
thinking probe, sweep, unload with `keep_alive: 0`.

## Power

`power_logger.py` samples `nvidia-smi` once a second for the whole run. That is **GPU power
only**: on this unified-memory machine it leaves out CPU and memory, and no other software
counter exists (no RAPL, no energy counter, hwmon is temperatures only). Use the wall meter for
system power:

```sh
# meter.csv needs two columns: t_unix,watts
uv run python bench/gemma4/harness/make_report.py bench/gemma4/results/<run_id> --wall-csv meter.csv
```

That fills `wall_meter_w` and adds wall watts and joules per output token to the Summary sheet.
Batches shorter than a few seconds get only a few samples, so trust the energy numbers from
full-length runs, not from the smoke test.

## Quality

`lm-evaluation-harness` against the same endpoint, generative tasks only (no logprobs):

- `mmlu_pro_computer_science`
- `humaneval_chat` (in [lm_eval_tasks/](lm_eval_tasks/)): the stock HumanEval tasks cut chat
  answers off at `\ndef` and score 0, so this variant extracts the code block. **It executes
  model-written code** on this machine; set `quality.allow_code_execution` to `false` to skip it.
- `gpqa_main_generative_n_shot`: the dataset is gated, so it is skipped unless `HF_TOKEN` is set.

The server is restarted for this phase with a 16384 context (5-shot prompts do not fit in 4096).
Use `--quality-limit N` for quick checks, `--skip-quality` or `--skip-speed` to run one phase.

## Another backend (vLLM next)

The runners need only `backend.base_url` and a model name. Copy the config, set
`"backend": {"type": "openai", "base_url": "http://<host>:8000/v1"}` and
`"server": {"mode": "external"}`, and put the served model names in `models[].tag`. Pull, digest
checks, load/unload and the server-timing check are Ollama-only and are skipped. Adjust
`request.extra_body` to whatever turns thinking off on that server.

## Files

| Path | Purpose |
|---|---|
| `harness/run_bench.py` | whole pipeline |
| `harness/preflight.py` | shared-machine check |
| `harness/pull_models.py` | pull and verify digests (also runs standalone) |
| `harness/run_speed.py` | concurrency sweep |
| `harness/power_logger.py` | 1 Hz GPU telemetry |
| `harness/run_quality.py` | lm-eval wrapper |
| `harness/env_snapshot.py` | `env.json` |
| `harness/make_report.py` | `findings.xlsx` and `report/*.csv` |
| `server/serve_native.sh`, `serve_docker.sh`, `install_ollama_native.sh` | start/stop the server |
