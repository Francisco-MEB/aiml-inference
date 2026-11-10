# scripts

Standalone benchmarking and measurement tools. Each script is self-contained and documented below.

| Script | What it answers |
|---|---|
| [`ollama_concurrency_probe.py`](ollama_concurrency_probe.py) | How does one Ollama server hold up as more users hit it at the same time? |

---

## `ollama_concurrency_probe.py`

A load probe for a single [Ollama](https://ollama.com) server. It sends **N identical requests at the same instant**, repeats that several times for each N, and records latency and throughput for every request. Use it to find out how many simultaneous users a node can serve before responses get too slow. That number matters for both of this project's use cases: classroom chatbots/RAG and coding agents.

It uses only the Python standard library, so you can copy it to any node and run it with no `pip install` and no virtualenv.

### Requirements

- Python 3.8+ (any `python3`; `uv run` works too)
- A reachable Ollama server (local or remote) with the target model already pulled (`ollama pull <model>`)

### Quick start

```bash
# Against a local server, default levels 1 2 4 8, 5 reps each, 256 output tokens
python3 scripts/ollama_concurrency_probe.py --model qwen3:4b-instruct

# A fuller, reproducible run against a remote node
python3 scripts/ollama_concurrency_probe.py \
    --host http://dell1.example.edu:11434 \
    --model qwen2.5:14b \
    --levels 1 2 4 8 16 --reps 5 --num-predict 256 \
    --server-note "NUM_PARALLEL=4 MAX_LOADED=1 CTX=4096 GPU=A100-40G" \
    --out results_dell1.csv
```

### Server configuration matters more than client flags

How much concurrency Ollama allows is set by environment variables **on the server process**, not by this script:

| Variable | Effect |
|---|---|
| `OLLAMA_NUM_PARALLEL` | Max requests a loaded model decodes at once. Requests beyond this are queued. |
| `OLLAMA_MAX_LOADED_MODELS` | How many models can stay resident at once. If it's too low, models get evicted and `load_s` goes up. |
| `OLLAMA_CONTEXT_LENGTH` | Context window per request. KV-cache memory grows roughly with `NUM_PARALLEL × CONTEXT_LENGTH`. |

To change them, restart the server, for example:

```bash
OLLAMA_NUM_PARALLEL=4 OLLAMA_CONTEXT_LENGTH=4096 ollama serve
# or, for the systemd service:  sudo systemctl edit ollama   (add Environment= lines), then restart
```

The script can't read these values from the server, so **always pass them in `--server-note`**. The note is stored in every CSV row so you can still interpret the results later.

### Options

| Flag | Default | Description |
|---|---|---|
| `--host` | `http://localhost:11434` | Base URL of the Ollama server. |
| `--model` | *(required)* | Model tag, exactly as shown by `ollama list`. |
| `--levels` | `1 2 4 8` | Concurrency levels to test (number of simultaneous requests). |
| `--reps` | `5` | Batches to run at each level. Each batch contains N requests. |
| `--num-predict` | `256` | Max output tokens per request. |
| `--prompt` | hash-table explanation prompt | Prompt sent with every request. |
| `--seed` | `42` | Sampling seed (requests also use `temperature: 0`). |
| `--server-note` | `""` | Free-text record of the server config. Copied into every row. |
| `--out` | `ollama_probe.csv` | Path of the output CSV. The file is overwritten if it exists. |

### What it does

1. **Metadata:** reads the server version from `/api/version` and prints a JSON metadata line.
2. **Warmup:** sends one short request (16 tokens) so model load time doesn't skew the N=1 numbers. If it fails (for example, the model hasn't been pulled), the script stops with a hint. After warmup it prints what `/api/ps` reports as loaded: total size, how much is in VRAM, and the context length. If `size_vram < size`, part of the model is running on CPU.
3. **Measurement:** for each level N, runs `--reps` batches. In each batch it starts N threads, holds them on a shared `threading.Event` until all are ready, then releases them together. Each thread streams `/api/generate` and records timings.
4. **Output:** writes one CSV row per request and prints a median (min, max) summary for each level.

### Metrics

| Column | Measured on | Meaning |
|---|---|---|
| `ttft_s` | client | Time to first token. **Includes time spent queued on the server**, so this is the delay a user actually feels. |
| `e2e_s` | client | Total request latency, from send to last token. |
| `decode_tps` | server | `eval_count / eval_duration`: generation speed for this request alone. |
| `prefill_tps` | server | `prompt_eval_count / prompt_eval_duration`: prompt-processing speed. |
| `load_s` | server | Model load time charged to this request. Should be about 0 after warmup. |
| `prompt_tokens`, `output_tokens` | server | Token counts reported by Ollama. |
| `batch_wall_s` | client | Time from the first request's start to the last request's finish in the batch. |
| `batch_agg_tps` | client | Total output tokens in the batch divided by `batch_wall_s`. This is the **node's throughput**. |
| `error` | client | Empty on success. Otherwise `HTTP <code>: <body>` or the exception text. |

Every row also includes the run metadata: `timestamp_utc`, `client_host`, `server`, `ollama_version`, `model`, `num_predict`, `server_note`, plus `concurrency`, `rep`, and `req`.

### Example output

From a laptop GPU with the default server env (`--levels 1 2 --reps 1 --num-predict 32`):

```
# loaded: qwen3:4b-instruct size=3525071337 size_vram=2728006778 ctx=4096

  N | per-request decode tok/s (median)    | TTFT s (median)              | aggregate tok/s (median)     | errors
  1 |   22.76 (min 22.76, max 22.76)       |    0.06 (min 0.06, max 0.06) |   21.78 (min 21.78, max 21.78) | 0
  2 |   23.39 (min 23.21, max 23.56)       |    0.76 (min 0.06, max 1.47) |   22.44 (min 22.44, max 22.44) | 0
```

Here's how to read it. At N=2, per-request speed stays the same and aggregate throughput stays flat. One request's TTFT jumps to about the time it takes to run a whole request. So the two requests were **served one after another rather than in parallel**: the second waited in the queue. The model also doesn't fully fit in VRAM (`size_vram < size`).

### Interpreting results

- **Healthy batching:** as N grows, per-request `decode_tps` drops gradually while `batch_agg_tps` rises. The node is trading single-user speed for total throughput.
- **Saturation:** `batch_agg_tps` stops rising. Adding more users after this point just makes everyone slower.
- **Queueing:** once N exceeds `OLLAMA_NUM_PARALLEL`, the max `ttft_s` jumps by roughly a full request's `e2e_s`. Users see this as a long pause before any text appears.
- **Eviction or reloads:** a nonzero `load_s` after warmup means the model was unloaded. Check `OLLAMA_MAX_LOADED_MODELS`, keep-alive settings, and whether other models are competing for memory.
- **Errors:** `HTTP 503` usually means the server's request queue is full (`OLLAMA_MAX_QUEUE`). Timeouts (600 s per request) mean the node is badly overloaded.

### Caveats

- **Identical prompts:** every request sends the same prompt, so prompt caching may make `prefill_tps` look better than it would with real, varied traffic.
- **Output length:** `temperature: 0` and a fixed seed keep output length similar across requests, but not identical. Compare tok/s figures, not raw latencies, when lengths differ.
- **Client location:** if the client runs on the same machine as the server, it competes for CPU. For clean numbers, run it from another host on the same network.
- **Short prompts:** the default prompt is short, so this measures decode-heavy, chat-style load. It doesn't cover RAG or coding-agent workloads with long contexts. Pass a long `--prompt` and raise `OLLAMA_CONTEXT_LENGTH` to test those.
