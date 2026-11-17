# Capacity benchmark

How many simultaneous users can one GB10 serve, and what runs out first: the GPU, the CPU (Python) or memory? This is the bake-off's core metric, and it also settles a question that keeps coming up: **would a C++ stack serve campus better than our Python-orchestrated one?**

*Status (2026-11-17): first single-Dell runs done (chat, RAG, and a RAG A/B with `--async-scheduling`). See [Results](#results). Two-Dell runs (two copies vs one model split across both) are next. An interactive version with charts, built from the same CSVs, is published as a private Claude page (ask Francisco for access).*

## The question: is Python a bottleneck at 15,000 users?

**Measured answer: no.** On one GB10 running our 35B, the GPU was busy 96% of the time at every load from 1 to 128 users. During answer generation, the vLLM engine's Python loop used 6–26% of one CPU core. Turning on `--async-scheduling`, which overlaps that Python work with the GPU, changed RAG throughput by less than 1%. The limits are GPU compute for reading prompts and memory bandwidth for writing answers, which means the number of GPUs, not the language. The reasoning behind the measurement:

1. **Python doesn't run per student or per token.** In vLLM, Python runs **once per batch step**: the scheduler picks which conversations advance, then the GPU computes one token for all of them together. The math runs in C++/CUDA kernels (FlashInfer, CUTLASS, NCCL). The forward pass replays as a CUDA graph (one launch, not thousands). HTTP handling and tokenizing run in separate processes, overlapping with the GPU. Python's cost per step grows slowly with batch size and not at all with the number of students.
2. **Decode speed is set by memory bandwidth.** Each step reads the active weights from memory. The GB10's ~273 GB/s caps one stream of our 35B-A3B at ~55 tok/s ([overview §8](../overview.md#8-measurements-2026-11-16-single-node-shared)). Rewriting the scheduler in C++ doesn't raise that cap.
3. **Scaling to campus size means more nodes, and each node's Python cost stays the same.** Every node runs its own engine, so Python overhead per node doesn't grow with total users. See [Sizing](#sizing-for-campus) below.
4. **Where Python *can* hurt, and the fix for each:**
   - **The front door** (thousands of open streams): put a C proxy (nginx/Envoy) in front and run several vLLM API processes (`--api-server-count`).
   - **Very short GPU steps** (small model, huge batch): scheduling time becomes comparable, so try vLLM `--async-scheduling`.
   - **If both fail:** try a C++-runtime engine (TensorRT-LLM) as a measured comparison arm.

   None of these is a rewrite of our code. Our own Python (`serve.sh`, `chat.py`, the eval) never sits in the request path at scale.

**What would prove us wrong:** throughput stops rising while the GPU sits idle between steps and the vLLM engine's main thread is pinned at ~100% CPU. [`sample_node.py`](../../scripts/sample_node.py) records exactly those signals.

## Sizing for campus

The number of nodes comes from [Little's law](https://en.wikipedia.org/wiki/Little%27s_law): *requests in flight = arrival rate × time per request*. The inputs below are **assumptions** until we have usage data:

| Input | Assumption |
|---|---|
| People with access | 15,000 (≈10,000 students) |
| Active in the peak hour | 10–20% → 1,500–3,000 |
| Chat: one message every | 1–2 min while active |
| Chat: answer length / acceptable speed | ~400 tokens at ≥20 tok/s → ~20 s |
| Share of active chat users generating at any moment | ~20 s ÷ ~90 s ≈ 1 in 5 |
| Coding agents | ~1 stream each while working (back-to-back calls, long contexts) |

**Peak load:**
- 1,500–3,000 active users × one ~400-token answer per ~90 s ≈ 17–33 answers/s.
- That's **~6,700–13,300 output tok/s** campus-wide, plus coding agents.
- This total doesn't depend on how fast each answer streams. Slower answers just mean more of them in flight at once.

**GB10s needed = campus tok/s ÷ one GB10's total tok/s at the speed we promise each user.** From the [chat run](#results):

| Promise per user | Users per GB10 | GB10 total tok/s | GB10s for 6,700–13,300 tok/s |
|---|---|---|---|
| ~20 tok/s (feels instant) | 8 | 160 | **42–84** |
| ~11 tok/s (faster than reading) | 32 | 354 | **19–38** |
| ~5 tok/s (about reading speed) | 128 | 676 | **10–20** |

The result scales directly with the assumptions: half the peak activity means half the GB10s. Real usage logs (gateway milestone) should replace these inputs. Long-document RAG is costlier still (next section), and coding agents add load on top.

## Running it

Two terminals on the server node. Check the node is free first (CLAUDE.md: results measured alongside someone else's job aren't benchmarks):

```sh
nvidia-smi; free -g; ps -eo user,pid,etime,args | grep -E 'vllm|ollama' | grep -v grep
```

**Terminal 1: server** (see the [runbook](../running-vllm.md)). It uses the campus default, 35B-A3B FP8 @ `9d1823d2`:
```sh
cd serving/vllm && ./serve.sh            # wait for "Application startup complete"
```

**Terminal 2: sampler, then load.** The sampler runs in the background while the probe climbs through the levels:
```sh
python3 scripts/sample_node.py --out docs/capacity/data/35b-a3b_chat_1node_samples.csv &
python3 scripts/openai_capacity_probe.py \
    --levels 1 2 4 8 16 32 64 128 --reps 3 --prompt-set chat \
    --revision 9d1823d2 --server-note "serve.sh GPU_UTIL=0.50 MAX_LEN=131072 1 node" \
    --out docs/capacity/data/35b-a3b_chat_1node.csv
kill %1                                   # stop the sampler
```

Repeat with `--prompt-set rag` (~4k-token prompts) and matching file names. Stop the server when done.

**Two replicas** (later, both nodes): run `serve.sh` on each node and pass `--base-url` once per server. The probe spreads requests round-robin, so no load balancer is needed.

## Reading the results

Read the probe's summary table (per concurrency level N) side by side with the sampler CSV:

| You see | Bottleneck | What to do |
|---|---|---|
| Aggregate tok/s rises with N, per-user tok/s falls slowly | none yet (healthy batching) | keep going |
| Aggregate plateaus, `gpu_busy_pct` ≈ 100 | **GPU** (expected) | more nodes, smaller/faster model, better quantization |
| Aggregate plateaus, `gpu_busy_pct` drops, `engine_main_pct` ≈ 100 | **CPU / Python** | `--async-scheduling`; then a TensorRT-LLM arm |
| `api_main_pct` ≈ 100, TTFT climbs while the GPU has headroom | **HTTP front end** | `--api-server-count`, proxy in front |
| `waiting` > 0 with `kv_cache_pct` ≈ 100 | **KV-cache memory** | raise `GPU_UTIL`, lower `MAX_LEN`, more nodes |

`gpu_busy_pct` is nvidia-smi's `utilization.gpu`: the share of time at least one kernel was running. Gaps in it are the GPU waiting on the host. It doesn't show how efficiently each kernel runs, but it's the right signal for "is Python starving the GPU?"

## Results

### Setup for all runs below
- **Date and machine:** 2026-11-17, one Dell Pro GB10 (`promaxgb10-2c9c`). Nothing else was on the GPU; the club's server had been stopped. The load generator ran on the same Dell.
- **Engine and model:** vLLM 0.33.0 with `serve.sh` defaults: Qwen3.5-35B-A3B-FP8 @ `9d1823d2`, `GPU_UTIL=0.50`, 128k context, KV cache of 1,084,872 tokens.
- **Requests:** every answer was exactly 400 tokens, and every prompt was unique, so there were no cache hits. Each level was run 3 times. There were 0 errors in 1,866 requests.

### Chat (short questions, ~45-token prompts)

| Users at once | Total tok/s | Per-user tok/s (p50) | First token p50 / p95 | Gap between tokens p95 |
|---|---|---|---|---|
| 1 | 53 | 53 | 0.09 / 0.09 s | 20 ms |
| 2 | 84 | 43 | 0.14 / 0.18 s | 25 ms |
| 4 | 122 | 31 | 0.21 / 0.21 s | 35 ms |
| 8 | 160 | 20 | 0.25 / 0.25 s | 53 ms |
| 16 | 245 | 16 | 0.32 / 0.44 s | 72 ms |
| 32 | 354 | 11 | 0.56 / 0.56 s | 103 ms |
| 64 | 494 | 7.8 | 0.75 / 1.19 s | 146 ms |
| 128 | 676 | 5.3 | 1.29 / 1.59 s | 211 ms |

- **Still rising at 128:** total throughput hadn't levelled off, so a GB10 can carry more chat users if each one accepts a slower stream.
- **Waiting is brief:** the first token stays under 1.6 s.
- **Bottleneck:** GPU busy 96% at every level; engine main thread 6–18% of a core; KV cache at most 48% full. This is the **GPU, with memory bandwidth for generation, not Python and not memory capacity.**

### RAG (~3,400-token retrieved context + question)

| Users at once | Total tok/s | Per-user tok/s (p50) | First token p50 / p95 |
|---|---|---|---|
| 1 | 48 | 52 | 0.68 / 0.69 s |
| 4 | 105 | 29 | 1.8 / 2.7 s |
| 8 | 129 | 19 | 3.1 / 5.5 s |
| 16 | 175 | 13 | 5.9 / 11.3 s |
| 32 | 208 | 8.4 | 12 / 25 s |
| 64 | 231 | 5.1 | 26 / 58 s |
| 128 | 229 | 2.8 | 63 / 154 s |

- **Prompt reading is the limit.** One GB10 reads prompts at **~5,000 tokens/s** whether 1 or 16 users are waiting. That's a fixed GPU compute rate, so long-document requests queue behind each other. Total throughput levels off at ~230 tok/s.
- **Where it stops being usable:** at about **8 simultaneous long-document requests per GB10**, while the first token is still under ~5 s.
- **Ways to raise it:**
  - Share prefixes. A class asking about the same reading hits the cache, and this run deliberately had none.
  - Retrieve fewer tokens.
  - More GB10s.
  - Later: run prompt reading and answer writing on separate GPUs (disaggregated prefill).

### Python check: RAG with `--async-scheduling` (A/B)

| Users | Total tok/s default → async | First token p95 default → async |
|---|---|---|
| 16 | 175 → 174 | 11.3 → 11.3 s |
| 32 | 208 → 209 | 24.7 → 24.8 s |
| 64 | 231 → 231 | 58.1 → 58.3 s |

**No change.** In the RAG runs, the engine's main thread did reach ~100% of a core, but only while prompts were being read: GPU power rose to ~45 W while few requests were running. Meanwhile the GPU stayed 96% busy. So the thread was waiting on long GPU steps, not holding them up. Overlapping it with async scheduling would have helped if it were holding them up, and it didn't help.

### Data
All in [`data/`](data/): `35b-a3b_{chat,rag}_1node.csv`, `35b-a3b_rag_1node_async.csv`, each with its `_samples.csv`.

### Caveats
- **Single GB10 only.** Two copies across both Dells should roughly double every "total tok/s" figure, but that's untested.
- **The load generator shared the Dell's CPU.** This is minor, since its threads mostly wait on the network.
- **`gpu_busy_pct` is coarse.** It means "some kernel was running", not "the GPU was fully used". The A/B test is the stronger evidence on Python.

## Files

- [`scripts/openai_capacity_probe.py`](../../scripts/openai_capacity_probe.py): load generator (OpenAI API, any engine, round-robin across servers)
- [`scripts/sample_node.py`](../../scripts/sample_node.py): GPU / CPU / vLLM-queue sampler
- [`data/`](data/): raw CSVs, named `<model>_<promptset>_<topology>[_samples].csv`
