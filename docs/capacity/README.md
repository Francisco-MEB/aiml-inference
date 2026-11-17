# Capacity benchmark

How many simultaneous users can one GB10 serve, and what runs out first: the GPU, the CPU (Python) or memory? This is the bake-off's core metric, and it also settles a question that keeps coming up: **would a C++ stack serve campus better than our Python-orchestrated one?**

*Status (2026-11-17): tools built and tested against a mock server. No real run yet, because the node was busy with another job (see [Running it](#running-it)). The results go in [Results](#results) and [`data/`](data/).*

## The question: is Python a bottleneck at 15,000 users?

**Short answer: we expect not. The limit is GPU memory bandwidth and node count. This benchmark checks that on our hardware.**

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

**Peak load:** ~**300–600 chat streams**, plus one per active coding-agent user. At ≥20 tok/s each, that's **~6,000–12,000 output tok/s** of campus throughput.

**Nodes needed ≈ peak tok/s ÷ (one node's measured aggregate tok/s at a concurrency where per-user speed is still ≥20 tok/s and TTFT p95 is acceptable).** That measured denominator is what this benchmark produces. Fill it in under [Results](#results).

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

*No runs yet.* For each run, record the probe table, the sampler's verdict at the plateau, and the sizing number:

| Date | Model @ rev | Node(s) | Prompt set | Peak agg tok/s (N) | Max N with ≥20 tok/s/user | TTFT p95 there | Bottleneck | Data |
|---|---|---|---|---|---|---|---|---|
| | | | | | | | | |

## Files

- [`scripts/openai_capacity_probe.py`](../../scripts/openai_capacity_probe.py): load generator (OpenAI API, any engine, round-robin across servers)
- [`scripts/sample_node.py`](../../scripts/sample_node.py): GPU / CPU / vLLM-queue sampler
- [`data/`](data/): raw CSVs, named `<model>_<promptset>_<topology>[_samples].csv`
