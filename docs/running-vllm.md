# Running an LLM with vLLM on a Dell Pro GB10

A step-by-step runbook for serving a general-purpose LLM with [vLLM](https://docs.vllm.ai) on one of our Dell Pro GB10 nodes and chatting with it from the terminal. It was first done on `promaxgb10-2c9c` on 2026-11-16.

**What you end up with:** `Qwen/Qwen3.5-35B-A3B-FP8` served by vLLM at `http://127.0.0.1:8000/v1` (OpenAI- and Anthropic-compatible APIs, tool calling on, 128k context). Both the plain-language terminal chat (`harness/chat.py`) and the campus coding agent ([`ndcode`](ndcode.md)) talk to it.

> The first version of this runbook (2026-11-16) used the smaller `Qwen/Qwen3.5-9B`. It's still available with `MODEL=Qwen/Qwen3.5-9B GPU_UTIL=0.30 MAX_LEN=32768 TOOL_PARSER= ./serving/vllm/serve.sh`, and some numbers below come from that run (marked **9B**).

---

## 0. Know the machine

| | |
|---|---|
| Node | Dell Pro GB10, NVIDIA Grace CPU (**ARM64 / aarch64**) + Blackwell GB10 GPU |
| Memory | ~121 GB usable, **shared by CPU and GPU** (unified memory) |
| Software | Ubuntu 24.04, NVIDIA driver 580, CUDA 13.0 |

Two consequences:

1. **Everything must have ARM64 builds.** vLLM ≥ 0.33 ships an aarch64 wheel on PyPI, built against CUDA 13.
2. **GPU memory *is* system memory, and the nodes are shared.** vLLM reserves its whole memory budget up front. Its default (90% of memory) would starve everyone else, so we always cap it. Before starting, check what's running:
   ```sh
   nvidia-smi        # who is using the GPU
   free -g           # "available" column = what you can actually use
   ```

## 1. One-time setup

You need [`uv`](https://docs.astral.sh/uv/) (already installed on the nodes) and the repo:

```sh
git clone https://github.com/Francisco-MEB/aiml-inference.git
cd aiml-inference/serving/vllm
uv sync
```

`uv sync` builds an isolated environment in `serving/vllm/.venv` (~9 GB: vLLM, PyTorch 2.14 + CUDA 13, FlashInfer, Triton). It's deliberately separate from the repo-root environment, so the lm-eval/Ollama tooling stays light.

Sanity check:

```sh
uv run python -c "import torch, vllm; print(vllm.__version__, torch.version.cuda, torch.cuda.is_available())"
# → 0.33.0 13.0 True
```

**Optional: pre-download the model** (~36 GB, into `~/.cache/huggingface`). Otherwise the first launch downloads it.

```sh
uvx --from huggingface_hub hf download Qwen/Qwen3.5-35B-A3B-FP8
```

## 2. Start the server (in tmux)

Run the server inside [tmux](https://github.com/tmux/tmux/wiki) so it survives SSH disconnects and you can watch its logs:

```sh
tmux new -s inference
cd ~/aiml-inference            # wherever your clone lives
./serving/vllm/serve.sh
```

Wait for this line:

```
INFO:     Application startup complete.
```

The first launch on a node takes ~5–7 min: loading weights (~1 min), `torch.compile`, and a one-time FlashInfer kernel build. Later launches reuse the caches in `~/.cache/vllm` and `~/.cache/flashinfer` and take ~1–2 min.

### What `serve.sh` runs

```sh
uv run vllm serve Qwen/Qwen3.5-35B-A3B-FP8 \
  --served-model-name chat \          # clients ask for model "chat"
  --host 127.0.0.1 --port 8000 \      # only reachable from this machine
  --gpu-memory-utilization 0.50 \     # reserve 50% of total memory (~60 GB): weights + KV cache
  --max-model-len 131072 \            # max tokens per conversation (prompt + answer); agents need a lot
  --generation-config auto \          # use the model's recommended sampling defaults
  --reasoning-parser qwen3 \          # put Qwen's "thinking" text in a separate field
  --enable-auto-tool-choice \         # let the model call tools (read/edit files, run commands)...
  --tool-call-parser qwen3_coder \    # ...in Qwen's tool-call format
  --language-model-only               # skip Qwen3.5's vision encoder (text only)
```

Change any of these with environment variables, e.g.:

```sh
GPU_UTIL=0.5 MAX_LEN=131072 ./serving/vllm/serve.sh     # more memory, longer conversations
HOST=0.0.0.0 ./serving/vllm/serve.sh                    # reachable from other machines on the network
MODEL=Qwen/Qwen3-8B ./serving/vllm/serve.sh             # a different model
```

The full table is in [`serving/vllm/README.md`](../serving/vllm/README.md).

### Reading the startup log

These lines tell you how the memory budget was spent (values from our first **9B** run):

```
Model loading took 16.71 GiB memory                       ← the weights
Available KV cache memory: 13.65 GiB                       ← room for conversations in flight
GPU KV cache size: 385,272 tokens, Maximum concurrency for 32,768 tokens per request: 11.76x
```

The **KV cache** holds the "working memory" of every active conversation. 385k tokens means ~11 users could each have a full 32k-token conversation at the same time. Raising `GPU_UTIL` grows the KV cache, not the model.

For the current **35B-A3B** default (`GPU_UTIL=0.50`, `MAX_LEN=131072`), vLLM's `/metrics` (`vllm:cache_config_info`) reports **1,097 KV blocks** and a **max concurrency of ~8.4 full 128k-token sessions**, i.e. roughly 1.1M tokens of KV cache. That's far more room per GB than the 9B, because Qwen3.5 is a *hybrid* model: only 1 in 4 layers uses full attention (which needs a KV cache), and the rest use linear attention with a small fixed-size state. You can check these numbers on a running server any time:

```sh
curl -s 127.0.0.1:8000/metrics | grep -E "cache_config_info|kv_cache_usage_perc|prefix_cache_(hits|queries)_total"
```

`prefix_cache_hits / prefix_cache_queries` is the share of prompt tokens vLLM reused instead of recomputing. Coding agents resend the same long system prompt and file context every turn, so this runs high (~90% during our coding eval).

## 3. Chat with it

In a second tmux pane (`Ctrl-b %`):

```sh
python3 harness/chat.py
```

```
You: What is an LLM, in simple terms?
Assistant: An LLM stands for Large Language Model. Think of it as ...
[TTFT 0.14s | 209 tokens | 13.2 tok/s | total 15.9s]
```

- `/reset` clears the conversation, and `/quit` or `Ctrl-d` exits. `Ctrl-c` stops an answer mid-stream.
- `--prompt "..."` asks one question and exits (handy for scripts).
- `--think` lets the model reason before answering (slower, better on hard questions), and `--show-thinking` prints that reasoning.
- The same client works against Ollama: `python3 harness/chat.py --base-url http://127.0.0.1:11434/v1 --model qwen3.5:9b`.

Or hit the API directly:

```sh
curl -s 127.0.0.1:8000/v1/models
curl -s 127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"model":"chat","messages":[{"role":"user","content":"Hello!"}],"chat_template_kwargs":{"enable_thinking":false}}'
```

## 4. Stop it (don't skip this on a shared node)

In the server pane, press `Ctrl-c`. Then confirm the memory came back with `free -g`.

### tmux cheat-sheet

| Keys | Does |
|---|---|
| `Ctrl-b %` / `Ctrl-b "` | split pane side-by-side / top-bottom |
| `Ctrl-b ←/→/↑/↓` | move between panes |
| `Ctrl-b d` | detach (everything keeps running; safe to log out) |
| `tmux attach -t inference` | re-attach later |
| `Ctrl-b [` then arrows / PgUp | scroll back through logs (`q` to exit) |
| `tmux ls` | list sessions |

## 5. Performance: what to expect

First measurements on the **9B** (single user, BF16 weights, while another team's Ollama job was also running, so treat them as smoke-test numbers, not benchmarks):

| Metric | Value |
|---|---|
| Time to first token | 0.12–0.19 s |
| Generation speed (1 user) | ~13 tokens/s |

Why only ~13 tok/s? Generating each token means reading all ~17 GB of weights from memory, and the GB10's memory bandwidth (~273 GB/s) caps one stream at roughly 15 tok/s for this model. Ways up:

- **Quantization** (FP8 / NVFP4 weights): fewer bytes per token, roughly 2–4× faster.
- **Many users at once:** vLLM batches requests, so *total* throughput across users is far higher than the single-user number. This is what matters for campus capacity, and it's the next thing to measure.
- **MoE models**: only the "active" parameters are read per token. This is why the campus default is now **Qwen3.5-35B-A3B-FP8**: 35B parameters of knowledge, but only ~3B (~3 GB in FP8) read per token.

## 6. Troubleshooting (problems we actually hit)

| Symptom | Cause | Fix |
|---|---|---|
| `fatal error: Python.h: No such file or directory`, then `Model architectures [...] failed to be inspected` | Triton compiles a small C helper at startup and needs Python's C headers. Ubuntu's system `python3.12` doesn't have them (that needs `python3.12-dev`, i.e. sudo). | Already handled: `serving/vllm` pins a **uv-managed CPython 3.12**, which ships headers (`.python-version` + `python-preference = "only-managed"`). If you see this, `rm -rf .venv && uv sync`. |
| `nvidia-cudnn-frontend ... doesn't have a wheel for the current platform` during `uv sync` | uv picked a too-new managed Python (3.15). Some vLLM dependencies have no wheels for it yet. | Keep the 3.12 pin (`.python-version`, `requires-python = ">=3.12,<3.13"`). |
| `Free memory on device ... is less than desired GPU memory utilization` | Someone else is using the memory. | Lower `GPU_UTIL`, or wait (check `nvidia-smi`). |
| `max seq len is larger than the maximum number of tokens that can be stored in KV cache` | Budget too small for `MAX_LEN`. | Lower `MAX_LEN` or raise `GPU_UTIL`. |
| `Can't reach the model server at http://127.0.0.1:8000/v1` (from `chat.py`) | Server not running or still starting. | Wait for `Application startup complete`. |
| Startup "hangs" at `flashinfer.jit: Building JIT module sampling` | One-time kernel compile on a fresh node. | Wait a few minutes. It's cached afterwards. |

## Next steps

- Load-test vLLM with many concurrent users (adapt `scripts/ollama_concurrency_probe.py` to the OpenAI API).
- Try more FP8/NVFP4 and MoE models, and compare against Ollama on the same node.
- ~~Grow `harness/` into a coding agent~~ Done differently: we ship [`ndcode`](ndcode.md), a locked-down OpenCode distribution, instead of building our own agent. Acceptance results are in [`coding-eval/`](coding-eval/).
