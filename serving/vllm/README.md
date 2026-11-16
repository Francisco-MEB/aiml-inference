# serving/vllm

Run an LLM behind [vLLM](https://docs.vllm.ai)'s OpenAI-compatible API on a Dell Pro GB10.

This is its own [uv](https://docs.astral.sh/uv/) project, separate from the repo-root environment, so vLLM's large dependencies (torch, CUDA libraries) don't end up in the lm-eval/Ollama env.

## Setup (once per machine)

```sh
cd serving/vllm
uv sync                                   # installs vLLM (aarch64 wheel) into serving/vllm/.venv
```

## Run

```sh
./serving/vllm/serve.sh                   # Qwen/Qwen3.5-9B, served as "chat" on 127.0.0.1:8000
```

The first launch downloads the model (~19 GB) into `~/.cache/huggingface`. The server is ready when it logs `Application startup complete`. Then chat with it:

```sh
python3 harness/chat.py
```

Stop the server with Ctrl-C. **Don't leave it running on a shared node when you're done:** vLLM holds its full memory reservation the whole time it's up.

## Configuration

Everything is set by environment variable. Extra arguments are passed straight to `vllm serve`.

| Variable | Default | Meaning |
|---|---|---|
| `MODEL` | `Qwen/Qwen3.5-9B` | Hugging Face model ID or local path |
| `NAME` | `chat` | Model name clients send in requests |
| `HOST` / `PORT` | `127.0.0.1` / `8000` | Bind address. Use `HOST=0.0.0.0` to expose it to other machines |
| `GPU_UTIL` | `0.30` | Fraction of **total** memory vLLM reserves (weights + KV cache) |
| `MAX_LEN` | `32768` | Max context length (prompt + answer) in tokens |
| `REASONING_PARSER` | `qwen3` | Splits "thinking" text into its own field. Set it empty for non-reasoning models |
| `TEXT_ONLY` | `1` | Skip the vision encoder of multimodal models. Set `0` for text-only models or to send images |

Examples:

```sh
GPU_UTIL=0.5 MAX_LEN=131072 ./serving/vllm/serve.sh            # more KV cache, longer context
MODEL=Qwen/Qwen3-8B TEXT_ONLY=0 ./serving/vllm/serve.sh          # a different model
./serving/vllm/serve.sh --enable-auto-tool-choice --tool-call-parser qwen3_coder   # tool calling
```

## ⚠️ Shared GPU memory on the GB10

The GB10's CPU and GPU share one ~121 GB memory pool. vLLM reserves `GPU_UTIL × total` **up front**, and refuses to start if that much isn't free. vLLM's own default is 0.9, which would starve every other process on the box (other people's Ollama runs, evals, even the desktop).

- Check what's running first: `nvidia-smi` and `free -g`.
- Pick `GPU_UTIL` so that weights (~19 GB for a 9B BF16 model) plus the KV cache fit in what's actually free.
- **Numbers measured while someone else's job is running are smoke tests, not benchmarks.**

## Troubleshooting

- **`fatal error: Python.h: No such file or directory`** (from Triton at startup): the env is using a Python without C headers. This project pins a uv-managed CPython 3.12 (`.python-version`, `python-preference = "only-managed"`), which includes them. Rebuild with `rm -rf .venv && uv sync`. Don't move off 3.12: newer Pythons lack wheels for some vLLM dependencies.

- **"Free memory on device ... is less than desired GPU memory utilization"**: something else is using memory. Lower `GPU_UTIL`, or wait for the other job to finish.
- **"To serve at least one request ... max seq len is larger than the maximum number of tokens that can be stored in KV cache"**: lower `MAX_LEN` or raise `GPU_UTIL`.
- **CUDA / torch errors on import**: the vLLM wheel's CUDA build must work with the driver (`nvidia-smi` shows the max CUDA version it supports). Check with `uv run python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"`.
