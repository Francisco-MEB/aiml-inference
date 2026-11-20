# serving/mlx

MLX (`mlx_lm.server`, mlx-lm 0.32.2 on mlx 0.32.4) for the club benchmark on Apple Silicon
Macs. It is its own [uv](https://docs.astral.sh/uv/) project, like `serving/vllm`, and resolves
for macOS arm64 only. The benchmark harness starts and stops the server itself
(`bench/harness/engines.py`); you normally never run it by hand.

## Setup (once per Mac)

```sh
cd serving/mlx
uv sync
uv run python -c "import mlx.core as mx, mlx_lm; print(mlx_lm.__version__, mx.__version__, mx.default_device())"
# → 0.32.2 0.32.4 Device(gpu, 0)
```

## What the harness runs

```sh
.venv/bin/python bench/harness/engines.py --config bench/configs/<study>.json --model <key> --slots 4 --dry-run
```

prints the exact command, e.g.

```sh
uv run --project serving/mlx --frozen mlx_lm.server --model ~/.cache/huggingface/hub/models--mlx-community--Qwen3-8B-8bit/snapshots/<commit> \
  --host 127.0.0.1 --port 8100 --max-tokens 4096 --decode-concurrency 4 --prompt-concurrency 4 \
  --chat-template-args '{"enable_thinking":false}'
```

Requests use the model name `default_model` (what `mlx_lm.server` maps to the model it started
with) and never carry a `seed`, because a seeded request is served alone, without batching.

## Smoke test before your first study (Mac owners)

These Mac paths were built and unit-tested on Linux only; nobody has run them on a Mac yet.
Before your first real study, run this once and report anything that fails to the benchmarks
lead:

1. `uv sync` here and the version check above.
2. `.venv/bin/python bench/harness/preflight.py`: `PASS` on an idle Mac (it reads `ps`,
   `vm_stat` and `sysctl`).
3. `.venv/bin/python bench/harness/catalog.py plan gpt-oss`: the plan matches your RAM.
4. A tiny test run (marked as deviating, never merged):
   ```sh
   .venv/bin/python bench/harness/catalog.py plan qwen3-8b --member <you> --write
   .venv/bin/python bench/harness/fill_digests.py bench/configs/<study_id>.json
   bench/run_protocol.sh --config bench/configs/<study_id>.json --calibrate --levels 1 2 \
     --test-overrides --reps 1 --allow-deviation
   bench/status.sh <study_id>
   ```
   Check that the log shows `server ready`, every level ends with `errors 0`, the thinking check
   shows `reasoning_chars: 0`, and `bench/.run/bench_server.pid` is gone afterwards.
5. **Vision-language conversions:** several catalog models are vision-language checkpoints
   (Gemma 4, Qwen3.5/3.6/3.8, Muse Glimmer, Mistral Small 3.2, Llama 4 Scout). Plain `mlx_lm`
   may refuse some of these `mlx-community` conversions (they can need `mlx-vlm`). If a model's
   server fails to start, send the end of `bench/results/<study>/<run>/server_<model>.log` to the
   benchmarks lead.
6. Remove the test results folder and config when done.

## Notes

- macOS lets the GPU wire about 75% of RAM by default (`sysctl iogpu.wired_limit_mb`); the
  catalog plans with that budget. Raising the limit is a deviation; record it in the study notes.
- Sleep: `bench/run_protocol.sh` keeps the Mac awake for the length of a run (`caffeinate -w`).
  Turn on System Settings → Energy → "Prevent automatic sleeping when the display is off" once,
  so the Mac also stays reachable between runs. Run Claude Code inside `tmux` on the Mac
  (`brew install tmux`) if its status loop should survive your SSH session.
- Power is not recorded on Macs in protocol 2.0 (`powermetrics` needs sudo).
