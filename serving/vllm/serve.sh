#!/usr/bin/env bash
# Launch a vLLM OpenAI-compatible server with defaults that are safe on a
# shared Dell Pro GB10 (unified CPU/GPU memory). Override anything via env vars:
#
#   MODEL=Qwen/Qwen3-8B GPU_UTIL=0.4 PORT=8001 ./serve.sh
#   ./serve.sh --enable-prefix-caching        # extra args go straight to vllm
#
# GPU_UTIL is a fraction of TOTAL memory (~121 GB on a GB10), and vLLM reserves
# all of it up front (weights + KV cache). Never leave it at vLLM's default of
# 0.9 on a node other people are using: it will starve or crash their jobs.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3.5-9B}"
NAME="${NAME:-chat}"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
GPU_UTIL="${GPU_UTIL:-0.30}"
MAX_LEN="${MAX_LEN:-32768}"
# Model-specific options (defaults suit Qwen3.5). Set to empty / 0 for other models:
#   REASONING_PARSER=  TEXT_ONLY=0 MODEL=meta-llama/Llama-3.1-8B-Instruct ./serve.sh
REASONING_PARSER="${REASONING_PARSER-qwen3}"  # splits <think> output into its own field
TEXT_ONLY="${TEXT_ONLY:-1}"                   # skip the vision encoder of multimodal models

cd "$(dirname "$0")"

args=(
  --served-model-name "$NAME"
  --host "$HOST"
  --port "$PORT"
  --gpu-memory-utilization "$GPU_UTIL"
  --max-model-len "$MAX_LEN"
  --generation-config auto
)
[[ -n "$REASONING_PARSER" ]] && args+=(--reasoning-parser "$REASONING_PARSER")
[[ "$TEXT_ONLY" == 1 ]] && args+=(--language-model-only)

echo "Serving $MODEL as '$NAME' on http://$HOST:$PORT/v1 (gpu-memory-utilization=$GPU_UTIL, max-model-len=$MAX_LEN)" >&2
exec uv run vllm serve "$MODEL" "${args[@]}" "$@"
