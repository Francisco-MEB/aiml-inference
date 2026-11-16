#!/usr/bin/env bash
# Launch a vLLM OpenAI-compatible server with defaults that are safe on a
# shared Dell Pro GB10 (unified CPU/GPU memory). Override anything via env vars:
#
#   ./serve.sh                                            # campus default: Qwen3.5-35B-A3B (chat + coding)
#   MODEL=Qwen/Qwen3.5-9B GPU_UTIL=0.30 MAX_LEN=32768 ./serve.sh   # small model, small footprint
#   ./serve.sh --max-num-seqs 16                          # extra args go straight to vllm
#
# GPU_UTIL is a fraction of TOTAL memory (~121 GB on a GB10), and vLLM reserves
# all of it up front (weights + KV cache). Never leave it at vLLM's default of
# 0.9 on a node other people are using: it will starve or crash their jobs.
set -euo pipefail

DEFAULT_MODEL="Qwen/Qwen3.5-35B-A3B-FP8"   # ~36 GB, MoE with ~3B active params/token
DEFAULT_REVISION="9d1823d2dee688a6b25e77009dc727688c44936e"  # the exact weights we tested (2026-11-16)
MODEL="${MODEL:-$DEFAULT_MODEL}"
# Pin the default model to its tested revision so results stay reproducible if the
# upstream repo changes. Other models use their latest revision unless REVISION is set.
if [[ -z "${REVISION+set}" ]]; then
  if [[ "$MODEL" == "$DEFAULT_MODEL" ]]; then REVISION="$DEFAULT_REVISION"; else REVISION=""; fi
fi
NAME="${NAME:-chat}"                        # the model name clients (chat.py, ndcode) ask for
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
GPU_UTIL="${GPU_UTIL:-0.50}"                # ~60 GB: weights + KV cache for 128k-token sessions
MAX_LEN="${MAX_LEN:-131072}"                # coding agents need long context (tools + files)
# Model-specific options (defaults suit Qwen3.5). Set to empty / 0 for other models:
#   REASONING_PARSER= TOOL_PARSER=hermes TEXT_ONLY=0 MODEL=... ./serve.sh
REASONING_PARSER="${REASONING_PARSER-qwen3}"  # splits <think> output into its own field
TOOL_PARSER="${TOOL_PARSER-qwen3_coder}"      # enables tool calling (needed by coding agents)
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
if [[ -n "$REVISION" ]]; then args+=(--revision "$REVISION"); fi
if [[ -n "$REASONING_PARSER" ]]; then args+=(--reasoning-parser "$REASONING_PARSER"); fi
if [[ -n "$TOOL_PARSER" ]]; then args+=(--enable-auto-tool-choice --tool-call-parser "$TOOL_PARSER"); fi
if [[ "$TEXT_ONLY" == 1 ]]; then args+=(--language-model-only); fi

echo "Serving $MODEL${REVISION:+@${REVISION:0:8}} as '$NAME' on http://$HOST:$PORT/v1 (gpu-memory-utilization=$GPU_UTIL, max-model-len=$MAX_LEN, tools=${TOOL_PARSER:-off})" >&2
exec uv run vllm serve "$MODEL" "${args[@]}" "$@"
