#!/usr/bin/env bash
# Start/stop a PRIVATE Ollama server (pinned native binary) under the current user.
# It never touches the system ollama.service on :11434.
#   OLLAMA_NUM_PARALLEL=16 OLLAMA_CONTEXT_LENGTH=4096 server/serve_native.sh start
#   server/serve_native.sh stop | status
# Both variables are required on start because changing them needs a restart.
set -euo pipefail
BENCH="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VER="${OLLAMA_BENCH_VERSION:-0.40.13}"
PORT="${OLLAMA_BENCH_PORT:-11435}"
BIN="${OLLAMA_BENCH_BIN:-$BENCH/.tools/ollama-$VER/bin/ollama}"
RUN="$BENCH/.run"; mkdir -p "$RUN"
PIDF="$RUN/ollama-$PORT.pid"
LOG="${OLLAMA_BENCH_LOG:-$RUN/ollama-$PORT.log}"
up() { curl -fs -o /dev/null --max-time 2 "http://127.0.0.1:$PORT/api/version"; }

case "${1:-}" in
  start)
    [[ -x "$BIN" ]] || { echo "missing $BIN; run server/install_ollama_native.sh" >&2; exit 1; }
    if up; then echo "port $PORT already has a server; refusing to start a second one" >&2; exit 1; fi
    : "${OLLAMA_NUM_PARALLEL:?set OLLAMA_NUM_PARALLEL}"
    : "${OLLAMA_CONTEXT_LENGTH:?set OLLAMA_CONTEXT_LENGTH}"
    export OLLAMA_HOST="127.0.0.1:$PORT"
    export OLLAMA_MODELS="${OLLAMA_MODELS:-$HOME/ollama-bench-models}"
    export OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-1}"
    export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-30m}"
    export OLLAMA_NO_CLOUD="${OLLAMA_NO_CLOUD:-1}"
    export OLLAMA_NUM_PARALLEL OLLAMA_CONTEXT_LENGTH
    mkdir -p "$OLLAMA_MODELS"
    setsid nohup "$BIN" serve >"$LOG" 2>&1 < /dev/null &
    echo $! > "$PIDF"
    for _ in $(seq 60); do up && { echo "up on :$PORT pid $(cat "$PIDF") log $LOG"; exit 0; }; sleep 0.5; done
    echo "server did not come up; see $LOG" >&2; exit 1 ;;
  stop)
    [[ -f "$PIDF" ]] || { echo "no pid file; nothing started by this script"; exit 0; }
    PID="$(cat "$PIDF")"
    if kill -0 "$PID" 2>/dev/null; then
      kill "$PID"; for _ in $(seq 40); do kill -0 "$PID" 2>/dev/null || break; sleep 0.5; done
    fi
    rm -f "$PIDF"; echo "stopped" ;;
  status)
    if up; then echo "up on :$PORT"; curl -s "http://127.0.0.1:$PORT/api/ps"; echo; else echo "down"; fi ;;
  *) echo "usage: $0 start|stop|status" >&2; exit 2 ;;
esac
