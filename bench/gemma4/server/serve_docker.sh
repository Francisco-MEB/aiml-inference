#!/usr/bin/env bash
# Same server, in the club's pinned Docker image. NOT YET TESTED on this host: it needs
# docker-group access. Set "server.mode": "docker" in the config to make run_bench.py use it.
#   OLLAMA_NUM_PARALLEL=16 OLLAMA_CONTEXT_LENGTH=4096 server/serve_docker.sh start
set -euo pipefail
IMAGE="${OLLAMA_BENCH_IMAGE:-ollama/ollama:0.40.13@sha256:b2f0122c476bc9fe7ce679d8bb4bd555351e25e2554eefb156f61cb0a89cf429}"
PORT="${OLLAMA_BENCH_PORT:-11435}"
NAME="ollama-bench-${USER}-${PORT}"
VOLUME="${OLLAMA_BENCH_VOLUME:-ollama-bench-models-${USER}}"
case "${1:-}" in
  start)
    : "${OLLAMA_NUM_PARALLEL:?set OLLAMA_NUM_PARALLEL}"
    : "${OLLAMA_CONTEXT_LENGTH:?set OLLAMA_CONTEXT_LENGTH}"
    docker run -d --rm --gpus all --name "$NAME" \
      -p "127.0.0.1:${PORT}:11434" -v "${VOLUME}:/root/.ollama" \
      -e OLLAMA_NUM_PARALLEL -e OLLAMA_CONTEXT_LENGTH \
      -e OLLAMA_MAX_LOADED_MODELS="${OLLAMA_MAX_LOADED_MODELS:-1}" \
      -e OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-30m}" \
      -e OLLAMA_NO_CLOUD="${OLLAMA_NO_CLOUD:-1}" \
      "$IMAGE"
    for _ in $(seq 60); do curl -fs -o /dev/null "http://127.0.0.1:$PORT/api/version" && { echo "up on :$PORT ($NAME)"; exit 0; }; sleep 0.5; done
    echo "container did not come up: docker logs $NAME" >&2; exit 1 ;;
  stop) docker stop "$NAME" ;;
  status) docker ps --filter "name=$NAME"; docker inspect --format '{{.Image}}' "$NAME" 2>/dev/null || true ;;
  logs) docker logs "$NAME" ;;
  *) echo "usage: $0 start|stop|status|logs" >&2; exit 2 ;;
esac
