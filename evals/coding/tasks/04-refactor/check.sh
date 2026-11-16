#!/usr/bin/env bash
# Pass if the new module exists, chat.py uses it instead of parsing SSE itself, and chat.py still works live.
set -euo pipefail
BASE="${EVAL_BASE_URL:-http://127.0.0.1:8000/v1}"
grep -q "def iter_deltas" harness/openai_stream.py || { echo "iter_deltas not found"; exit 1; }
grep -q "openai_stream" harness/chat.py || { echo "chat.py doesn't import openai_stream"; exit 1; }
! grep -q 'startswith("data:")' harness/chat.py || { echo "chat.py still parses SSE lines itself"; exit 1; }
out=$(python3 harness/chat.py --base-url "$BASE" --prompt "Reply with just the word OK." --no-stats --max-tokens 50)
[[ -n "${out// /}" ]] || { echo "empty answer from refactored chat.py"; exit 1; }
echo "live answer: $out"
