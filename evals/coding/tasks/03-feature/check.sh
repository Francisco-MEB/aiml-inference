#!/usr/bin/env bash
# Pass if --json works against the live server, a new test exists and passes, and the README mentions it.
set -euo pipefail
BASE="${EVAL_BASE_URL:-http://127.0.0.1:8000/v1}"
out=$(python3 harness/chat.py --base-url "$BASE" --prompt "Reply with just the word OK." --json --max-tokens 50)
python3 - "$out" <<'EOF'
import json, sys
obj = json.loads(sys.argv[1])
missing = {"answer", "ttft", "tokens", "tok_s", "total"} - set(obj)
assert not missing, f"missing keys: {missing}"
assert isinstance(obj["answer"], str) and obj["answer"].strip(), "empty answer"
print("json ok:", {k: obj[k] for k in ("tokens", "tok_s")})
EOF
ls tests/test_*.py >/dev/null 2>&1 || { echo "no test file under tests/"; exit 1; }
python3 -m unittest discover -s tests 2>&1 | tail -3
grep -q -- "--json" harness/README.md || { echo "README doesn't mention --json"; exit 1; }
