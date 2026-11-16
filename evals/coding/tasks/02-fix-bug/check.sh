#!/usr/bin/env bash
# Pass if the tests pass and the test file is untouched.
set -euo pipefail
git diff --quiet HEAD -- tests/test_chat_stats.py || { echo "test file was modified"; exit 1; }
python3 -m unittest tests.test_chat_stats 2>&1 | tail -3
