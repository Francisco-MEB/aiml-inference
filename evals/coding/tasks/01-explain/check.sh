#!/usr/bin/env bash
# Pass if ANSWER.md exists, covers most top-level directories, and nothing else changed.
set -euo pipefail
[[ -s ANSWER.md ]] || { echo "ANSWER.md missing or empty"; exit 1; }
hits=0
for d in architecture clients docs evals examples harness scripts serving; do
  grep -qi "$d" ANSWER.md && hits=$((hits + 1))
done
echo "directories mentioned: $hits/8"
(( hits >= 6 )) || exit 1
changed=$(git status --porcelain | grep -v ' ANSWER.md$' || true)
[[ -z "$changed" ]] || { echo "unexpected changes:"; echo "$changed"; exit 1; }
