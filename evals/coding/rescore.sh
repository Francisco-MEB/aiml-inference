#!/usr/bin/env bash
# Re-run the current check.sh scripts against an existing eval output dir
# (no agents are re-run). Use after fixing a check. Writes results-rescored.csv.
#
#   evals/coding/rescore.sh /tmp/coding-eval-20261116-023500
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${1:?usage: rescore.sh <eval output dir>}"
VLLM="${VLLM_URL:-http://127.0.0.1:8000}"
IN="$OUT/results.csv"; NEW="$OUT/results-rescored.csv"
[[ -f "$IN" ]] || { echo "no $IN" >&2; exit 1; }

head -1 "$IN" > "$NEW"
tail -n +2 "$IN" | while IFS=, read -r task arm pass rest; do
  work="$OUT/$task.$arm/repo"
  if [[ -d "$work" ]] && (cd "$work" && EVAL_BASE_URL="$VLLM/v1" bash "$HERE/tasks/$task/check.sh") > "$OUT/$task.$arm/check-rescored.log" 2>&1; then
    newpass=1
  else
    newpass=0
  fi
  [[ "$newpass" != "$pass" ]] && echo "changed: $task/$arm $pass -> $newpass" >&2
  echo "$task,$arm,$newpass,$rest" >> "$NEW"
done
column -s, -t < "$NEW"
