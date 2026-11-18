#!/usr/bin/env bash
# Speed sweep with the server's parallel slots equal to the number of concurrent users.
# One run directory per model and level (the server must restart to change the slot count);
# combine them afterwards with harness/merge_runs.py.
#   bench/gemma4/run_matched.sh 31b 26b-a4b 12b
# Optional: LEVELS="1 2 4 8 16"  DEADLINE_MIN=110 (no new level starts after that many minutes)
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
START=$(date +%s)
for model in "$@"; do
  for n in ${LEVELS:-1 2 4 8 16}; do
    if [[ -n "${DEADLINE_MIN:-}" ]] && (( ($(date +%s) - START) / 60 >= DEADLINE_MIN )); then
      echo "DEADLINE reached before $model level $n"; echo MATCHED_DONE; exit 0
    fi
    echo "### $model level $n ($(date -u +%H:%M:%SZ))"
    .venv/bin/python -u bench/gemma4/harness/run_bench.py --models "$model" --skip-pull \
      --levels "$n" --num-parallel "$n" || { echo "FAILED $model level $n"; echo MATCHED_DONE; exit 1; }
  done
done
echo MATCHED_DONE
