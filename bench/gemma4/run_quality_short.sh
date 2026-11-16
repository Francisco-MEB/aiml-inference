#!/usr/bin/env bash
# Short output-quality check per model: MMLU-Pro (N questions from each of the 14 subjects) and
# the chat HumanEval variant. Thinking off, temperature 0. HumanEval runs model-written code.
#   bench/gemma4/run_quality_short.sh e4b 12b 26b-a4b 31b
# Optional: MMLU_LIMIT=10 HUMANEVAL_LIMIT=40 DEADLINE_MIN=50 (no new model starts after that)
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
START=$(date +%s)
for model in "$@"; do
  if [[ -n "${DEADLINE_MIN:-}" ]] && (( ($(date +%s) - START) / 60 >= DEADLINE_MIN )); then
    echo "DEADLINE reached before quality for $model"; break
  fi
  echo "### quality $model ($(date -u +%H:%M:%SZ))"
  .venv/bin/python -u bench/gemma4/harness/run_bench.py --models "$model" --skip-pull --skip-speed \
    --with-quality --quality-tasks mmlu_pro --quality-limit "${MMLU_LIMIT:-10}" \
    || echo "FAILED mmlu_pro $model"
  .venv/bin/python -u bench/gemma4/harness/run_bench.py --models "$model" --skip-pull --skip-speed \
    --with-quality --quality-tasks humaneval_chat --quality-limit "${HUMANEVAL_LIMIT:-40}" \
    || echo "FAILED humaneval_chat $model"
done
echo QUALITY_DONE
