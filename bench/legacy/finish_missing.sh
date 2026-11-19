#!/usr/bin/env bash
# Runs whatever is still missing from the Gemma 4 session, one step at a time, then rebuilds
# the combined spreadsheet. Before each step it waits until nobody else is using the GPU.
#
#   setsid nohup bench/gemma4/finish_missing.sh > bench/gemma4/.run/finish.log 2>&1 < /dev/null &
#   tail -f bench/gemma4/.run/finish.log
#
# Order: 26B-A4B speed (slots = users), 12B speed (slots = users), missing quality runs.
# Settings (environment variables, all optional):
#   FREE_MIN=10       minutes the GPU must be free of other users before the first step
#                     (later steps only need the pre-flight check to pass)
#   MAX_WAIT_MIN=240  give up if it has not been free for that long
#   CAP_MIN=150       no new step starts after this many minutes of running
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
PY=.venv/bin/python
FREE_MIN=${FREE_MIN:-10}; MAX_WAIT_MIN=${MAX_WAIT_MIN:-240}; CAP_MIN=${CAP_MIN:-150}
START=$(date +%s); RUN_S=0
log() { echo "[$(date -u +%H:%M:%SZ)] $*"; }

wait_free() {             # returns 1 if the GPU never stayed free long enough
  local since="" t0; t0=$(date +%s)
  while true; do
    if $PY bench/gemma4/harness/preflight.py > /dev/null 2>&1; then
      since=${since:-$(date +%s)}
      (( $(date +%s) - since >= FREE_MIN * 60 )) && { FREE_MIN=0; return 0; }   # full wait only once
    else
      [[ -n "$since" ]] && log "someone started using the machine; waiting again"
      since=""
    fi
    (( $(date +%s) - t0 >= MAX_WAIT_MIN * 60 )) && return 1
    sleep 30
  done
}

step() {                  # step <description> <command...>
  local desc=$1; shift
  if (( RUN_S / 60 >= CAP_MIN )); then log "CAP reached ($CAP_MIN min of running); skipping: $desc"; return; fi
  log "waiting for a free GPU before: $desc"
  if ! wait_free; then log "GPU not free for $MAX_WAIT_MIN min; giving up"; finish; exit 1; fi
  log "START $desc"
  local t0; t0=$(date +%s)
  "$@" || log "FAILED $desc"
  RUN_S=$(( RUN_S + $(date +%s) - t0 ))
  log "END $desc ($(( ($(date +%s) - t0) / 60 )) min)"
}

finish() {
  bench/gemma4/server/serve_native.sh stop > /dev/null 2>&1
  log "building combined spreadsheet"
  $PY bench/gemma4/harness/merge_session.py && log "spreadsheet: bench/gemma4/results/combined_gemma4_gb10/findings.xlsx"
  log "FINISH_DONE (running time $(( RUN_S / 60 )) min, wall time $(( ($(date +%s) - START) / 60 )) min)"
}

for model in $($PY bench/gemma4/harness/merge_session.py missing-speed); do
  [[ "$model" == "31b" || "$model" == "e4b" ]] && continue      # already complete or not wanted
  for n in 1 2 4 8 16; do
    step "$model speed, $n users" $PY -u bench/gemma4/harness/run_bench.py --models "$model" \
      --skip-pull --levels "$n" --num-parallel "$n"
  done
done
while read -r task models; do
  limit=10; [[ "$task" == humaneval_chat ]] && limit=40
  for model in $models; do
    step "$model quality: $task" $PY -u bench/gemma4/harness/run_bench.py --models "$model" \
      --skip-pull --skip-speed --with-quality --quality-tasks "$task" --quality-limit "$limit"
  done
done < <($PY bench/gemma4/harness/merge_session.py missing-quality)
finish
