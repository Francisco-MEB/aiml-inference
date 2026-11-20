#!/usr/bin/env bash
# Starts a protocol study DETACHED from your session: it keeps running if you close VS Code,
# lose SSH or turn off your laptop. Same arguments as harness/run_protocol.py.
#
#   bench/run_protocol.sh --config bench/configs/<study>.json --calibrate
#   bench/run_protocol.sh --config bench/configs/<study>.json --mode stop --cap-hours 6
#
# Progress: bench/status.sh <study_id>     Stop: bench/stop_protocol.sh <study_id>
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY=.venv/bin/python; [[ -x $PY ]] || PY=python3
CONFIG=""; args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
  [[ "${args[i]}" == "--config" ]] && CONFIG="${args[i+1]}"
done
[[ -n "$CONFIG" ]] || { echo "usage: $0 --config bench/configs/<study>.json (--calibrate | --mode ...)"; exit 2; }
STUDY=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['study']['id'])" "$CONFIG")
mkdir -p bench/.run
if pgrep -f "python[^ ]* .*harness/run_protocol\.py" > /dev/null; then
  echo "a protocol run is already going on this machine (one at a time): bench/status.sh <study_id>"; exit 1
fi
# Absolute config path so the detached job doesn't depend on this shell's directory.
args=("${args[@]/#$CONFIG/$(realpath "$CONFIG")}")
LOG="bench/.run/$STUDY.log"
if command -v setsid > /dev/null; then          # Linux
  setsid nohup "$PY" -u bench/harness/run_protocol.py "${args[@]}" >> "$LOG" 2>&1 < /dev/null &
else                                             # macOS has no setsid: start a new session from Python
  nohup "$PY" -c 'import os, sys; os.setsid(); os.execv(sys.argv[1], sys.argv[1:])' \
    "$PY" -u bench/harness/run_protocol.py "${args[@]}" >> "$LOG" 2>&1 < /dev/null &
fi
PID=$!
if [[ "$(uname)" == "Darwin" ]]; then
  # Keep the Mac awake exactly as long as the run (an idle Mac sleeps even while it works).
  nohup caffeinate -dims -w "$PID" > /dev/null 2>&1 &
  echo "keeping this Mac awake until the run ends (caffeinate)"
fi
echo "started (pid $PID), detached from this session"
echo "  progress: bench/status.sh $STUDY"
echo "  log:      tail -f $LOG"
echo "  stop:     bench/stop_protocol.sh $STUDY"
