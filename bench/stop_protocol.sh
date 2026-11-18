#!/usr/bin/env bash
# Stops a running protocol study cleanly: the level in progress is abandoned (its server and
# power logger are shut down), completed levels are kept. Rerun the same run_protocol.sh
# command later to resume where it stopped.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
STUDY=${1:?usage: bench/stop_protocol.sh <study_id>}
PAT="python[^ ]* .*harness/run_protocol\.py"
if ! pgrep -f "$PAT" > /dev/null; then echo "no protocol run is going"; exit 0; fi
pkill -TERM -f "$PAT"
for _ in $(seq 60); do pgrep -f "$PAT" > /dev/null || break; sleep 1; done
if pgrep -f "$PAT" > /dev/null; then
  echo "still running after 60 s; forcing"; pkill -KILL -f "$PAT"; pkill -TERM -f "harness/run_bench\.py"
  pkill -TERM -f "harness/power_logger\.py"; bench/server/serve_native.sh stop
fi
echo "stopped. Completed levels are kept in bench/results/$STUDY/."
echo "Resume with the same bench/run_protocol.sh command."
