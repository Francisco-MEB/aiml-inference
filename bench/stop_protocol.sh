#!/usr/bin/env bash
# Stops a running protocol study cleanly: the level in progress is abandoned (its server and
# power logger are shut down), completed levels are kept. Rerun the same run_protocol.sh
# command later to resume where it stopped.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
STUDY=${1:?usage: bench/stop_protocol.sh <study_id>}
PAT="python[^ ]* .*harness/run_protocol\.py"
if ! pgrep -f "$PAT" > /dev/null; then echo "no protocol run is going"; exit 0; fi
pkill -TERM -f "$PAT"
for _ in $(seq 120); do pgrep -f "$PAT" > /dev/null || break; sleep 1; done
if pgrep -f "$PAT" > /dev/null; then
  echo "still running after 120 s; forcing"; pkill -KILL -f "$PAT"; pkill -TERM -f "harness/run_bench\.py"
  pkill -TERM -f "harness/power_logger\.py"
fi
# The benchmark server runs in its own process group; make sure it is gone.
PIDF=bench/.run/bench_server.pid
if [[ -f $PIDF ]]; then
  PGID=$(cat "$PIDF")
  if kill -0 "$PGID" 2> /dev/null; then
    kill -TERM -- "-$PGID" 2> /dev/null
    for _ in $(seq 60); do kill -0 "$PGID" 2> /dev/null || break; sleep 1; done
    kill -0 "$PGID" 2> /dev/null && kill -KILL -- "-$PGID" 2> /dev/null
  fi
  rm -f "$PIDF"
fi
echo "stopped. Completed levels are kept in bench/results/$STUDY/."
echo "Resume with the same bench/run_protocol.sh command."
