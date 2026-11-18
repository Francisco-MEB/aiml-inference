#!/usr/bin/env bash
# Progress of a protocol study: bench/status.sh <study_id>
cd "$(dirname "${BASH_SOURCE[0]}")/.."
STUDY=${1:?usage: bench/status.sh <study_id>}
S="bench/.run/$STUDY.status.json"
[[ -f $S ]] || { echo "no status for $STUDY (never started here?)"; exit 1; }
python3 - "$S" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
print(f"study {d['study_id']}  mode {d['mode']}  state: {d['state']}  (updated {d['updated_utc']})")
if d.get("current"):
    c = d["current"]; print(f"now: {c['model']} at {c['users']} users, workloads {', '.join(c['workloads'])}")
e = d.get("estimate_remaining_min")
if e:
    print(f"remaining: about {e['low']} to {e['high']} min")
print(f"levels done: {len(d['done'])}" + (f", last: {d['done'][-1]['model']} {d['done'][-1]['users']} users ({d['done'][-1]['minutes']} min)" if d['done'] else ""))
for s in d.get("skipped", []):
    print("skipped:", ", ".join(f"{k}={v}" for k, v in s.items()))
PY
if pgrep -f "python[^ ]* .*harness/run_protocol\.py" > /dev/null; then echo "process: running"; else echo "process: not running"; fi
tail -n 3 "bench/.run/$STUDY.log" 2>/dev/null
