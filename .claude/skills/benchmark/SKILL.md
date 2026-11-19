---
name: benchmark
description: Run a club inference benchmark that follows the aiml-bench/1.0 protocol (bench/benchmark-protocol.md) - pick models and quantization, calibrate, estimate the time on this machine, let the member choose how far to push simultaneous users (up to 128), run it detached from the session, then validate and report. Use when a member wants to benchmark, compare or measure a model's speed, user capacity or throughput on a club machine, or asks how long a benchmark will take.
---

# Club benchmark (aiml-bench/1.0)

The protocol is `bench/benchmark-protocol.md`. **Read it before anything else.** It fixes the
prompts, answer length, user levels, server settings and column names, so that every member's
results line up. Never change a fixed value to make a run faster or fit a machine. If something
can't follow the protocol, stop and tell the member what and why.

Everything runs from the repo root with the repo's `.venv` (`uv sync` once if it's missing).
`$PY` below is `.venv/bin/python`.

## 1. Is the machine free?

```sh
$PY bench/harness/preflight.py
```

On `STOP`, show the member who or what is on the GPU and stop. Don't load a model on a shared
machine someone else is using. Never touch another person's server, container or model, and
never the system Ollama service.

Also check that the session closing won't kill the run (protocol section 9):
`loginctl show-user $USER -p Linger` and `grep KillUserProcesses /etc/systemd/logind.conf`.
If `KillUserProcesses=yes` and `Linger=no`, tell the member before going further.

## 2. Ask the member

Use one question round (AskUserQuestion) for:
- **Models**: the exact Ollama tags. Don't guess tag names. The next step checks them in the
  registry.
- **Quantization**: the member's choice (e.g. `q4_K_M`, `Q4_0` QAT, `q8_0`). It must be the same
  for every model in the study. Say that it will be recorded in every row, the folder name and
  the spreadsheet.
- **Backend**: Ollama in Docker if their account can use Docker (`docker ps` works), otherwise
  native Ollama on port 11435. The deviation is recorded automatically.

## 3. Write the study config

1. Copy `bench/configs/TEMPLATE.json` to `bench/configs/<study_id>.json`, with
   `study_id = <family>_<quant>_<machine>_<YYYYMMDD>`.
2. Fill every `FILL_ME`: study id, member, machine (`hostname`), hardware, quantization, one
   model entry per tag (key `<family>-<size>[-a<active>]`, label, arch, params, active params,
   quantization, `thinking_off`, `extra_body`). For native mode on a host with Ollama already
   installed, set `server.native.binary_path` and `binary_sha256` (`sha256sum $(which ollama)`)
   and check that its `--version` equals `server.ollama_version`.
3. Fill the checksums from the registry. This also proves each tag exists:
   ```sh
   $PY bench/harness/fill_digests.py bench/configs/<study_id>.json
   ```
   If the registry's `file_type` doesn't match the member's quantization, say so and ask before
   continuing.
4. Thinking off: for Ollama, `extra_body: {"reasoning_effort": "none"}`. If the model can't turn
   thinking off (e.g. gpt-oss), use its lowest setting and write that in `thinking_off`.
5. Check it:
   ```sh
   $PY -c "import sys,json; sys.path.insert(0,'bench/harness'); import protocol; print(protocol.check_config(json.load(open('bench/configs/<study_id>.json'))) or 'compliant')"
   ```

## 4. Calibrate (levels 1, 2, 4)

```sh
bench/run_protocol.sh --config bench/configs/<study_id>.json --calibrate
bench/status.sh <study_id>
```

This pulls each model, verifies its weights file, and measures levels 1, 2 and 4. That is real
data and stays in the study. Watch the status until it says `done`, and report any `FAILED`
line from `bench/.run/<study_id>.log`.

## 5. Show the time, then let the member choose

```sh
$PY bench/harness/estimate_time.py --config bench/configs/<study_id>.json
```

Show the member the table for each mode (`stop`, `all`, `cap:10`), per model and in total. Say
it's a range: actual time was 0.73× to 2.05× of the projection on the GB10 reference, and
64 or 128 users is the least certain. Then ask (AskUserQuestion):
- **Mode** for 8 to 128 users: `stop` (recommended), `all`, or `cap:<minutes per level>`, each
  option with its estimated time.
- **Total cap** in hours (no new level starts after it). Default: the high end of the chosen
  mode's range, rounded up.

Before a run longer than an hour, remind them to tell the club channel.

## 6. Run it detached

```sh
bench/run_protocol.sh --config bench/configs/<study_id>.json --mode <mode> --cap-hours <h>
```

It keeps running if this session, VS Code or SSH closes. Give the member these commands:
- `bench/status.sh <study_id>`: progress and remaining time
- `tail -f bench/.run/<study_id>.log`: live log
- `bench/stop_protocol.sh <study_id>`: clean stop; completed levels are kept
- the same `run_protocol.sh` command again resumes after a stop or a crash

Don't poll in a loop while it runs. When the member comes back, read `status.sh` and the log.

## 7. When it's done

```sh
$PY bench/harness/validate_results.py <study_id>
```

- On `FAIL`, explain each one; don't hide them.
- The merged data and `findings.xlsx` are in `bench/results/<study_id>/combined/`. Rebuild them
  with `$PY bench/harness/merge_study.py <study_id>` if needed.
- Summarize for the member: single-user speed, capacity per workload under the interactive
  target, peak throughput, what limited capacity, which workloads the `stop` rule ended where,
  the actual time against the estimate (Timing sheet), and the caveats (GPU-only power, one
  machine, the quantization).
- Commit the config and results on the member's branch only if they ask, and never push
  without being asked.

## Don'ts

- Don't edit `request`, `prompts`, `sweep` or `thresholds` in a study config, or pass
  `--test-overrides` for real results. Overrides are for testing the harness; those runs are
  marked as deviating and are never merged.
- Don't mix quantizations within one study.
- Don't invent tags, versions or numbers. Check them or say you couldn't.
- Don't run two studies at once on one machine.
