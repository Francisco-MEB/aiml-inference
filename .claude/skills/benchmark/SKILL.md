---
name: benchmark
description: Run a club inference benchmark under the aiml-bench/2.0 protocol (bench/benchmark-protocol.md) on vLLM (Dell GB10) or MLX (Mac). Use when a member names their assignment (a model family like "Qwen" or one model) to benchmark, asks to measure a model's speed, user capacity, throughput or quality on a club machine, or asks how long a benchmark will take.
---

# Club benchmark (aiml-bench/2.0)

The member says what they were assigned: a family (`qwen`, `gemma`, `gpt-oss`, `llama`, `muse`,
`mistral`, `glm`) or one model (catalog key or Hugging Face repo). Everything else comes from
the catalog and the machine; the member's only decisions are the high-level mode and the time
cap (step 6). The protocol is `bench/benchmark-protocol.md`: read sections 1, 5 and 10 before
step 1, and look up the rest when a step needs it. Values it marks as fixed stay fixed; when
something can't follow the protocol, stop and tell the member what and why.

Run from the repo root. `$PY` is `.venv/bin/python` (run `uv sync` once if it's missing).

## 1. Machine is free and ready

```sh
$PY bench/harness/preflight.py
```

- `STOP`: show the member who or what is on the machine and end here. Their run waits for a
  free machine; other people's servers, containers and models stay untouched.
- Engine environment, once per machine: `cd serving/vllm && uv sync` on a Dell,
  `cd serving/mlx && uv sync` on a Mac.
- `HF_TOKEN` exported (`echo ${HF_TOKEN:+set}`). Without it the gated GPQA task is skipped and
  gated models (Llama 4 Scout) can't download; tell the member, then continue if they agree.
- Linux: `loginctl show-user $USER -p Linger` and `grep KillUserProcesses /etc/systemd/logind.conf`
  (protocol section 10). Mac: remind them to run `caffeinate -dims` for the length of the run.

Done when preflight says `PASS` and the engine env exists.

## 2. Plan the study

```sh
$PY bench/harness/catalog.py plan <assignment>
```

Show the member the plan as printed: machine and budget, study quantization and publisher, each
model (size class, dense or MoE, GB), every **left out** model with its reason, and any
DEVIATION line. A name the catalog refuses goes back to the benchmarks lead; offer the closest
catalog names. Done when the member has seen the plan.

## 3. Write and pin the config

```sh
$PY bench/harness/catalog.py plan <assignment> --member <name or netid> --write
$PY bench/harness/fill_digests.py bench/configs/<study_id>.json
$PY -c "import sys,json; sys.path.insert(0,'bench/harness'); import protocol; print(protocol.check_config(json.load(open('bench/configs/<study_id>.json'))) or 'compliant')"
```

Done when `fill_digests.py` lists every model's weight files and the check prints `compliant`.

## 4. Calibrate (levels 1, 2, 4)

```sh
bench/run_protocol.sh --config bench/configs/<study_id>.json --calibrate
```

It downloads each model at its pinned commit, verifies every weight file, and measures levels
1, 2 and 4: real data that stays in the study. Large models take a while to download; give the
member `bench/status.sh <study_id>` and `tail -f bench/.run/<study_id>.log`, and check back
when they return. Done when status says `done`; report every `FAILED` or `SERVER FAILED` line
from the log with its reason.

## 5. Estimate the time

```sh
$PY bench/harness/estimate_time.py --config bench/configs/<study_id>.json
```

Build one table per mode (`stop`, `all`, `cap:10`): minutes per model and in total, plus about
0.5 to 1 hour per model for the quality phase (protocol section 7). State it as a range:
actual time ran 0.73× to 2.05× of the projection on the GB10 reference, least certain at 64 and
128 users.

## 6. Member chooses

One AskUserQuestion round:
- **Mode** for 8 to 128 users: `stop` (recommended), `all`, or `cap:<minutes per level>`, each
  option with its estimated time.
- **Total cap** in hours (no new level starts after it). Default: the high end of the chosen
  mode's range plus the quality phase, rounded up.

For a run longer than an hour, remind them to book the machine in the club channel.

## 7. Run it detached

```sh
bench/run_protocol.sh --config bench/configs/<study_id>.json --mode <mode> --cap-hours <h>
```

This runs the remaining levels, then each model's quality phase (skipped and noted when the
same repo@revision already has a quality result anywhere in `bench/results/`). It survives
this session, VS Code and SSH closing. Give the member:
- `bench/status.sh <study_id>`: progress and remaining time
- `tail -f bench/.run/<study_id>.log`: live log
- `bench/stop_protocol.sh <study_id>`: clean stop; completed levels are kept
- the same `run_protocol.sh` command again: resume after a stop or a crash

Then start the status loop yourself, without asking: invoke the `loop` skill with the
argument below (`<study_id>` filled in). It reports every 30 minutes and ends itself when the
run does. Tell the member it only runs while this Claude Code session stays open (on a Dell,
inside tmux), that the benchmark keeps going either way, and that *stop the loop* ends it.

```
30m Run bench/status.sh <study_id> and read the last lines of bench/.run/<study_id>.log. In one or two lines, tell me whether my benchmark is on track: current model and level, time remaining, and any FAILED lines. When status says done or the process is no longer running, give me a short final summary, stop the loop, and offer to validate and report (benchmark skill step 8).
```

## 8. Validate and report

```sh
$PY bench/harness/validate_results.py <study_id>
```

- Explain every `FAIL` line and what fixes it; the study is shareable only when it passes.
- `bench/results/<study_id>/combined/findings.xlsx` holds the merged data (rebuild with
  `$PY bench/harness/merge_study.py <study_id>`).
- Summarize: per model (small to large) single-user speed, capacity per workload under the
  interactive target, peak throughput, what limited capacity, quality per task, where `stop`
  ended each workload, actual time against the estimate (Timing sheet), and the caveats (the
  study's quantization, thinking off, GPU-only power, one machine).

## 9. Share

Once validation passes, and the member says to:

```sh
git switch -c results/<study_id>
git add bench/configs/<study_id>.json bench/results/<study_id>
git commit -m "results: <study_id>"
git push -u origin results/<study_id>
```

Without push access on the machine, give them
`tar czf <study_id>.tgz bench/configs/<study_id>.json bench/results/<study_id>` to send to the
benchmarks lead instead.

## Guardrails

- Models, builds and quantizations come only from the catalog; one study holds one quantization.
- Fixed sections of a config (`request`, `prompts`, `sweep`, `thresholds`, `quality`) keep the
  values `catalog.py` wrote. `--test-overrides` is for testing the harness: those runs are
  marked as deviating and never merged.
- Report tags, versions and numbers as the tools printed them, or say you couldn't check.
- One study at a time per machine.
