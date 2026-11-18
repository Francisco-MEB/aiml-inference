# Club inference benchmarks

Everything for measuring how fast a model answers and how many simultaneous users one machine
can serve, in a way that lines up across members, models and machines.

- **The standard:** [benchmark-protocol.md](benchmark-protocol.md) (`aiml-bench/1.0`): fixed
  prompts, 256-token answers, 1 to 128 simultaneous users, column names, how to estimate time,
  how to run unattended.
- **With Claude Code:** ask it to run a benchmark; the `benchmark` skill
  (`.claude/skills/benchmark/SKILL.md`) follows the protocol step by step.

## Quick start

```sh
uv sync                                                    # once
python3 bench/harness/preflight.py                         # is anyone else on the GPU?
cp bench/configs/TEMPLATE.json bench/configs/<study_id>.json   # fill every FILL_ME
.venv/bin/python bench/harness/fill_digests.py bench/configs/<study_id>.json
bench/run_protocol.sh --config bench/configs/<study_id>.json --calibrate
.venv/bin/python bench/harness/estimate_time.py --config bench/configs/<study_id>.json
bench/run_protocol.sh --config bench/configs/<study_id>.json --mode stop --cap-hours 6
bench/status.sh <study_id>
.venv/bin/python bench/harness/validate_results.py <study_id>
```

Results: `bench/results/<study_id>/combined/findings.xlsx`.

## Layout

| Path | What |
|---|---|
| `benchmark-protocol.md` | the protocol |
| `configs/TEMPLATE.json` | starting point for a study config |
| `prompts/prompts_v1.jsonl` | the three fixed workloads (checksum pinned) |
| `run_protocol.sh`, `status.sh`, `stop_protocol.sh` | start detached, check, stop |
| `harness/run_protocol.py` | level-by-level orchestrator: calibration, modes, stop rule, resume |
| `harness/run_bench.py` | one model at one level (called by the orchestrator) |
| `harness/estimate_time.py` | time projection per machine and mode; `--backtest` checks the method |
| `harness/merge_study.py`, `make_report.py` | combined data and `findings.xlsx` |
| `harness/validate_results.py` | protocol compliance check before sharing |
| `harness/fill_digests.py` | registry checksums into a config |
| `harness/protocol.py` | the fixed protocol values in code |
| `server/` | private Ollama server: native binary or pinned Docker image |
| `lm_eval_tasks/` | quality tasks (optional, `--with-quality`) |
| `results/` | one folder per study |
| `legacy/` | scripts used for the first Gemma 4 study, kept for reference |

## Studies so far

| Study | Notes |
|---|---|
| [`gemma4_qat-q4_0_promaxgb10-2c9c_20261115`](results/gemma4_qat-q4_0_promaxgb10-2c9c_20261115/README.md) | Gemma 4 E4B, 12B, 26B A4B, 31B on one GB10, levels 1 to 16. Ran before the protocol (`pre-1.0`): same prompts and settings, without the new row columns. Its config is `configs/gemma4_qat-q4_0_ollama_native.json`. |
