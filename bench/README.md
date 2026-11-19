# Club inference benchmarks

Everything for measuring how fast a model answers, how many simultaneous users one machine can
serve, and how good its answers are, in a way that lines up across members, models and
machines. vLLM on the Dell GB10s, MLX on the Macs.

- **The standard:** [benchmark-protocol.md](benchmark-protocol.md) (`aiml-bench/2.0`): the
  catalog and the study rule, fixed prompts, 256-token answers, 1 to 128 simultaneous users,
  the quality suite, column names, how to estimate time, how to run unattended.
- **With Claude Code:** say your assignment ("benchmark the Qwen family"); the `benchmark`
  skill (`.claude/skills/benchmark/SKILL.md`) runs the protocol step by step.

## Quick start

```sh
uv sync                                                # once: harness + lm-eval
(cd serving/vllm && uv sync)                           # once per Dell  (Mac: serving/mlx)
export HF_TOKEN=...                                    # gated GPQA dataset, Llama 4 Scout
.venv/bin/python bench/harness/preflight.py            # is anyone else on the machine?
.venv/bin/python bench/harness/catalog.py plan qwen    # what a Qwen study runs here
.venv/bin/python bench/harness/catalog.py plan qwen --member <you> --write
.venv/bin/python bench/harness/fill_digests.py bench/configs/<study_id>.json
bench/run_protocol.sh --config bench/configs/<study_id>.json --calibrate
.venv/bin/python bench/harness/estimate_time.py --config bench/configs/<study_id>.json
bench/run_protocol.sh --config bench/configs/<study_id>.json --mode stop --cap-hours 12
bench/status.sh <study_id>
.venv/bin/python bench/harness/validate_results.py <study_id>
```

Results: `bench/results/<study_id>/combined/findings.xlsx`.

## Layout

| Path | What |
|---|---|
| `benchmark-protocol.md` | the protocol |
| `catalog/models.json` | every family's models and their pinned builds per engine |
| `configs/TEMPLATE.json` | the fixed sections of a study config (configs are written by `catalog.py`) |
| `prompts/prompts_v1.jsonl` | the three fixed workloads (checksum pinned) |
| `run_protocol.sh`, `status.sh`, `stop_protocol.sh` | start detached, check, stop |
| `harness/catalog.py` | list the catalog, plan a study on this machine, pin and verify builds |
| `harness/run_protocol.py` | level-by-level orchestrator: calibration, modes, stop rule, quality phase, resume |
| `harness/run_bench.py` | one model at one level, or one model's quality phase |
| `harness/engines.py` | the private vLLM / MLX server (`--dry-run` prints its exact command) |
| `harness/hf.py`, `fill_digests.py`, `pull_models.py` | Hugging Face pins, downloads, checksum verification |
| `harness/run_quality.py`, `lm_eval_bench.py` | the quality phase through lm-evaluation-harness |
| `harness/estimate_time.py` | time projection per machine and mode; `--backtest` checks the method |
| `harness/merge_study.py`, `make_report.py` | combined data and `findings.xlsx` |
| `harness/validate_results.py` | protocol compliance check before sharing |
| `harness/protocol.py` | the fixed protocol values in code |
| `tests/` | `.venv/bin/python -m unittest discover -s bench/tests` (no GPU needed: `fake_server.py`) |
| `results/` | one folder per study |
| `lm_eval_tasks/` | the club's own lm-eval tasks: `humaneval_chat` (HumanEval for chat endpoints) |
| `legacy/` | 1.0 (Ollama) server scripts, the Gemma config, the first study's scripts |

## Adding a model (benchmarks lead)

Add the model to its family in `catalog/models.json` (metadata, plus a build per engine and
format, highest precision first), then `.venv/bin/python bench/harness/catalog.py pin` to fill
revision, size, format and gating from the Hub, and `catalog.py plan <family> --machine dell`
(or `mac:<GiB>`) to see the study it produces. Commit the catalog.

## Studies so far

| Study | Notes |
|---|---|
| [`gemma4_qat-q4_0_promaxgb10-2c9c_20261115`](results/gemma4_qat-q4_0_promaxgb10-2c9c_20261115/README.md) | Gemma 4 E4B, 12B, 26B A4B, 31B on one GB10 under Ollama, levels 1 to 16. `pre-1.0`: same prompts, Ollama engine; not comparable with 2.0. Its config is `legacy/configs/gemma4_qat-q4_0_ollama_native.json`. |
