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
uv run bench/download_models.py qwen                   # fetch a study's models ahead of time
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

**Weights** live in the repo's `models/` folder (gitignored), one folder per machine, each a
Hugging Face cache (`<folder>/models--<org>--<name>/snapshots/<commit>/`) that the harness
verifies by checksum:

| Folder | Machine | Lineup (`bench/download_models.py`, no family named) |
|---|---|---|
| `models/dell/` | Dell GB10 | 12 models, NVFP4 / W4A16, about 343 GB |
| `models/mac/studio/` | Mac Studio, 128 GB | the same 12 models (GLM 4 9B for GLM 4.6V Flash), MLX 4-bit, about 296 GB |
| `models/mac/mini/` | Mac mini, 32 GB | the 9 that fit (no large models; Kimi is Moonlight only), MLX 4-bit, about 114 GB |

The harness picks the folder from the machine it runs on (a Mac by its model name: other Macs
get `models/mac/<model>`). `bench/download_models.py` fills a folder with exactly what the
benchmark runs there; `--machine studio` (or `mini`, `dell`) fetches another machine's builds
from wherever you are, and `--dry-run` lists them first. Copy a folder to its machine as is
(`rsync -a models/mac/studio/ <studio>:<repo>/models/mac/studio/`, with the same catalog commit
checked out there), or run the script on that machine. Set `HF_HUB_CACHE` to keep the weights
somewhere else, such as one copy shared between accounts.

## Downloading the models onto a machine (one person per machine)

Same steps on the Dell, the Mac Studio and each Mac mini: the script recognizes the machine and
fetches that machine's builds into its folder (table above). It needs only `uv`, which brings
its own Python, and no Hugging Face token for the Qwen, Gemma, GLM and Kimi models.

```sh
# 0. Once: uv, and the repo at the commit with the current catalog (models are pinned there).
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone <repo url> aiml-inference && cd aiml-inference      # or: git pull

# 1. Check: the "machine:" line and the folder must match your machine in the table above,
#    and the free space must exceed the size it lists.
uv run bench/download_models.py --dry-run

# 2. Download: takes hours, keeps going if you log out, a re-run resumes. On a Mac, caffeinate
#    keeps it awake (on the Dell, drop "caffeinate -i").
nohup caffeinate -i uv run bench/download_models.py > ~/download-models.log 2>&1 &
tail -f ~/download-models.log

# 3. Done when the log ends with "all N models verified". Re-check any time:
uv run bench/download_models.py --verify
```

The second Mac mini can copy the first one's folder instead of downloading it again:
`rsync -a models/mac/mini/ <mini2>:aiml-inference/models/mac/mini/`, then run step 3 there. If
step 3 reports MISMATCH, delete that model's `models--<org>--<name>` folder and repeat step 2.

## Layout

| Path | What |
|---|---|
| `benchmark-protocol.md` | the protocol |
| `example-output.md` | what a run looks like in Claude Code and what the results contain (real pilot numbers) |
| `catalog/models.json` | every family's models and their pinned builds per engine |
| `configs/TEMPLATE.json` | the fixed sections of a study config (configs are written by `catalog.py`) |
| `prompts/prompts_v1.jsonl` | the three fixed workloads (checksum pinned) |
| `download_models.py` | fetch the models a study runs on a machine into its `models/` folder |
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
format, highest precision first; a family's `prefer` overrides that order, with the
reason in its notes), then `.venv/bin/python bench/harness/catalog.py pin` to fill
revision, size, format and gating from the Hub, and `catalog.py plan <family> --machine dell`
(or `mac:<GiB>`) to see the study it produces. Commit the catalog.

## Studies so far

| Study | Notes |
|---|---|
| [`gemma4_qat-q4_0_promaxgb10-2c9c_20261115`](results/gemma4_qat-q4_0_promaxgb10-2c9c_20261115/README.md) | Gemma 4 E4B, 12B, 26B A4B, 31B on one GB10 under Ollama, levels 1 to 16. `pre-1.0`: same prompts, Ollama engine; not comparable with 2.0. Its config is `legacy/configs/gemma4_qat-q4_0_ollama_native.json`. |
