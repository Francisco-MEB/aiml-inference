# Raw coding-eval results

CSV output of [`evals/coding/run.sh`](../../../evals/coding/run.sh), one file per run. The numbers in [`../README.md`](../README.md) come from these files.

Columns: `task, arm, pass (1/0), seconds (wall time), gen_tokens, prompt_tokens (from vLLM /metrics; empty for the cloud arm), cost_usd (cloud arm only)`.

| File | What it is |
|---|---|
| `2026-11-16-baseline.csv` | All 5 tasks × 3 arms, **re-scored** with the fixed 05 check (`evals/coding/rescore.sh`). Use this one. |
| `2026-11-16-baseline-raw.csv` | The same run as originally scored (05 / claude-local was a false fail). Kept for the record. |
| `2026-11-16-ndcode-rules-v1-rep{1,2}.csv` | ndcode only, with the first version of `clients/AGENTS.md`. |
| `2026-11-16-ndcode-rules-v2-rep1.csv` | ndcode only, with the current `clients/AGENTS.md` (adds "exercise the feature end to end"). |
| `2026-11-16-ndcode-rules-v2-03only-rep{2,3}.csv` | Two extra runs of task 03 with the current rules. |

Setup for all runs: one Dell Pro GB10, vLLM 0.33.0, Qwen3.5-35B-A3B-FP8 @ `9d1823d2`, `GPU_UTIL=0.50`, `MAX_LEN=131072`, OpenCode v2.2.5. Tasks ran sequentially (single user). Per-run agent logs weren't kept. To produce new ones, re-run `evals/coding/run.sh`, which writes logs next to its CSV.
