# Coding-agent acceptance eval: results (2026-11-16)

**Question:** Is `ndcode` (OpenCode + our campus Qwen3.5-35B-A3B on one Dell Pro GB10) good enough for day-to-day coding work, and how close does it get to what people use today (Claude Code on cloud Claude)?

**Short answer:** It handles everyday tasks: explaining a codebase, fixing a bug from failing tests, a multi-file refactor, and writing docs. It passed **16 of 20** full-suite runs, at **$0 marginal cost**, with nothing leaving campus. It consistently fails one class of task: a new feature with a precise output spec, where it ships a subtly wrong result without running the feature end to end. Cloud Claude passed **5/5** at **$1.68** for the suite.

## Method

Five real tasks on this repo, each with an automatic check ([`evals/coding/tasks/`](../../evals/coding/tasks/)):

| # | Task | Check |
|---|---|---|
| 01 | Explain the repo layout in `ANSWER.md` | covers ≥ 6/8 directories, nothing else changed |
| 02 | Fix a planted bug that makes unit tests fail | tests pass, test file untouched |
| 03 | Add `--json` to `harness/chat.py` + test + docs | live run prints exactly one JSON object with the right keys, a new test passes, README mentions it |
| 04 | Extract SSE parsing into `harness/openai_stream.py` | module exists and is used; `python3 harness/chat.py` still works live |
| 05 | Write a Troubleshooting section for non-experts | covers 3 required topics, only `harness/README.md` changed |

Three "arms", each run non-interactively in a **throwaway clone** with permissions auto-approved ([`evals/coding/run.sh`](../../evals/coding/run.sh)):

- **ndcode**: OpenCode v2.2.5 → campus vLLM (Qwen3.5-35B-A3B-FP8, 128k, tool calling)
- **claude-local**: Claude Code → the *same* campus vLLM via its Anthropic-compatible `/v1/messages`
- **claude-cloud**: Claude Code → Anthropic's cloud (quality ceiling)

Tasks ran one at a time on one GB10 (single-user latency, not capacity). Tokens are from vLLM's `/metrics`. Cost is what Claude Code reported for the cloud arm.

## Results

### Baseline (one run per task)

| Arm | Passed | Wall time (5 tasks) | Tokens on our server (prompt / generated) | Cost |
|---|---|---|---|---|
| **ndcode** | **4/5** | 455 s | 972k / 34k | $0 |
| Claude Code → campus vLLM | 3/5 | 390 s | 1,042k / 17k | $0 |
| Claude Code → cloud Claude | **5/5** | 283 s | n/a | **$1.68** |

| Task | ndcode | Claude Code → vLLM | Claude Code → cloud |
|---|---|---|---|
| 01 explain | ✅ 38 s | ✅ 44 s | ✅ 25 s |
| 02 fix bug | ✅ 35 s | ✅ 21 s | ✅ 14 s |
| 03 feature | ❌ 186 s | ❌ 230 s | ✅ 85 s |
| 04 refactor | ✅ 145 s | ❌ 68 s | ✅ 54 s |
| 05 docs | ✅ 51 s | ✅ 27 s | ✅ 105 s |

### After adding campus working rules to ndcode

All three local failures had one root cause: **the agent declared success without running its change.**

- 03 (ndcode): still streamed the answer text before the JSON.
- 03 (Claude Code → vLLM): sent `stream_options` on a non-streaming request, so the server returned HTTP 400.
- 04 (Claude Code → vLLM): switched to `from harness import …`, which broke `python3 harness/chat.py`.

So `ndcode` now ships [`clients/AGENTS.md`](../../clients/AGENTS.md), loaded into every session via OpenCode's `instructions` setting: run the real entry point, run tests, compare output to the spec, keep entry points working, report what you verified. We confirmed the model sees the rules (it quotes them back verbatim).

| ndcode run | Passed | Wall time | Prompt / generated tokens |
|---|---|---|---|
| Baseline (no rules) | 4/5 | 455 s | 972k / 34k |
| Rules v1, rep 1 | 4/5 | 438 s | 859k / 31k |
| Rules v1, rep 2 | 4/5 | 390 s | 771k / 28k |
| Rules v2 (adds "exercise the feature end to end"), rep 1 | 4/5 | 352 s | 687k / 26k |
| Rules v2, extra runs of task 03 only (×2) | 0/2 | 120 s, 194 s | 336k / 9k, 437k / 14k |

**Across 4 full-suite runs, tasks 01, 02, 04 and 05 passed every time (16/16). Task 03 failed in all 7 attempts, always the same way** (the answer text streamed before the JSON object).

- The rules made runs leaner (fewer wasted turns). Rules v1 averaged **~9% less wall time and ~16% fewer prompt tokens** than the baseline. The one rules-v2 run was **~23% faster with ~29% fewer prompt tokens**. With this few runs, read that as a direction, not a precise number.
- They did **not** fix 03. With the rules, the agent ran unit tests, compile checks and `--help`, but never the feature itself (`--prompt … --json`), so it missed the stray streamed text. We read this as a **model capability limit** on "verify precisely against a spec", not a configuration problem.

## What this means

- **Good enough for daily use** on explain / debug-from-tests / refactor / docs work, at zero marginal cost with code staying on campus. Engineers should **review diffs and run the feature** before trusting "done" on new features with exact output requirements.
- **OpenCode beats Claude Code on the same local model** (4/5 vs 3/5). Claude Code's prompts are tuned for Claude models, which supports the "open-source agent on our platform" decision.
- **The gap to cloud is the hardest task class**, and it's what the cluster bake-off should probe: a larger or coding-specialist model (e.g. Qwen3-Coder-Next, ~80 GB FP8, needs the bigger cluster), and more tasks in that class.
- **Cost context:** the cloud arm cost ~$0.34 per task. At campus scale that adds up quickly, while the local arms cost only hardware and power.

## Caveats

- **Small sample:** 5 tasks, and 1–4 runs per arm. Treat these as directional, not statistically solid. The next milestone should grow the suite (more tasks per class, more repos).
- The tasks live in *this* repo, which the cloud model may handle more easily thanks to `CLAUDE.md`. All arms saw the same files.
- Single-user, sequential runs on a shared GB10. No concurrency or capacity measurement yet.
- One eval check had a bug (05 treated a `# comment` in a code block as a heading). It was fixed, and all runs were re-scored with [`evals/coding/rescore.sh`](../../evals/coding/rescore.sh) (one result flipped: 05 / Claude Code → vLLM, fail → pass).

## Raw data

Every CSV behind these tables is in [`data/`](data/) (columns and setup are described there).

## Reproduce

```sh
./serving/vllm/serve.sh                       # campus server (35B, tools, 128k)
./clients/install.sh                          # ndcode
evals/coding/run.sh                           # all tasks x all arms (cloud arm uses your Claude account)
evals/coding/run.sh --arms ndcode             # just ndcode
```
