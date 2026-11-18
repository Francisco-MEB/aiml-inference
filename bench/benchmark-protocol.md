# Club benchmark protocol `aiml-bench/1.0`

This is the standard every member follows when benchmarking a model for the inference-cluster
study, so that results from different people, models and machines line up. If you use Claude
Code, the `benchmark` skill (`.claude/skills/benchmark/SKILL.md`) walks you through it.

Results only compare within the same protocol version. Changing anything in "Fixed by the
protocol" means a new version, agreed by the benchmarks team.

## 1. What is fixed and what you choose

| Fixed by the protocol | Your choice (recorded in every result) |
|---|---|
| Prompts, answer length, sampling settings | Models |
| Simultaneous-user levels and repetitions | **Quantization** (e.g. `q4_K_M`, `Q4_0` QAT, `q8_0`) |
| Server settings per level, context per user | Backend: Ollama in Docker (club default), native Ollama, or another OpenAI-compatible server |
| Metrics, targets, capacity rule | High-level mode: `stop`, `all` or `cap:<min>` (section 5) |
| Column names and file layout | Machine; total time cap |

The quantization is yours to pick, but one study uses **one** quantization for all its models,
and it appears in the config, in every CSV row, in the study folder name, in each model's
name in the spreadsheet (`qwen3.5-35b-a3b (q4_K_M)`) and on the Parameters sheet. Compare
studies with different quantizations only with that in mind.

## 2. Workloads (prompts)

File `bench/prompts/prompts_v1.jsonl`, sha256
`5c7772860192c2ae65e360358407585aca613810ddfa05687ff67b5b7a8e199d`. The harness refuses to run
if the file differs. Every model sees identical text.

| Workload id | Prompt | Input size (Gemma tokenizer) | Stands for |
|---|---|---|---|
| `short_chat` | "Explain in plain language what a hash table is and why lookups are fast." | ~36 tokens | a quick chatbot question |
| `long_prefill` | A synthetic course catalog (48 records) followed by: summarize which departments appear most, flag over-enrolled courses, describe scheduling patterns | ~2,865 tokens | RAG / chat with a document |
| `coding` | Write `merge_intervals(intervals)` with type hints, docstring, empty-list handling, three asserts, and explain its complexity | ~82 tokens | coding assistant |

Token counts differ between tokenizers. The text is what is fixed; the measured
`prompt_tokens` is recorded per request.

## 3. Requests

- OpenAI-compatible `/v1/chat/completions`, streaming, with `stream_options.include_usage`.
- `max_tokens` **256**, `temperature` **0**, `seed` **42**.
- Each request's user message starts with a deterministic 8-character tag (`[req 1a2b3c4d]`), so
  the server cannot reuse an earlier request's prompt cache and skip prefill.
- **Thinking off.** Each model in the config says how (`thinking_off`) and sends it
  (`extra_body`). For Ollama: `reasoning_effort: "none"`. A model that cannot turn thinking off
  (e.g. gpt-oss) uses its lowest setting and says so. Every model gets a probe at load time that
  records whether reasoning text comes back; the validator flags any.
- Context: **4,096 tokens per user** (`OLLAMA_CONTEXT_LENGTH=4096`).

## 4. Levels, server slots and repetitions

- Simultaneous users: **1, 2, 4, 8, 16, 32, 64, 128**.
- **Server slots = users.** At each level the server is restarted with exactly that many
  parallel slots (`OLLAMA_NUM_PARALLEL=N`). The Gemma study showed that a server fixed at 16
  slots made 2 to 8 users about half as fast, so fixed slots are not allowed.
- A batch is N identical requests released at the same instant. **5 batches** per level up to 16
  users, **3** from 32 users (already 96+ requests per level).
- Per model, per level: load, one discarded warm-up request, one discarded batch filling every
  slot, the thinking probe, the batches, unload.
- Levels 1, 2 and 4 are the **calibration** (section 6). They are real data and are kept.

## 5. High levels (32 to 128 users): you choose, after seeing the time

Large models and long prompts get slow fast. The 31B Gemma took 262 s per batch for long context
at 16 users. So at the start, after calibration, you choose one mode:

| Mode | What runs | Use when |
|---|---|---|
| `stop` | Per model and workload, levels go up until one **fails the relaxed target** (p95 first token > 5 s, median per-user speed < 5 tok/s, or any error) or the model **no longer fits fully on the GPU**. Higher levels are skipped: they cannot change any capacity figure. | Default. Gives every capacity number for the least time. |
| `all` | Every level to 128 for every workload. | You want complete curves and have the hours. Big models may fail to load at high levels; that is recorded. |
| `cap:<min>` | Every level, but each workload level stops starting new batches after `<min>` minutes and is marked as capped. | Complete curves with a bounded time. |

You also set a total cap (`--cap-hours`): no new level starts after that.

## 6. How long it will take on your machine

Time depends on the machine and the model, so the protocol measures it instead of guessing:

1. **Calibrate:** run levels 1, 2 and 4 (`run_protocol.sh ... --calibrate`).
2. **Estimate:** `estimate_time.py` fits each model and workload's batch time, p95 first-token
   time and per-user speed through the two highest measured levels, adds the measured per-level
   overhead (server restart, load, warm-ups), and projects every mode, including where `stop`
   would end.
3. **Read it as a range:** projection ×1 to ×2. On the Gemma GB10 study, projecting from levels
   1, 2 and 4, actual batch time was 0.73× to 2.05× of the projection (median 1.05×), and the
   worst cases were long context at 16 users. Projections far beyond the calibrated levels
   (64, 128) are the least reliable.
4. **While running**, the estimate is redone after every level from the newest data and shown
   by `status.sh`. The final spreadsheet's Timing sheet compares projected and actual minutes
   per level, so each machine builds its own timing record.

Before any calibration, `estimate_time.py --from-reference` scales the GB10 reference below by
active parameters. It is rough: use it only to decide whether to calibrate today.

**Reference: Dell Pro Max GB10 (`promaxgb10-2c9c`), Gemma 4 QAT Q4_0, protocol settings**

Measured minutes per level, all three workloads, including restart and load:

| Model | 1 user | 2 | 4 | 8 | 16 | Total 1 to 16 |
|---|---|---|---|---|---|---|
| E4B | 1.6 | 1.8 | 2.0 | 2.5 | 3.8 | 12 min |
| 26B A4B (MoE) | 1.7 | 2.1 | 2.7 | 3.9 | 7.0 | 17 min |
| 12B | 3.3 | 3.6 | 4.2 | 5.5 | 11.0 | 28 min |
| 31B | 7.9 | 8.6 | 10.0 | 14.3 | 32.0 | 73 min |

Projected time **after** calibration (levels 8 to 128), from the same data:

| Model | `stop` | `all` |
|---|---|---|
| E4B | 6 to 12 min (long context stops after 8 users) | 26 to 52 min |
| 26B A4B | 7 to 14 min | 57 min to 1.9 h |
| 12B | 11 to 23 min | 1.0 to 2.0 h |
| 31B | 15 to 31 min (long context stops after 2 users) | 2.3 to 4.6 h |

Long context dominates: its batch time roughly doubles with each doubling of users.

## 7. Metrics and targets

All timing is measured by the client, so it means the same on any backend.

- **Time to first token (TTFT):** request sent to the first answer text. Includes queueing.
- **Per-user decode speed:** (output tokens − 1) / (last token time − first token time).
- **Prefill speed (estimate):** prompt tokens / TTFT. Meaningful only at 1 user.
- **Aggregate throughput:** all output tokens of a batch / batch wall time.
- **GPU power:** `nvidia-smi power.draw` at 1 Hz. GPU only; record a wall meter when you can
  (`make_report.py --wall-csv`, columns `t_unix,watts`).

Targets (capacity is reported for all three):

| Scenario | p95 TTFT at most | Per-user speed at least |
|---|---|---|
| interactive (headline) | 2 s | 10 tok/s |
| relaxed (used by `stop`) | 5 s | 5 tok/s |
| strict | 1 s | 20 tok/s |

**User capacity** = the highest tested level that meets the target with **every lower level
also meeting it** and no errors, using the median per-user speed (mean, 5th percentile and
minimum are also reported). "≥N" means the highest level tested still passed.

## 8. Columns and files

### `requests.csv`: one row per request

| Column | Meaning |
|---|---|
| `protocol_version` | `aiml-bench/1.0` |
| `study_id` | study folder name |
| `machine` | hostname of the machine measured |
| `backend` | `ollama/native`, `ollama/docker` or `openai/external` |
| `server_version` | server version it reported |
| `quantization` | the study's quantization |
| `slots` | parallel slots the server had (= users) |
| `run_id` | run folder name |
| `model_key`, `model_tag` | your key (`<family>-<size>[-a<active>]`) and the exact server tag |
| `prompt_id`, `prompt_class` | workload id and its descriptive class |
| `concurrency` | simultaneous users in this batch |
| `rep`, `req` | batch number within the level, request number within the batch |
| `batch_start_unix`, `batch_end_unix`, `batch_wall_s` | batch timing (Unix seconds, same clock as `power.csv`) |
| `batch_agg_tps` | aggregate output tokens/s of the batch |
| `req_start_unix`, `ttft_s`, `e2e_s` | request start, time to first token, end-to-end time (s) |
| `prompt_tokens`, `output_tokens` | token counts the server reported |
| `decode_tps`, `prefill_tps_est` | per-request decode speed and prefill estimate (tok/s) |
| `reasoning_chars`, `content_chars` | length of reasoning text (should be 0) and of the answer |
| `finish_reason`, `error` | `length` for a full 256-token answer; error text if the request failed |

### `power.csv`
`t_unix, t_iso, gpu_power_w, gpu_power_instant_w, gpu_power_avg_w, gpu_util_pct, gpu_temp_c,
gpu_clock_mhz, wall_meter_w`

### `findings.xlsx` (in `combined/`)
| Sheet | Contents |
|---|---|
| Parameters | protocol version, study, member, machine, **quantization**, every server and request setting, checksums, deviations |
| Models | one row per model: size, memory, single-user speed, peak throughput, capacity |
| Summary | per model × workload × users: TTFT mean/p50/p95/max, decode mean/median/p05/min, aggregate, scaling efficiency, power, J/token |
| Capacity | capacity per scenario × decode statistic, and which limit was hit |
| Scaling | throughput and per-user speed against users, with charts |
| Timing | projected against actual minutes per model and level |
| ServerTimingCheck, Environment, Requests, Power | cross-check against server timings, run metadata, raw rows |

### Folders
- Config: `bench/configs/<study_id>.json`, copied from `bench/configs/TEMPLATE.json`.
- `study_id` = `<family>_<quant>_<machine>_<YYYYMMDD>`, e.g. `qwen3.5_q4_K_M_promaxgb10-2c9c_20261120`.
- Results: `bench/results/<study_id>/<run_id>/` (one folder per model and level) and
  `bench/results/<study_id>/combined/` (merged data and `findings.xlsx`).

## 9. Running it without your session

Everything is started **detached**, so it keeps running if VS Code closes, SSH drops or your
laptop sleeps:

```sh
bench/run_protocol.sh --config bench/configs/<study>.json --calibrate
.venv/bin/python bench/harness/estimate_time.py --config bench/configs/<study>.json
bench/run_protocol.sh --config bench/configs/<study>.json --mode stop --cap-hours 6
bench/status.sh <study_id>              # progress and remaining time
tail -f bench/.run/<study_id>.log       # live log
bench/stop_protocol.sh <study_id>       # clean stop; completed levels are kept
```

- **Resume:** run the same `run_protocol.sh` command again. Completed levels are skipped; an
  interrupted level is redone.
- **Free GPU:** before the first level it waits until nobody else has used the GPU for 10
  minutes (`--free-min`), gives up after 4 hours (`--max-wait-min`), and before every later
  level re-checks and pauses if someone else started.
- **One study at a time per machine.** The launcher refuses a second one.
- **Check once per machine** that logging out doesn't kill your processes:
  `loginctl show-user $USER -p Linger` and `grep KillUserProcesses /etc/systemd/logind.conf`.
  If `KillUserProcesses=yes` and `Linger=no`, ask an admin for `loginctl enable-linger <you>`,
  or keep a session open (e.g. inside `tmux`). On `promaxgb10-2c9c` it is safe.

## 10. Shared-machine rules

- Run `python3 bench/harness/preflight.py` first; the runner repeats it before every level.
- Tell the club channel before a run longer than an hour, with the estimate from section 6.
- Never stop or change someone else's server, container or model.
- Use your own model folder and port (default `~/ollama-bench-models`, port 11435); never the
  system Ollama service.

## 11. A valid result

Share a study only when `python3 bench/harness/validate_results.py <study_id>` passes. It checks:
- protocol version, prompt checksum and request settings;
- server slots equal users at every level;
- every completed level has N × repetitions requests;
- no errors or missing token counts;
- one quantization per study;
- all columns present.

Test runs made with `--test-overrides` are marked as deviating and are never merged into
results unless asked.
