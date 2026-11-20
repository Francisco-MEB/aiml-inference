# Club benchmark protocol `aiml-bench/2.0`

This is the standard every member follows when benchmarking a model for the inference-cluster
study, so that results from different people, models and machines line up. With Claude Code,
say what you were assigned ("benchmark the Qwen family") and the `benchmark` skill
(`.claude/skills/benchmark/SKILL.md`) runs this protocol for you.

Results only compare within the same protocol version. Changing anything in "Fixed by the
protocol" means a new version, agreed by the benchmarks team. 2.0 replaced Ollama with vLLM
(NVIDIA) and MLX (Apple Silicon); `aiml-bench/1.0` and the pre-1.0 Gemma study are kept in
`bench/results/` for reference and don't compare with 2.0 numbers.

## 1. What is fixed, what the catalog decides, what you choose

| Fixed by the protocol | Decided by the catalog and the machine | Your choice |
|---|---|---|
| Prompts, answer length, sampling, thinking off | The exact models (Hugging Face repo + commit) | Which family or model (your assignment) |
| User levels, repetitions, slots = users | The study's quantization | How far the high levels go: `stop`, `all` or `cap:<min>` (section 5) |
| Server settings per level, context per user | Engine: **vLLM** on NVIDIA, **MLX** on Apple Silicon | Total time cap |
| Quality tasks and their sizes | Which models fit this machine | |
| Metrics, targets, capacity rule, columns, file layout | | |

**The catalog** (`bench/catalog/models.json`) lists every family's models, small to large, and
the published builds of each per engine, pinned to a commit. Only the benchmarks lead edits it
(`catalog.py pin` fills the pins from the Hub). A model that isn't in the catalog isn't
benchmarked until it is added.

**A study** is one family × one machine × one quantization: every model in it shares the same
build format, so small → large is a like-for-like comparison. `catalog.py plan <family>` picks:

1. the **largest** model of the family that fits this machine in any build;
2. the **highest-precision** build in which that model fits (order: BF16, then 8-bit / FP8,
   then 4-bit);
3. every family model that has that build and fits. A model without a build in that format,
   or too large, is listed as **left out** with the reason, never swapped for another build.

"Fits" means the weights take at most 85% of the engine's memory budget, which leaves 15% for
the KV cache. The budget is `0.85 × total memory` with vLLM (`--gpu-memory-utilization 0.85`)
and `0.75 × RAM` on a Mac (macOS lets the GPU use about three quarters of RAM).

### Quantization formats

The study's quantization appears in the config, every CSV row, the study folder name and the
spreadsheet. Formats in 2.0:

| Name | Engine | What it is | Bits per weight (quantized layers) |
|---|---|---|---|
| `bf16` | both | unquantized 16-bit brain float | 16 |
| `fp8` | vLLM | 8-bit float weights (E4M3), usually with per-channel or block scales | 8 |
| `8bit` | MLX | MLX affine 8-bit, group size 64 | 8 (+ scales) |
| `nvfp4` | vLLM | NVIDIA 4-bit float (E2M1) with an FP8 scale per 16 weights (ModelOpt or compressed-tensors) | 4.5 |
| `mxfp4` | both | OCP 4-bit float with a shared scale per 32 weights; gpt-oss is trained in it | 4.25 |
| `w4a16` | vLLM | 4-bit integer weights, 16-bit activations (GPTQ/AWQ-style, compressed-tensors) | 4 (+ scales) |
| `4bit` | MLX | MLX affine 4-bit, group size 64 | 4.5 |

Each row also records `bits_per_weight`: the weights' size in bits divided by the parameter
count. It is always higher than the nominal bits, because embeddings, norms and scales stay at
higher precision. Different formats with the same nominal bits (`nvfp4` on a Dell, `4bit` on a
Mac) are close but not identical; compare them across machines with that in mind. Community
builds (not from the model's vendor, NVIDIA, Red Hat AI or mlx-community) are marked as a
deviation in the spreadsheet.

How to read a build's format yourself: `config.json` → `quantization_config` (`quant_method`:
`modelopt`, `compressed-tensors`, `mxfp4`, `fp8`, ...) for vLLM builds; `quantization`
(`bits`, `group_size`, `mode`) for MLX builds. No such block means unquantized.

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
`prompt_tokens` is recorded per request. The speed phase measures how fast answers come, not
how good they are; quality is section 7.

## 3. Requests

- OpenAI-compatible `/v1/chat/completions`, streaming, with `stream_options.include_usage`.
- `max_tokens` **256**, `temperature` **0** (greedy). **No `seed`**: greedy decoding needs
  none, and `mlx_lm.server` serves a request that carries a seed on its own, which would
  turn off batching on the Macs.
- Each request's user message starts with a deterministic 8-character tag (`[req 1a2b3c4d]`), so
  the server cannot reuse an earlier request's prompt cache and skip prefill. Prefix caching
  stays at the engine default (on), as a campus server would run it.
- **Thinking off**, set by the catalog per family: the server's chat-template default
  `enable_thinking=false` (vLLM `--default-chat-template-kwargs`, MLX `--chat-template-args`).
  gpt-oss cannot turn reasoning off; it runs at `reasoning_effort=low`, its lowest setting, and
  says so in the `thinking` column. Every model gets a probe at load time that records whether
  reasoning text comes back; the validator fails a study whose models return reasoning text,
  except the ones that cannot turn it off.
- Context: **4,096 tokens per sequence** (vLLM `--max-model-len 4096`).

## 4. Levels, server slots and repetitions

- Simultaneous users: **1, 2, 4, 8, 16, 32, 64, 128**.
- **Server slots = users.** At each level the server is restarted with exactly that many
  parallel sequences: vLLM `--max-num-seqs N`, MLX `--decode-concurrency N --prompt-concurrency N`.
  The Gemma 1.0 study showed that a server fixed at 16 slots made 2 to 8 users about half as
  fast on Ollama; 2.0 keeps the rule so the levels stay comparable.
- vLLM runs at a fixed `--gpu-memory-utilization 0.85` for every model, from the pinned local
  snapshot (`HF_HUB_OFFLINE=1`). Each run records the KV-cache size vLLM got
  (`kv_cache_tokens`), its dtype and any preemptions. A server that cannot start (the model or
  its KV cache does not fit) is recorded as not fitting at that level.
- A batch is N identical requests released at the same instant. **5 batches** per level up to 16
  users, **3** from 32 users (already 96+ requests per level).
- Per model, per level: start the server, one discarded warm-up request (vLLM compiles kernels
  on it), one discarded batch filling every slot, the thinking probe, the batches, stop the
  server.
- Levels 1, 2 and 4 are the **calibration** (section 6). They are real data and are kept.

## 5. High levels (32 to 128 users): `stop` by default

Large models and long prompts get slow fast. The 31B Gemma took 262 s per batch for long context
at 16 users on Ollama. So after calibration the study runs in one of three modes. The skill uses
`stop` with a time cap set from the estimate, and asks you to choose only when the estimate is
over 4 hours or you want complete curves:

| Mode | What runs | Use when |
|---|---|---|
| `stop` | Per model and workload, levels go up until one **fails the relaxed target** (p95 first token > 5 s, median per-user speed < 5 tok/s, or any error) or the server **no longer starts** at that many slots. Higher levels are skipped: they cannot change any capacity figure. | Default. Gives every capacity number for the least time. |
| `all` | Every level to 128 for every workload. | You want complete curves and have the hours. |
| `cap:<min>` | Every level, but each workload level stops starting new batches after `<min>` minutes and is marked as capped. | Complete curves with a bounded time. |

A total cap (`--cap-hours`, by default the high end of the estimate) bounds the run: no new
level starts after it.

## 6. How long it will take on your machine

Time depends on the machine and the model, so the protocol measures it instead of guessing:

1. **Calibrate:** run levels 1, 2 and 4 (`run_protocol.sh ... --calibrate`).
2. **Estimate:** `estimate_time.py` fits each model and workload's batch time, p95 first-token
   time and per-user speed through the two highest measured levels, adds the measured per-level
   overhead (server restart, load, warm-ups), and projects every mode, including where `stop`
   would end.
3. **Read it as a range:** projection ×1 to ×2. On the Gemma GB10 study, projecting from levels
   1, 2 and 4, actual batch time was 0.73× to 2.05× of the projection (median 1.05×).
   Projections far beyond the calibrated levels (64, 128) are the least reliable. A vLLM
   restart costs more than an Ollama one (about 1 to 2 minutes once its compile caches are
   warm), and the measured overhead includes it.
4. **While running**, the estimate is redone after every level from the newest data and shown
   by `status.sh`. The final spreadsheet's Timing sheet compares projected and actual minutes.
5. **Add the quality phase:** with thinking off, about 0.5 to 1 hour per model on a GB10
   (section 7).

Before any calibration, `estimate_time.py --from-reference` scales the GB10 Gemma 1.0
reference below by gigabytes read per token. It is rough: use it only to decide whether to
calibrate today.

**Reference: Dell Pro Max GB10 (`promaxgb10-2c9c`), Gemma 4 QAT Q4_0 on Ollama (1.0 settings)**

| Model | 1 user | 2 | 4 | 8 | 16 | Total 1 to 16 |
|---|---|---|---|---|---|---|
| E4B | 1.6 | 1.8 | 2.0 | 2.5 | 3.8 | 12 min |
| 26B A4B (MoE) | 1.7 | 2.1 | 2.7 | 3.9 | 7.0 | 17 min |
| 12B | 3.3 | 3.6 | 4.2 | 5.5 | 11.0 | 28 min |
| 31B | 7.9 | 8.6 | 10.0 | 14.3 | 32.0 | 73 min |

Long context dominates: its batch time roughly doubles with each doubling of users.

## 7. Quality (core suite, thinking off)

After a model's speed levels, its quality phase runs once, on a server restarted with 32 slots
and a 16,384-token context, through lm-evaluation-harness 0.4.13 (`--model
bench-chat-completions`: the stock chat client without the fixed seed, see section 3).
Greedy, thinking off as in section 3, at most 4,096 generated tokens per answer.

| Task (lm-eval name) | What it measures | Size | Headline metric |
|---|---|---|---|
| `mmlu_pro` | knowledge and reasoning, 14 subjects, 10 choices | 100 per subject = 1,400 | `exact_match,custom-extract` |
| `gpqa_diamond_cot_zeroshot` | graduate-level science (gated: needs `HF_TOKEN` and accepted terms) | 198 | `exact_match,flexible-extract` |
| `ifeval` | following checkable instructions | 541 | `prompt_level_strict_acc` |
| `gsm8k_cot_zeroshot` | grade-school math; a harness sanity check against published scores | 1,319 | `exact_match,flexible-extract` |
| `humaneval_chat` | Python functions graded by hidden tests (runs model-written code) | 164 | `pass@1` |
| `aime25` | competition math, exact integer answers | 30 | `exact_match` |

`humaneval_chat` is the club's own task (`bench/lm_eval_tasks/`): the model answers in a code
block, which is extracted and tested. The stock `humaneval_instruct` cuts chat answers off at its
`\ndef` stop string and scored Qwen3 8B 0/5 in the pilot. GSM8K's `strict-match` expects the
answer in a fixed "The answer is" format that chat models rarely use, so it reads near 0; report
`flexible-extract`.

A quality score belongs to the **build**, not the machine: the same repo at the same commit
scores the same on any machine. So the phase is skipped, and the skip recorded, when this study
or another study in `bench/results/` already measured that exact repo@revision. Machines whose
build differs (a Mac's `4bit` against a Dell's `nvfp4`) measure their own.

Scores with thinking off understate what reasoning models reach with thinking on; that is a
deliberate 2.0 choice (fast, comparable, one setting for every model). Thinking-on quality is
planned for a later version.

## 8. Metrics and targets

All timing is measured by the client, so it means the same on any engine.

- **Time to first token (TTFT):** request sent to the first answer text. Includes queueing.
- **Per-user decode speed:** (output tokens − 1) / (last token time − first token time).
- **Prefill speed (estimate):** prompt tokens / TTFT. Meaningful only at 1 user.
- **Aggregate throughput:** all output tokens of a batch / batch wall time.
- **GPU power:** `nvidia-smi power.draw` at 1 Hz on NVIDIA. GPU only; record a wall meter when
  you can (`make_report.py --wall-csv`, columns `t_unix,watts`). Not recorded on Macs in 2.0.

Targets (capacity is reported for all three):

| Scenario | p95 TTFT at most | Per-user speed at least |
|---|---|---|
| interactive (headline) | 2 s | 10 tok/s |
| relaxed (used by `stop`) | 5 s | 5 tok/s |
| strict | 1 s | 20 tok/s |

**User capacity** = the highest tested level that meets the target with **every lower level
also meeting it** and no errors, using the median per-user speed (mean, 5th percentile and
minimum are also reported). "≥N" means the highest level tested still passed.

## 9. Columns and files

### `requests.csv`: one row per request

| Column | Meaning |
|---|---|
| `protocol_version` | `aiml-bench/2.0` |
| `study_id` | study folder name |
| `machine` | hostname of the machine measured |
| `backend`, `engine`, `engine_version` | `vllm/native` or `mlx/native`; engine name; version it reported |
| `server_version` | same as `engine_version` (kept for 1.0 compatibility) |
| `quantization` | the study's build format (section 1) |
| `slots` | parallel sequences the server had (= users) |
| `model_repo`, `model_revision` | the Hugging Face repo and commit served |
| `weight_format`, `bits_per_weight` | format the Hub reports; stored bits per parameter |
| `kv_cache_dtype`, `kv_cache_tokens` | vLLM's KV cache (`n/a (mlx)` on Macs) |
| `thinking` | how thinking was turned off |
| `run_id` | run folder name |
| `model_key`, `model_tag` | catalog key (`<family>-<size>[-a<active>]`) and the repo |
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

### Per run
`env.json` (machine, OS, driver, git commit, every server's exact command, version and KV cache
under `servers`), `models.json` (weights verified against the pinned sha256 of every file),
`run.json` (phases, per-model facts, quality results), `power.csv`, `server_*.log`, and
`lm_eval/<model>/` for the quality phase.

### `findings.xlsx` (in `combined/`)
| Sheet | Contents |
|---|---|
| Parameters | protocol version, study, member, machine, **quantization** and publisher, every server command and setting, checksums, deviations |
| Models | one row per model: repo@revision, bits per weight, memory, KV cache, single-user speed, peak throughput, capacity |
| Summary | per model × workload × users: TTFT mean/p50/p95/max, decode mean/median/p05/min, aggregate, scaling efficiency, power, J/token |
| Capacity | capacity per scenario × decode statistic, and which limit was hit |
| Scaling | throughput and per-user speed against users, with charts |
| Quality | lm-eval score per model × task, with stderr and sample count |
| Timing | projected against actual minutes per model and level |
| Environment, Requests, Power | run metadata, raw rows |

### Folders
- Config: `bench/configs/<study_id>.json`, written by `catalog.py plan <family> --write`.
- `study_id` = `<family>_<quant>_<machine>_<YYYYMMDD>`, e.g. `qwen_nvfp4_promaxgb10-2c9c_20261120`.
- Results: `bench/results/<study_id>/<run_id>/` (one folder per model and level, and per
  quality phase) and `bench/results/<study_id>/combined/` (merged data and `findings.xlsx`).

## 10. Running it without your session

Everything is started **detached**, so it keeps running if VS Code closes, SSH drops or your
laptop sleeps:

```sh
bench/run_protocol.sh --config bench/configs/<study>.json --calibrate
.venv/bin/python bench/harness/estimate_time.py --config bench/configs/<study>.json
bench/run_protocol.sh --config bench/configs/<study>.json --mode stop --cap-hours 12
bench/status.sh <study_id>              # progress and remaining time
tail -f bench/.run/<study_id>.log       # live log
bench/stop_protocol.sh <study_id>       # clean stop; completed levels are kept
```

The second `run_protocol.sh` runs every remaining level and then each model's quality phase.

- **Resume:** run the same `run_protocol.sh` command again. Completed levels and quality phases
  are skipped; an interrupted one is redone.
- **Free machine:** before the first level it waits until nobody else has used the GPU for 10
  minutes (`--free-min`), gives up after 4 hours (`--max-wait-min`), and before every later
  level re-checks and pauses if someone else started.
- **One study at a time per machine.** The launcher refuses a second one.
- **Check once per machine** that logging out doesn't kill your processes (Linux):
  `loginctl show-user $USER -p Linger` and `grep KillUserProcesses /etc/systemd/logind.conf`.
  If `KillUserProcesses=yes` and `Linger=no`, ask an admin for `loginctl enable-linger <you>`,
  or keep a session open (e.g. inside `tmux`). On a Mac, `run_protocol.sh` keeps the machine
  awake for the length of the run (`caffeinate -w`); turning on System Settings → Energy →
  "Prevent automatic sleeping when the display is off" once covers the time between runs too.

## 11. Shared-machine rules

- Run `.venv/bin/python bench/harness/preflight.py` first; the runner repeats it before every
  level. It stops on another person's GPU process or inference server (vLLM, MLX, Ollama with
  a model loaded, LM Studio, ...), on sustained CPU use by someone else, and on vLLM machines
  when less than 85% of memory is free.
- **A benchmark needs the machine to itself.** Book it in the club channel before a run longer
  than an hour, with the estimate from section 6.
- Never stop or change someone else's server, container or model.
- The benchmark server listens on `127.0.0.1:8100` (`backend.port`) and is always stopped by
  the harness, including after `stop_protocol.sh`.

## 12. A valid result

Share a study only when `.venv/bin/python bench/harness/validate_results.py <study_id>` passes.
It checks:
- protocol version, prompt checksum and request settings;
- server slots equal users at every level;
- every completed level has N × repetitions requests;
- no errors or missing token counts;
- weights verified against the pinned checksums, revision pinned in every row;
- thinking off (no reasoning text, except models that cannot turn it off);
- one quantization per study;
- all columns present; a quality result per model (or a note where the same build's was measured).

Test runs (`--test-overrides ... --allow-deviation`) are marked as deviating and are never
merged into results or counted as a model's quality.
