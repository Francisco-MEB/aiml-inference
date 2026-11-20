# What a benchmark looks like

Two parts: what you see in Claude Code while `/benchmark` runs, and what the results contain
when it's done. The numbers for **Qwen3 8B are real**: they come from the GB10 pilot on
2026-11-19 (levels 1 to 4, and a 5-question quality check). The 35B and 122B lines are
illustrative, to show the shape of a full study.

## 1. In Claude Code

```
> /benchmark qwen family

Preflight: PASS (GPU free, 115 GB available)
Plan for promaxgb10-2c9c (GB10, weights budget 94 GB): Qwen at nvfp4 (NVIDIA)
  qwen3-8b            small   dense  nvidia/Qwen3-8B-NVFP4            6.4 GB
  qwen3.6-35b-a3b     medium  moe    nvidia/Qwen3.6-35B-A3B-NVFP4    23.4 GB
  qwen3.5-122b-a10b   large   moe    nvidia/Qwen3.5-122B-A10B-NVFP4  83.5 GB
What's your name or netid?  > fborlong

Calibrating (levels 1, 2, 4), downloading ~113 GB first...
Calibration done (52 min, 0 errors). Estimate: about 3.5–6 h, running in stop mode; cap 6 h.
Started. Status every 30 min.

[30 min]  On track: qwen3.6-35b-a3b at 16 users, ~3 h left, no failures.
[1 h]     On track: qwen3.5-122b-a10b at 8 users, long_prefill stopped after 8 users (too slow); ~2 h left.
...
[done]    Finished in 4 h 40 min. Validation: PASS.
  Single-user speed: 8B 39 tok/s · 35B-A3B ~55 tok/s · 122B ~20 tok/s
  Users served at the interactive target (2 s first token, 10 tok/s each):
    short chat 32 / 16 / 4 · long document 8 / 4 / 1
  Quality: MMLU-Pro 53% / … · IFEval … · GSM8K … · HumanEval … · AIME …
  Spreadsheet: bench/results/qwen_nvfp4_promaxgb10-2c9c_20261120/combined/findings.xlsx
  Pushed to branch results/qwen_nvfp4_promaxgb10-2c9c_20261120.
```

## 2. The results

Everything lands in `bench/results/<study_id>/`. The pilot's folder (2.6 MB, 3 speed levels and
3 quality attempts for one model) looked like this:

```
bench/results/qwen_nvfp4_promaxgb10-2c9c_20261119/
├── combined/                       ← read this
│   ├── findings.xlsx               ← the spreadsheet (sheets below)
│   ├── report/                     ← every sheet as CSV: Models, Summary, Capacity, CapacityPivot,
│   │                                 Scaling, Quality, Timing, Parameters, Environment, Power, Requests
│   ├── requests.csv                ← every request from every level (105 rows here)
│   ├── power.csv  timing.csv  run.json  env.json  effective_config.json
├── time_estimate.json              ← the projection made after calibration
├── 20261119T223411Z_promaxgb10-2c9c_2eac4691/    ← one folder per model × level (here: 1 user)
│   ├── requests.csv                ← that level's requests
│   ├── env.json                    ← machine, versions, the exact server command, KV cache
│   ├── models.json                 ← weights checked against the pinned sha256 of every file
│   ├── run.json                    ← phases and timings, thinking check, per-model facts
│   ├── power.csv                   ← GPU power once a second
│   ├── preflight.json              ← who else was on the machine (nobody, or it wouldn't run)
│   ├── effective_config.json       ← the config as run
│   └── server_qwen3-8b.log         ← vLLM's own log
├── 20261119T224017Z_…/  20261119T224337Z_…/      ← 2 users, 4 users
└── 20261119T225914Z_…/                            ← the quality phase
    ├── lm_eval/qwen3-8b/…/results_*.json          ← lm-eval scores
    ├── lm_eval/qwen3-8b/…/samples_<task>_*.jsonl  ← every question, the model's answer, the grade
    └── server_quality_qwen3-8b.log  env.json  run.json  models.json  power.csv …
```

A full study has the same shape: one folder per model and level it ran (up to 8 levels × 3
models), one quality folder per model, and one `combined/`.

What's inside the main files (real excerpts):

`env.json`, the reproducibility record of one run:
```
host promaxgb10-2c9c · Ubuntu 24.04.5 LTS · NVIDIA GB10 · driver 580.178.04 · CUDA 13.0 · 121.6 GiB
protocol aiml-bench/2.0 · engine vllm 0.33.0 · quantization nvfp4 · slots 4 · git 6b5f5f98
server ready in 78 s · weights 5.98 GiB · KV cache 1,374,736 tokens (fp8_e4m3) · prefix caching on
command: vllm serve <pinned snapshot of nvidia/Qwen3-8B-NVFP4 @ ccd10a89> --max-num-seqs 4
  --max-model-len 4096 --gpu-memory-utilization 0.85
  --default-chat-template-kwargs {"enable_thinking":false} --reasoning-parser qwen3
```

`models.json`: `nvidia/Qwen3-8B-NVFP4 @ ccd10a89…, 2 weight files, 6.40 GB, verified: true`.

`requests.csv`, one row per request (32 columns; a few of them):
```
model_repo             prompt_id   users rep req ttft_s  e2e_s  prompt_tok out_tok decode_tps reasoning finish thinking
nvidia/Qwen3-8B-NVFP4  short_chat  4     0   0   0.039   6.31   38         256     40.6       0         length enable_thinking=false
```

`power.csv`: `t_unix, t_iso, gpu_power_w, …, gpu_util_pct, gpu_temp_c, gpu_clock_mhz, wall_meter_w`,
one row per second (wall meter blank unless you add one).

`timing.csv`: actual against projected minutes per model and level (1 user: 6.1 min, the first
server start compiles kernels; 2 and 4 users: 3.3 min each).

### Models sheet: one row per model

Real, Qwen3 8B on the GB10:

| Field | Value |
|---|---|
| Build | `nvidia/Qwen3-8B-NVFP4` @ `ccd10a89`, NVFP4, 6.24 bits per weight |
| Memory | 6.4 GB weights; KV cache room for 1,374,736 tokens |
| Server start | 78 s (warm caches) |
| Single-user speed | 39 tok/s (about 93% of the GB10's memory-bandwidth limit for this size) |
| Peak throughput (pilot stopped at 4 users) | 161 tok/s, short chat at 4 users |
| GPU energy at that peak | 0.17 J per generated token, 27 W GPU |
| User capacity, interactive target | ≥4 for every workload (still passing at the highest level tested) |

### Summary sheet: per model × workload × users

Real, Qwen3 8B. A full study continues upward (8, 16, 32 … 128 users) until `stop` mode ends
each workload.

| Workload | Users | First token p50 / p95 | Speed per user (median) | All users together | J / token |
|---|---|---|---|---|---|
| Short chat | 1 | 0.04 / 0.07 s | 39.2 tok/s | 39 tok/s | 0.69 |
| Short chat | 2 | 0.04 / 0.07 s | 38.0 tok/s | 76 tok/s | 0.36 |
| Short chat | 4 | 0.07 / 0.07 s | 40.6 tok/s | 161 tok/s | 0.17 |
| Long document (~2,900 tokens) | 1 | 0.29 / 0.31 s | 37.9 tok/s | 36 tok/s | 0.80 |
| Long document | 2 | 0.39 / 0.52 s | 35.2 tok/s | 67 tok/s | 0.47 |
| Long document | 4 | 0.56 / 1.03 s | 33.8 tok/s | 125 tok/s | 0.28 |
| Coding | 1 | 0.04 / 0.07 s | 39.1 tok/s | 39 tok/s | 0.70 |
| Coding | 2 | 0.06 / 0.10 s | 37.9 tok/s | 75 tok/s | 0.37 |
| Coding | 4 | 0.08 / 0.08 s | 40.4 tok/s | 160 tok/s | 0.18 |

How to read it: as users are added, speed per user should stay roughly flat while the total
grows. That's batching paying off. The first-token time climbs fastest on long documents,
because every user's document is read before answering, and that's usually what limits capacity.

### Capacity sheet: how many users per workload

For each target (interactive: first token within 2 s and at least 10 tok/s each; relaxed: 5 s
and 5 tok/s; strict: 1 s and 20 tok/s), the highest level that passed with every lower level
also passing, and what stopped it. Pilot: `≥4` everywhere, "not reached: passes at the highest
level tested" (it only went to 4 users). A full study gives a number like `16`, and the reason,
e.g. "p95 first token 2.4 s at 32 users".

### Quality sheet: score per model × task

Real, Qwen3 8B, but only **5 questions per task** in the pilot (a test run, so the numbers are
noisy; a full study uses 1,400 MMLU-Pro questions and the full sets below):

| Task | Metric | Pilot score (5 questions each) | Full study size |
|---|---|---|---|
| MMLU-Pro | exact match | 53% (70 questions: 5 per subject) | 1,400 |
| GPQA Diamond | exact match | skipped: no `HF_TOKEN` on the pilot machine | 198 |
| IFEval | prompt-level strict | 80% | 541 |
| GSM8K | exact match (flexible) | 80% | 1,319 |
| HumanEval | pass@1 | 100% (5/5) | 164 |
| AIME 2025 | exact match | 20% (1/5) | 30 |

### Other sheets

| Sheet | What |
|---|---|
| Parameters | protocol version, member, machine, quantization and who published it, the exact server command per model, checksums, any deviation |
| Scaling | throughput and speed per user against users, with charts |
| Timing | estimated against actual minutes per model and level |
| Environment, Requests, Power | run metadata, raw rows, GPU power once a second |
