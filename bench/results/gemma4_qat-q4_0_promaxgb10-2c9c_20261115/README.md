# Gemma 4 benchmark results (Dell GB10 `promaxgb10-2c9c`)

One folder per run: `<UTC start>_<host>_<config hash>`. Each holds `env.json` (every setting,
versions and checksums), `effective_config.json`, `preflight.json`, `requests.csv` (one row per
request), `power.csv` (1 Hz GPU telemetry), `run.json`, and `findings.xlsx` / `report/`.
Quality runs add `lm_eval/` with per-question samples.

Raw Ollama server logs (`server_*.log`) are not committed (55 MB); their effective settings
are copied into each `env.json` under `server_effective_config_logged`.

The meeting numbers come from `combined_gemma4_gb10/`, rebuilt with
`python3 bench/gemma4/harness/merge_session.py`.

| Run | Models | Users | Slots | What it is |
|---|---|---|---|---|
| `20261115T192639Z_promaxgb10-2c9c_3a4fcb0d` | e4b | 1 2 | 2 | Smoke test 1 (superseded: long prompt was too long; HumanEval stock task scored 0). |
| `20261115T193201Z_promaxgb10-2c9c_dc131d33` | e4b | 1 2 | 2 | Smoke test 2 (fixed prompt, chat HumanEval). |
| `20261115T193428Z_promaxgb10-2c9c_dc131d33` | e4b | 1 2 | 2 | Smoke test 3, clean tree. |
| `20261115T195641Z_promaxgb10-2c9c_8ad16415` | e4b | 1 2 | 2 | Smoke test 4 for the speed/capacity report. |
| `20261115T195725Z_promaxgb10-2c9c_8ad16415` | e4b 12b 31b 26b-a4b | - | 16 | Main sweep, server fixed at 16 slots. E4B and 12B complete; 31B interrupted after 31 requests (session ended). Used as the 'slots=16' comparison. |
| `20261115T213600Z_promaxgb10-2c9c_8ad16415` | e4b | 1 | 1 | Speed sweep, slots matched to users (one level per run). |
| `20261115T213834Z_promaxgb10-2c9c_8ad16415` | e4b | 2 | 2 | Speed sweep, slots matched to users (one level per run). |
| `20261115T214023Z_promaxgb10-2c9c_8ad16415` | e4b | 4 | 4 | Speed sweep, slots matched to users (one level per run). |
| `20261115T214226Z_promaxgb10-2c9c_8ad16415` | e4b | 8 | 8 | Speed sweep, slots matched to users (one level per run). |
| `20261115T214459Z_promaxgb10-2c9c_8ad16415` | e4b | 16 | 16 | Speed sweep, slots matched to users (one level per run). |
| `20261115T214922Z_promaxgb10-2c9c_8ad16415` | 31b | 1 | 1 | Speed sweep, slots matched to users (one level per run). |
| `20261115T215715Z_promaxgb10-2c9c_8ad16415` | 31b | 2 | 2 | Speed sweep, slots matched to users (one level per run). |
| `20261115T220553Z_promaxgb10-2c9c_8ad16415` | 31b | 4 | 4 | Speed sweep, slots matched to users (one level per run). |
| `20261115T221553Z_promaxgb10-2c9c_8ad16415` | 31b | 8 | 8 | Speed sweep, slots matched to users (one level per run). |
| `20261115T223009Z_promaxgb10-2c9c_8ad16415` | 31b | 16 | 16 | Speed sweep, slots matched to users (one level per run). |
| `20261115T230210Z_promaxgb10-2c9c_8ad16415` | - | - | - | 26B A4B sweep stopped by a false pre-flight alarm (root nvidia-smi -q); alarm fixed in beef58d. |
| `20261115T230228Z_promaxgb10-2c9c_8ad16415` | e4b | - | - | Quality: mmlu_pro, limit 10. |
| `20261115T231104Z_promaxgb10-2c9c_8ad16415` | e4b | - | - | Quality: humaneval_chat, limit 40. |
| `20261115T231223Z_promaxgb10-2c9c_8ad16415` | 12b | - | - | Quality: mmlu_pro, limit 10. |
| `20261115T232823Z_promaxgb10-2c9c_8ad16415` | 12b | - | - | Quality: humaneval_chat, limit 40. |
| `20261115T233016Z_promaxgb10-2c9c_8ad16415` | 26b-a4b | - | - | Quality: mmlu_pro, limit 10. |
| `20261115T234212Z_promaxgb10-2c9c_8ad16415` | 26b-a4b | - | - | Quality: humaneval_chat, limit 40. |
| `20261115T234343Z_promaxgb10-2c9c_8ad16415` | 31b | - | - | Quality: mmlu_pro, limit 10. |
| `20261116T001438Z_promaxgb10-2c9c_8ad16415` | - | - | - | Quality run stopped by pre-flight: another user's vLLM on the GPU. |
| `20261116T001455Z_promaxgb10-2c9c_8ad16415` | - | - | - | Quality run stopped by pre-flight: another user's vLLM on the GPU. |
| `20261116T001456Z_promaxgb10-2c9c_8ad16415` | - | - | - | Quality run stopped by pre-flight: another user's vLLM on the GPU. |
| `20261116T153426Z_promaxgb10-2c9c_8ad16415` | 26b-a4b | 1 | 1 | 26B A4B, 1 user, first attempt (repeated in 20261116T154215Z after a script restart). |
| `20261116T154215Z_promaxgb10-2c9c_8ad16415` | 26b-a4b | 1 | 1 | Speed sweep, slots matched to users (one level per run). |
| `20261116T154358Z_promaxgb10-2c9c_8ad16415` | 26b-a4b | 2 | 2 | Speed sweep, slots matched to users (one level per run). |
| `20261116T154603Z_promaxgb10-2c9c_8ad16415` | 26b-a4b | 4 | 4 | Speed sweep, slots matched to users (one level per run). |
| `20261116T154845Z_promaxgb10-2c9c_8ad16415` | 26b-a4b | 8 | 8 | Speed sweep, slots matched to users (one level per run). |
| `20261116T155239Z_promaxgb10-2c9c_8ad16415` | 26b-a4b | 16 | 16 | Speed sweep, slots matched to users (one level per run). |
| `20261116T155943Z_promaxgb10-2c9c_8ad16415` | 12b | 1 | 1 | Speed sweep, slots matched to users (one level per run). |
| `20261116T160302Z_promaxgb10-2c9c_8ad16415` | 12b | 2 | 2 | Speed sweep, slots matched to users (one level per run). |
| `20261116T160640Z_promaxgb10-2c9c_8ad16415` | 12b | 4 | 4 | Speed sweep, slots matched to users (one level per run). |
| `20261116T161055Z_promaxgb10-2c9c_8ad16415` | 12b | 8 | 8 | Speed sweep, slots matched to users (one level per run). |
| `20261116T161622Z_promaxgb10-2c9c_8ad16415` | 12b | 16 | 16 | Speed sweep, slots matched to users (one level per run). |
| `20261116T162725Z_promaxgb10-2c9c_8ad16415` | 31b | - | - | Quality: humaneval_chat, limit 40. |
| `combined_gemma4_gb10` | all |  |  | Merged dataset and findings.xlsx used for the meeting dashboard. |
