# Campus LLM stack: overview

The one-page map of what we've built, why, how it fits together, and what's still open. Step-by-step instructions live in the linked runbooks. **Keep this file current:** when you change a decision, a pinned version, or the debt register, update it in the same PR.

*State as of 2026-11-19, branch `dev`.* A formatted version of this memo is published as a private Claude page (ask Francisco for access).

---

## 1. Bottom line

- One Dell Pro GB10 serves **Qwen3.5-35B-A3B (FP8)** through **vLLM 0.33**. That gives an OpenAI- and Anthropic-compatible API with tool calling, a 128k-token context, ~55 tok/s for one user, and room for ~8 simultaneous full-length sessions.
- Two clients sit on top:
  - [`harness/chat.py`](../harness/chat.py): a plain-language terminal chat.
  - **[`ndcode`](ndcode.md)**: a zero-config "campus Claude Code" (a pinned, locked-down OpenCode).
- On a 5-task [acceptance eval](coding-eval/README.md), `ndcode` passed every everyday task (16/16 runs) at $0 marginal cost, and failed "new feature with an exact output spec" in all 7 attempts. Cloud Claude passed 5/5 for $1.68.
- **Capacity (2026-11-17, one GB10):**
  - Chat: 160 tok/s total at 8 users × 20 tok/s each, rising to 676 tok/s at 128 users × 5 tok/s.
  - Long-document RAG: ~8 users per GB10 before the first token takes >5 s (prompt reading is capped at ~5k tok/s).
  - Python is not the bottleneck (async-scheduling A/B: <1% change).
  - At the stated peak assumptions, 15k people need roughly 19–38 GB10s at ~11 tok/s per user ([capacity](capacity/README.md)).
- Strategy: **ND owns the platform** (model choice, hardware, serving, later a gateway). Coding agents are swappable open-source clients. We don't build our own agent.

## 2. Goal

CRC funds **$50k of hardware** for **two competing inference clusters**. Both get built and benchmarked head to head, and CRC scales up the winner. The use cases are FERPA-safe classroom chatbots/RAG (long context) and coding agents (tool calling, long context). See the [README](../README.md).

## 3. Decisions and why

| Decision | Choice | Why | Revisit when |
|---|---|---|---|
| Serving engine | vLLM 0.33 | Continuous batching, prefix caching, OpenAI + Anthropic APIs, tool-call parsers, aarch64/CUDA 13 wheels | SGLang/TensorRT-LLM clearly win on cluster hardware |
| vLLM install | Own uv project `serving/vllm/` | No sudo/Docker; keeps ~9 GB torch stack out of the team env | Moving to a managed service |
| Language | Python orchestrates; the math runs in C++/CUDA (FlashInfer, CUTLASS, NCCL, CUDA graphs). No C++ rewrite | Measured 2026-11-17: GPU 96% busy from 1 to 128 users, the engine's Python loop at 6–26% of a core while generating, and `--async-scheduling` changed throughput <1%. Campus scale is a GPU-count question ([capacity](capacity/README.md#the-question-is-python-a-bottleneck-at-15000-users)) | The capacity benchmark shows the GPU idling while the engine's main thread is pinned, even after `--async-scheduling` → TensorRT-LLM arm |
| Python for vLLM | uv-managed CPython 3.12 | System Python lacks `Python.h` (Triton needs it); 3.15 lacks wheels | Deps ship 3.13+ wheels |
| Model | Qwen3.5-35B-A3B-FP8 @ `9d1823d2` | MoE: ~3B active/token, fast on the GB10's ~273 GB/s; strong tool use; one model for chat + coding | Bigger cluster can host Qwen3-Coder-Next / 100B+ |
| Memory budget | `GPU_UTIL=0.50` (~60 GB) | Shared node, unified memory | Node becomes dedicated |
| Coding agent | Adopt open source, don't build | Building is months of work and adds no independence (open source can be forked) | No suitable agent for a workflow |
| Agent | OpenCode v2.2.5 | MIT, most-used OSS agent, Claude-Code-like UX, standalone arm64 binary | Quarterly / eval shows a better one |
| Claude Code | Comparison baseline only | Proprietary, Claude-tuned; scored 3/5 vs OpenCode 4/5 on our model | n/a |
| UX | Zero config: install once, type `ndcode` | "As if using Claude Code, but the campus version" | n/a |
| Access | localhost + SSH tunnel, no API key | No exposure until a gateway exists | Gateway milestone |
| Permissions | Ask before edits and shell | Same safety model as Claude Code | Per project |
| Lockdown | Campus provider only; no share, update checks, web search | Code and conversations stay on campus | CRC policy |
| Git | Work on `dev`, PRs to `main` | Team preference | n/a |
| Club benchmark | `aiml-bench/2.0` (`bench/`): vLLM on Dells, MLX on Macs, one study = family × machine × one quantization, small → large; thinking off; slots = users; core quality via lm-eval once per build | Same numbers from every member; Ollama isn't what campus would serve; quality belongs to the build, not the machine | Thinking-on quality, ShareGPT traffic, LiveCodeBench / HLE / MRCR / ALCE, two-node runs |
| Benchmark models | `bench/catalog/models.json`: every build pinned to a commit, per engine, highest precision first unless a family names a `prefer` build; members can't improvise tags. Single-node lineup (2026-11-20): small / medium / large per family, MoE for the bigger classes where the family has one (Qwen3.5 9B · 3.6 35B-A3B · 3.5 122B-A10B; Gemma 4 12B · 26B-A4B · 31B; GLM 4.6V Flash · 4.7 Flash · 4.5 Air; Moonlight · Kimi-Linear · Kimi-Dev). Gemma prefers NVFP4 on Dells and 4-bit on Macs, so every study is 4-bit | Reproducible and comparable small → large; MoE is what campus would deploy; 4-bit everywhere keeps Dell vs Mac like-for-like and is what a multi-user server runs | New families or builds (catalog `pin`) |

## 4. Architecture

```
 Clients      ndcode → OpenCode v2.2.5 · harness/chat.py · evals/coding/run.sh · (Claude Code, baseline)
                 │  HTTP 127.0.0.1:8000
 API          /v1/chat/completions · /v1/messages (Anthropic) · /v1/responses · /v1/models · /metrics
 Engine       vLLM 0.33.0 · qwen3 reasoning parser · qwen3_coder tool parser · prefix caching · torch.compile + FlashInfer
 Model        Qwen3.5-35B-A3B-FP8 @9d1823d2, served as "chat", 128k context, text only
 Runtime      PyTorch 2.14 + CUDA 13.0 · CPython 3.12 (uv) · tmux session
 Hardware     Dell Pro GB10 · Grace ARM64 + Blackwell · ~121 GB unified memory · ~273 GB/s
```

**An `ndcode` turn:**
1. OpenCode sends the conversation, the tool definitions and the campus rules to `/v1/chat/completions`.
2. vLLM's `qwen3_coder` parser turns the model's output into structured `tool_calls`.
3. OpenCode asks the user before an edit or shell command, runs the tool, and sends the result back.
4. Steps 1–3 repeat until the model answers without calling a tool.

## 5. Tech stack and pins

| Component | Version | Set in | Pinned |
|---|---|---|---|
| OS / arch | Ubuntu 24.04.5, aarch64 | machine | machine |
| NVIDIA driver / CUDA | 580.178.04 / 13.0 | machine | machine |
| uv | 0.13.6 | machine | machine |
| Python (vLLM env) | CPython 3.12.16 (uv-managed) | `serving/vllm/.python-version`, `python-preference = "only-managed"` | yes |
| vLLM | 0.33.0 | `serving/vllm/pyproject.toml` + `uv.lock` | yes |
| PyTorch / Transformers / FlashInfer | 2.14.0+cu130 / 5.20.2 / 0.7.2.post1 | `serving/vllm/uv.lock` | yes |
| Model | Qwen/Qwen3.5-35B-A3B-FP8 @ `9d1823d2dee688a6b25e77009dc727688c44936e` | `serve.sh` `DEFAULT_REVISION` | yes |
| OpenCode | v2.2.5 (30 Oct 2026), SHA-256 `5a528f42caf576c6cdece6a191eb273d3db54effa5518e0402684d8f8026aeb6` | `clients/install.sh` | yes |
| Root env | `lm-eval[api]>=0.4.13` (0.4.13 locked), `openpyxl==3.1.5`, `ollama>=0.6.3` | `pyproject.toml` + `uv.lock` | lockfile |
| NVFP4 quantization (`quantize` group) | llmcompressor 0.15.0 (torch 2.14.1+cu130, transformers 5.21.0, compressed-tensors 0.20.1) | `pyproject.toml` + `uv.lock` | yes |
| MLX (Macs) | mlx 0.32.4, mlx-lm 0.32.2, CPython 3.12 | `serving/mlx/pyproject.toml` + `uv.lock` | yes |
| Benchmark protocol / catalog | `aiml-bench/2.0`; 91 builds pinned (latest 2026-11-20) | `bench/harness/protocol.py`, `bench/catalog/models.json` | yes |
| huggingface_hub CLI (downloads) | 2.4.0 | `bench/harness/pull_models.py` `HF_HUB_VERSION` | yes |
| Claude Code plugins | superpowers 7.0.2, mattpocock-skills 1.3.1, frontend-design 1.1.1 | `.claude/settings.json` | marketplace |

Caches:
- `models/dell/`, `models/mac/studio/`, `models/mac/mini/` (repo, gitignored): club benchmark weights per machine, Hugging Face cache layout; the harness picks this machine's folder, `bench/download_models.py` fills any of them; `HF_HUB_CACHE` overrides
- `~/.cache/huggingface/hub`: model weights for `serve.sh`
- `~/.cache/vllm`: compiled graphs
- `~/.cache/flashinfer`: JIT kernels
- `~/.cache/uv`: packages
- `~/.local/share/uv/python`: managed Pythons

## 6. Where to find things

| Need | Go to |
|---|---|
| Start / stop / check the server, tmux, troubleshooting | [`running-vllm.md`](running-vllm.md) |
| Every `serve.sh` setting | [`serving/vllm/README.md`](../serving/vllm/README.md) |
| Install and use `ndcode`, what's locked down, upgrading OpenCode | [`ndcode.md`](ndcode.md) |
| Chat client options | [`harness/README.md`](../harness/README.md) |
| Eval method, results, raw data | [`coding-eval/README.md`](coding-eval/README.md), [`coding-eval/data/`](coding-eval/data/) |
| Capacity benchmark: method, campus sizing, Python-vs-GPU bottleneck | [`capacity/README.md`](capacity/README.md), [`scripts/README.md`](../scripts/README.md) |
| Club benchmark (every family, Dells and Macs): protocol, catalog, running a study | [`bench/README.md`](../bench/README.md), [`bench/benchmark-protocol.md`](../bench/benchmark-protocol.md), [`serving/mlx/README.md`](../serving/mlx/README.md) |
| Download a study's models ahead of time | `uv run bench/download_models.py [family ...] [--machine dell\|studio\|mini]` → `models/dell/`, `models/mac/studio/`, `models/mac/mini/` ([bench/README.md](../bench/README.md)) |
| Run the eval | `evals/coding/run.sh` (see the header of that script) |
| Conventions for Claude sessions | [`CLAUDE.md`](../CLAUDE.md) |

## 7. Change procedures

| Change | Steps |
|---|---|
| Try another model | `MODEL=… TEXT_ONLY=0\|1 TOOL_PARSER=… ./serving/vllm/serve.sh` (use a tool parser vLLM supports for that family). Adjust `clients/opencode.json` limits if the context differs. |
| Change the default model | Update `DEFAULT_MODEL` + `DEFAULT_REVISION` in `serve.sh`, the model name/limits in `clients/opencode.json`, and this file + the runbooks. Run `evals/coding/run.sh`, save the CSVs in `docs/coding-eval/data/`, update the results doc. |
| Upgrade vLLM | Bump `vllm==` in `serving/vllm/pyproject.toml` (release ≥ 2 weeks old) → `uv lock && uv sync` → start the server → tool-call `curl` test ([runbook](running-vllm.md)) → eval. |
| Upgrade OpenCode | See [`ndcode.md`](ndcode.md#how-its-built-for-maintainers). Validate the config with `opencode debug config` (zero "normalization diagnostic" warnings) and migrate deprecated keys. |
| Fix an eval check | Edit `check.sh`, then `evals/coding/rescore.sh <output dir>` re-scores old runs without re-running agents. |
| Add a model to the club benchmark | Add it to its family in `bench/catalog/models.json` (a build per engine and format) → `bench/harness/catalog.py pin` → `catalog.py plan <family> --machine dell` / `mac:<GiB>` → commit. |
| Change the benchmark protocol | Edit `bench/benchmark-protocol.md` and `bench/harness/protocol.py` together, bump `VERSION`, run `bench/tests`. Old results stay, marked as another version. |

## 8. Measurements (2026-11-16, single node, shared)

| Metric | Qwen3.5-9B (BF16) | Qwen3.5-35B-A3B (FP8) |
|---|---|---|
| Memory budget | 0.30 (~36 GB) | 0.50 (~60 GB) |
| Weights | 16.7 GiB loaded | ~36 GB download |
| Context | 32,768 | 131,072 |
| KV cache | 385k tokens (11.8 × 32k) | 1,097 blocks (~8.4 × 128k) |
| Single-user speed | 13.2–13.4 tok/s | 54.9 tok/s (53 in the capacity run) |
| Time to first token | 0.12–0.19 s | 0.13 s |
| Tool calling | off | on, verified |

**Capacity** (2026-11-17, idle node, full tables in [`capacity/`](capacity/README.md#results)):

| Workload | 1 user | 8 users | 32 users | 128 users |
|---|---|---|---|---|
| Chat: total tok/s (per user) | 53 (53) | 160 (20) | 354 (11) | 676 (5.3) |
| RAG ~3.4k-token prompt: total tok/s, first-token p95 | 48, 0.7 s | 129, 5.5 s | 208, 25 s | 229, 154 s |

During the coding eval, vLLM served ~90% of prompt tokens from its prefix cache (4.79M of 5.33M). Single-user speed is bounded by memory bandwidth ÷ bytes read per token. That's why the MoE, which reads ~3B params per token, is ~4× faster than the dense 9B.

## 9. Problems hit and fixes

| Problem | Cause | Fix |
|---|---|---|
| `Python.h: No such file or directory` at vLLM startup | Triton JIT needs headers; system Python has none | uv-managed CPython |
| transformers `NVFP4Config` can't load its kernel on the GB10 | Hub kernel `kernels-community/nvfp4-gemm` is built for sm_100a/sm_120a only (arch-specific, won't run on sm_121) and is no longer public | `scripts/nvfp4_quantization.py` uses llm-compressor; vLLM runs the FP4 kernels |
| `Python.h` missing when llm-compressor saves NVFP4 | compressed-tensors packs FP4 with Triton; root env uses system Python | Script sets `CT_ENFORCE_EAGER=1` (PyTorch packing) |
| `uv sync`: no wheel for `nvidia-cudnn-frontend` | uv picked Python 3.15 | Pinned 3.12 |
| `"auto" tool choice requires --enable-auto-tool-choice` | Server started without tool calling | 35B default with `TOOL_PARSER=qwen3_coder` |
| OpenCode ignored `attachment`/`reasoning`/`temperature` | Unsupported legacy keys in v2.2.5 | Removed |
| Settings could leak between OpenCode sessions | v2 background service keeps its start-up env | `ndcode` always runs `--standalone` |
| Eval 05 false fail | `# comment` in a code block parsed as a heading | Fixed check, `rescore.sh`, re-scored |
| $0.29 "cost" on a local run | Claude Code prices at Anthropic rates | Cost recorded for cloud arm only |
| Couldn't push | No credentials on the machine | SSH key added to GitHub; `origin` uses SSH |

## 10. Technical debt register

**Paid down 2026-11-19:** the club benchmark moved from Ollama to vLLM/MLX (`aiml-bench/2.0`) with a pinned model catalog and a unit/end-to-end test suite for `bench/` (`bench/tests`, fake server, no GPU). Pilot on this GB10 the same day (`nvidia/Qwen3-8B-NVFP4`, test run, not merged): levels 1 to 4 clean, 39 tok/s for one user (~93% of the bandwidth limit), 0.07 s to first token, ~9k tok/s prefill, 1.03 s p95 first token for the long document at 4 users, thinking off verified, vLLM restart ~75 s with warm caches; quality phase end to end on 5 questions per task. It found and fixed: HF 2.x cache verification, single-model plans using the family's build, `--quality-only`, the missing `lm-eval[ifeval]` extra, lm-eval failures reported as success, and stock `humaneval_instruct` truncating chat answers (now the club's `humaneval_chat`).

**Paid down 2026-11-17:** the vLLM capacity benchmark (tools, chat + RAG runs, Python A/B).

**Paid down 2026-11-16:** pinned the default model revision; CLAUDE.md refreshed; runbook has the 35B KV numbers; removed a tracked `.pyc`; OpenCode config cleaned to supported keys; raw eval data saved.

| Item | Risk | Severity | Next action |
|---|---|---|---|
| No auth, quotas, usage logs | Anyone with a shell on the GB10 can use the server | medium | Gateway (per-user keys, quotas, logs) before any network exposure |
| Server runs in a personal tmux | Dies with the session/reboot; no restart or alerting | medium | systemd unit or container, restart policy, log rotation, health check |
| Small, in-repo eval suite | 5 tasks, few runs; checks partly keyword-based | medium | More tasks per class, other repos |
| Exact-spec feature weakness | Agent can claim success on wrong output | medium | Documented for users; try coding-specialist / larger models |
| OpenCode config uses deprecated keys | Next upgrade may break the lockdown | medium | Migrate to `providers`/`permissions`/`policies` when upgrading |
| Project `opencode.json` can override campus settings | A repo could loosen permissions | low | Later: `OPENCODE_DISABLE_PROJECT_CONFIG` or managed policies |
| `ndcode models`/`session` not standalone | Use the shared background service | low | Revisit if issues appear |
| Installer is linux-arm64 only | No laptop installs | low | Pin hashes for other platforms (laptop milestone) |
| No CI; unit tests only for `bench/` | Regressions elsewhere caught only by hand/eval | medium | CI workflow running `bench/tests` + shellcheck |
| Benchmark Mac (MLX) paths never run on a Mac | Preflight, power, launcher or vision-language MLX conversions may fail there | medium | Mac owners run the smoke test in `serving/mlx/README.md` before their first study |
| Benchmark builds not all exercised on GB10 | NVFP4 / W4A16 / FP8 kernels on sm_121 had bugs in older vLLM; a build may fail to start or produce garbage | medium | Each first study per family: check server logs, thinking probe and GSM8K against published scores |
| Benchmark 2.0 deferred items | Thinking-off quality understates reasoning models; no ShareGPT traffic, LiveCodeBench, HLE, MRCR, ALCE, agentic coding, Mac power | low | Phase 2 of the benchmark after the 2.0 results |
| GLM benchmark builds are community W4A16 (cyankiwi) | Quantization quality not vendor-checked | low | Swap for a vendor 4-bit build when one fits a GB10 |
| SSH key without passphrase on a shared box | Anyone with the Unix account can push | medium | `ssh-keygen -p -f ~/.ssh/id_ed25519` or a repo deploy key |
| Stale branch `origin/worktree-init-project` | Clutter | low | Delete on GitHub |
| Leftover local state | Unused uv Python 3.15; 9B weights (~19 GB) | low | `uv python uninstall 3.15`; keep 9B as fallback |
| Shared GB10, no scheduling | Memory collisions between teammates | medium | Booking rule until there's a scheduler |

## 11. Accounts and machine

- **GitHub:** `origin` = `git@github.com:Francisco-MEB/aiml-inference.git` over SSH. Key `~/.ssh/id_ed25519` (comment `promaxgb10`) authenticates as `Francisco-MEB`. There's no global git identity on the GB10, so commit with `git -c user.name=… -c user.email=…` or set it in `~/.gitconfig`.
- **Branches:**
  - `dev`: serving, ndcode and eval work
  - `main`: PR target
  - `architecture`: the architecture group
  - `test/ollama-concurrency`
- **Shared node:**
  - `gdasilv2` runs Ollama + lm-eval benchmarks (~34 GB).
  - On 2026-11-17 the `aiml_club` account had a two-node vLLM running (stopped before our capacity runs that day): `nvidia/Qwen3.8-27B-NVFP4`, `--tensor-parallel-size 2 --nnodes 2`, master `10.100.64.2`, `--gpu-memory-utilization 0.8`. It ran in a Docker container and held ~100 of 121 GB on this Dell.
  - Check `nvidia-smi` / `free -g` / `ps` before starting vLLM.
- **Two-node link:** the two GB10s are cabled over ConnectX-7 (2 × 200 Gb/s, RoCE). This node is `10.100.64.2` / `10.100.65.2`, the peer is `.1`. NVIDIA Sync (laptop app) manages the pair. vLLM's native `--nnodes/--node-rank/--master-addr` already works across it (the club server above). Two-node serving for our stack is parked until the single-node capacity numbers exist.

## 12. Next milestones

1. **Two-Dell capacity:** two copies (one per GB10) vs one model split across both, same probe. Single-GB10 results are done ([`capacity/`](capacity/README.md)). Needs our vLLM env, the weights and SSH on the peer.
2. **Managed service** (systemd, restarts, logs, health checks).
3. **Campus gateway** (auth, keys, quotas, logs). Then network exposure and laptop installs of `ndcode`.
4. **Model bake-off:** the club benchmark (`aiml-bench/2.0`) runs every family small → large on each Dell and Mac (assignments: Dell 1 Will Carter, Dell 2 Gabriel, Mac Studio Brandon, Mac Minis Jeremy and Tyler, single node each); then the coding eval on the finalists.
5. **Grow the eval** and add CI.

## 13. Glossary

| Term | Meaning |
|---|---|
| MoE | Mixture of experts: a router activates a few expert blocks per token. "35B-A3B" = 35B total, ~3B active |
| FP8 / BF16 | 1 vs 2 bytes per weight |
| KV cache | Per-session memory of past tokens; sets how many long conversations fit |
| Prefix caching | vLLM reuses work for prompt prefixes it has already seen |
| TTFT | Time to first token |
| Tool calling | Model emits a structured request to run a function; the client runs it |
| Thinking | Qwen3.5's optional reasoning before answering. `chat.py` turns it off; `ndcode` leaves it on |
| Unified memory | GB10's CPU and GPU share one ~121 GB pool |
| Standalone (OpenCode) | Private OpenCode server per session instead of the shared background service |
| Arm | One agent + backend combination in the eval |
