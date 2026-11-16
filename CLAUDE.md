# CLAUDE.md

## Project
Notre Dame CRC is funding $50k of hardware for two competing on-prem LLM inference clusters, serving general-purpose and coding models to campus. Both get built and benchmarked head to head, and CRC scales up the winner. See README.md for the use cases (FERPA-safe classroom chatbots/RAG; coding agents).

## Environment
- Nodes: Dell Pro GB10 (Grace ARM64 CPU + Blackwell GPU, ~128 GB unified memory). Assume **aarch64**: check that wheels and containers have arm64/sbsa builds.
- Python ≥ 3.12 managed with **uv** (`uv run <script>`). Dependencies: `ollama`, `lm-eval[api]`.
- Current serving engine: Ollama, using `q4_K_M` quants for now (see `architecture/week2.md`).

## Layout
- `architecture/`: per-node evals. Copy `base_eval.py` to add an eval (a function taking `model` and returning a metrics dict, added to `EVALS`).
- `scripts/`: standalone, stdlib-only benchmark tools (e.g. `ollama_concurrency_probe.py`), documented in `scripts/README.md`.
- `serving/vllm/`: **separate uv project** for vLLM (`cd serving/vllm && uv sync`). Don't add vllm/torch to the root `pyproject.toml`. Launch with `serve.sh` (env-var config).
- `harness/`: stdlib-only clients for OpenAI-compatible servers (`chat.py`). Keep them engine-agnostic so the same client runs against vLLM, Ollama, or either cluster.
- `clients/`: the `ndcode` distribution (pinned OpenCode + `opencode.json` campus config + `ndcode` launcher). **We don't build our own coding agent.** ND owns the platform (model, hardware, serving), and agents are swappable open-source clients. Keep the lockdown (campus provider only, no share/auto-update/websearch, ask before edits/shell). Pin OpenCode to releases ≥ 2 weeks old with a SHA-256.
- `evals/coding/`: per-task `prompt.md` / optional `setup.sh` / `check.sh`, run by `run.sh` across agent "arms" in throwaway clones. Results summaries go in `docs/coding-eval/`.
- `examples/`: minimal Ollama / lm-eval usage.

## Conventions
- Git: architecture work branches off the `architecture` branch (see `architecture/week2.md`), and PRs merge into `main`.
- Benchmark both clusters with the same workloads, models, and quantization wherever the hardware allows. Document every deviation.
- Make results reproducible: record the model tag, quantization, engine and version, node, and concurrency alongside the numbers.
- GB10 nodes are **shared** and the memory is unified: always pass an explicit, modest `--gpu-memory-utilization` to vLLM (never its 0.9 default), check `nvidia-smi` / `free -g` first, and stop servers when done. Results measured alongside someone else's job aren't benchmarks.
- Don't commit model weights, caches, or `__pycache__` (see .gitignore).

## Skills (enabled for this repo via .claude/settings.json)
- **superpowers**: `brainstorming` before designing anything new (cluster architectures, eval harness), then `writing-plans` / `executing-plans`, TDD, and systematic debugging.
- **mattpocock-skills**: engineering workflows: `grill-me` (stress-test a design), `to-spec` / `to-tickets`, `tdd`, `code-review`, `domain-modeling`, `diagnosing-bugs`, `research`.
- **frontend-design**: for results dashboards and any public-facing pages.
