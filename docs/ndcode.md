# ndcode: the campus coding agent

`ndcode` is Notre Dame's "campus Claude Code": an AI coding agent in your terminal that reads your code, edits files, and runs commands. It runs **entirely on campus hardware**: the model is an open-weight LLM served by our own vLLM server, and nothing you type is sent to a cloud AI provider.

Under the hood it's [OpenCode](https://opencode.ai) (open source, MIT), pre-configured and locked down for campus use. There's nothing to set up: install it once, then type `ndcode`.

| | |
|---|---|
| Model | Qwen3.5-35B-A3B (FP8), open-weight, served by vLLM on a Dell Pro GB10 |
| Agent | OpenCode v2.2.5 (pinned) |
| Context | 128k tokens per session |
| Cost to you | $0 per token |

---

## Install (once, no sudo)

On the GB10:

```sh
cd ~/aiml-inference          # your clone of this repo
./clients/install.sh
```

This downloads the pinned OpenCode release, checks its SHA-256, and installs `ndcode` into `~/.local/bin`. If that directory isn't on your `PATH`, the installer tells you the line to add to `~/.bashrc`. Re-run the installer any time to update.

## Use

The campus model server must be running (see [running-vllm.md](running-vllm.md)). Then, in any project:

```sh
cd ~/my-project
ndcode                                        # interactive session, like `claude`
ndcode run "add type hints to utils.py"      # one-shot, non-interactive
ndcode --continue                             # pick up your last session
```

It behaves like Claude Code. Ask in plain English, and the agent explores the code, proposes edits, runs tests, and iterates.

### Permissions: it asks first

Out of the box, `ndcode` **asks before it edits a file or runs a shell command**, just like Claude Code. You approve or reject each action.

- To loosen this for one project, add an `opencode.json` in that project (see the [OpenCode permission docs](https://opencode.ai/docs/permissions/)), e.g. `{"permission": {"edit": "allow"}}`.
- `ndcode run --auto "..."` auto-approves everything for that one run. Only use it in a throwaway copy or a clean git checkout you can reset.

### Project instructions

Like Claude Code, the agent reads project instructions from `AGENTS.md`, and also from `CLAUDE.md` for compatibility. If your repo already has a `CLAUDE.md`, `ndcode` picks it up.

On top of that, every session loads the **campus working rules** ([`clients/AGENTS.md`](../clients/AGENTS.md)): run the real command and the tests before reporting "done", keep existing entry points working, and summarize what was verified. In our [acceptance eval](coding-eval/README.md) the rules made sessions leaner, but the local model can still claim success on a new feature it hasn't run end to end. **Review the diff and run the feature yourself before trusting "done".**

## What's locked down, and why

The point of `ndcode` is that your code and conversations stay on campus. Compared to a stock OpenCode install:

| Setting | ndcode | Why |
|---|---|---|
| Model providers | **only `campus`** (all others denied by policy) | No accidental use of cloud models |
| OpenCode's free hosted models | hidden (`OPENCODE_DISABLE_FREE_MODELS`) | They run in the cloud |
| Session share links | **disabled** | Sharing uploads conversations to opencode.ai |
| Auto-update / update checks | **off** | We pin and test versions; update via `install.sh` |
| models.dev catalog download | **off** (`OPENCODE_DISABLE_MODELS_FETCH`) | Not needed; avoids calls home |
| Web search tool | **denied** | It uses a third-party search service |
| Web fetch tool | **asks first** | Fetching docs is useful, but you decide |
| Edits / shell commands | **ask first** | Same safety model as Claude Code |

Each session also runs its own private OpenCode server (`--standalone`), so it never attaches to a background OpenCode service that was started with different settings.

**Network check:** see "What still touches the internet" below for what we measured on a real session.

## Using it from your laptop (SSH tunnel)

The model server listens only on the GB10 itself (`127.0.0.1:8000`). From a laptop, forward the port, then point `ndcode` at it:

```sh
ssh -N -L 8000:localhost:8000 <you>@<gb10-host>     # leave running
NDCODE_BASE_URL=http://127.0.0.1:8000/v1 ndcode      # (laptop install comes in a later milestone)
```

## Troubleshooting

| Message / symptom | Fix |
|---|---|
| `the campus model server isn't reachable` | Start it: `./serving/vllm/serve.sh` (see [running-vllm.md](running-vllm.md)) |
| `isn't serving a model named 'chat'` | The server was started with a different `NAME=`. Restart it with defaults |
| `Error: "auto" tool choice requires --enable-auto-tool-choice and --tool-call-parser to be set` | The server was started without tool calling (e.g. the old 9B command). Restart it with the default `serve.sh` (`TOOL_PARSER=qwen3_coder`) |
| `ndcode: command not found` | Add `~/.local/bin` to `PATH` (the installer prints the line) |
| Answers stop mid-way on huge tasks | 128k-token context limit. Start a fresh `ndcode` session or break the task up |

## How it's built (for maintainers)

| File | Role |
|---|---|
| [`clients/install.sh`](../clients/install.sh) | Pinned OpenCode version + SHA-256, per-user install |
| [`clients/opencode.json`](../clients/opencode.json) | Campus config: provider, model limits, permissions, lockdown, loads the working rules |
| [`clients/AGENTS.md`](../clients/AGENTS.md) | Campus working rules loaded into every session (verify before "done", keep entry points working) |
| [`clients/ndcode`](../clients/ndcode) | Launcher: sets env switches, health-checks the server, runs OpenCode `--standalone` |

To upgrade OpenCode: pick a release **at least two weeks old**, update `OPENCODE_VERSION` and the SHA-256 in `install.sh`, check the config still validates (`OPENCODE_CONFIG=clients/opencode.json opencode debug config`), and re-run the coding eval (`evals/coding/run.sh --arms ndcode`).
