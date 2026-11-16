# aiml-inference

The goal of this project is to provide an ***evaluation*** and ***analysis*** of potential architectures for the university to use when they scale up compute. 

## Context

Notre Dame's Center for Research Computing (CRC) is funding **$50,000 in hardware** to design, build, and benchmark **two competing AI inference clusters**. We build both, benchmark them head to head, and CRC scales up whichever one wins. The question: how much inference capacity can we actually deliver for campus at that price point, running local LLMs (general-purpose and coding) instead of relying on expensive cloud services?

## Use Cases

1. Faculty/Staff: Professors want to be able to create tools for their classroom which might require access to sensitive information. Localized LLMs provide a way for them to avoid violating copyright and FERPA laws. **PRIMARY USE: Chatbot like architectures, tool calling, RAG (large context windows needed)**

2. Coding Agents: Tokens are EXPENSIVE. We want to give engineers on campus a way to utilize the power of coding agents while not incurring the variable cost from the tokens. **PRIMARY USE: development environments, coding jobs, harness level tasks (the model must interact with its environment)**

## Repository layout

| Path | Contents |
|---|---|
| [`architecture/`](architecture/) | Architecture group work: per-node model evals on the Dell Pro GB10s. Start with [`week2.md`](architecture/week2.md) and [`base_eval.py`](architecture/base_eval.py). |
| [`docs/`](docs/) | Runbooks. **Start here:** [`running-vllm.md`](docs/running-vllm.md) (serve the campus model) and [`ndcode.md`](docs/ndcode.md) (the campus coding agent). |
| [`serving/`](serving/) | Model servers. [`serving/vllm/`](serving/vllm/) runs vLLM in its own uv env (`./serving/vllm/serve.sh`): Qwen3.5-35B-A3B, tool calling, 128k context. |
| [`clients/`](clients/) | **`ndcode`**, the zero-config "campus Claude Code" (a locked-down OpenCode distribution). Install with `./clients/install.sh`. |
| [`harness/`](harness/) | Lightweight clients for the served models. [`chat.py`](harness/chat.py) is a plain-language terminal chat that works with vLLM or Ollama. |
| [`evals/`](evals/) | Automated evaluations. [`evals/coding/`](evals/coding/) runs real coding tasks through `ndcode`, Claude Code on our model, and cloud Claude. |
| [`examples/`](examples/) | Minimal "hello world" usage of local inference servers (e.g. Ollama from Python). |
| [`scripts/`](scripts/) | Benchmarking and measurement tools. See [`scripts/README.md`](scripts/README.md) for full docs, including the [Ollama concurrency probe](scripts/README.md#ollama_concurrency_probepy). |

## Claude Code skills

This repo ships a shared set of [Claude Code](https://claude.com/claude-code) plugins, configured in [`.claude/settings.json`](.claude/settings.json):

| Plugin | What it's for | Try |
|---|---|---|
| **superpowers** | Brainstorm → plan → execute workflow, TDD, systematic debugging | `/superpowers:brainstorming` |
| **mattpocock-skills** | Grill a design, turn it into specs/tickets, code review, domain modeling | `/mattpocock-skills:grill-me` |
| **frontend-design** | Polished, non-generic UIs (e.g. results dashboards) | `/frontend-design:frontend-design` |

**Setup (one time):** run `claude` from the repo root and trust the folder. Claude Code then asks to install the plugins: accept. If it doesn't, run:

```sh
claude plugin install superpowers@claude-plugins-official
claude plugin install mattpocock-skills@claude-plugins-official
claude plugin install frontend-design@claude-plugins-official
```

**Use:** type `/` in Claude Code to browse the skills, or describe the task. Claude also picks up the relevant skill on its own (e.g. "let's design cluster B" triggers brainstorming).
