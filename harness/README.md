# harness

Front ends for talking to the models we serve. Each one works with any OpenAI-compatible server (vLLM, Ollama, SGLang, llama.cpp), so the same client can be pointed at either cluster.

| Script | What it does |
|---|---|
| [`chat.py`](chat.py) | Terminal chat in plain language, for non-experts. Streams answers, remembers the conversation, prints a speed stats line. |

Next: a coding/agent harness (tool calling, reading and editing files) built on the same request code.

## `chat.py`

Standard library only. Run it with plain `python3`, no install needed.

```sh
# Start a server first, e.g. vLLM:  ./serving/vllm/serve.sh
python3 harness/chat.py                                        # interactive chat with vLLM on :8000
python3 harness/chat.py --prompt "Explain RAM like I'm 10"      # one question, then exit

# Same client, different engine (Ollama's OpenAI-compatible endpoint):
python3 harness/chat.py --base-url http://127.0.0.1:11434/v1 --model qwen3.5:9b
```

In the chat: `/reset` clears the conversation, and `/quit` or Ctrl-D exits. Ctrl-C while an answer is streaming stops that answer.

Each answer ends with a stats line:

```
[TTFT 0.21s | 143 tokens | 38.5 tok/s | total 3.9s]
```

- **TTFT**: time to first token (how long before text starts appearing).
- **tokens**: output length, as reported by the server. `~` means estimated from stream chunks.
- **tok/s**: generation speed after the first token.

Treat these as a sanity check. Proper load testing (many concurrent users) lives in `scripts/`.

### Options

| Flag | Default | Notes |
|---|---|---|
| `--base-url` | `http://127.0.0.1:8000/v1` | Server API root |
| `--model` | `chat` | Must match the name the server exposes (`serve.sh` uses `chat`) |
| `--system` | plain-language assistant | Replace the system prompt |
| `--think` | off | Let reasoning models (Qwen3/3.5) think first: better on hard questions, slower |
| `--show-thinking` | off | Print the model's reasoning, dimmed |
| `--temperature` `--top-p` `--top-k` `--presence-penalty` | 0.7 / 0.8 / 20 / 1.5 | Qwen3.5's recommended non-thinking settings |
| `--max-tokens` | 2048 | Max answer length |
| `--no-stats` | off | Hide the stats line |
