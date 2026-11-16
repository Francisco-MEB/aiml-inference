#!/usr/bin/env python3
"""
Terminal chat client for any OpenAI-compatible LLM server (vLLM, Ollama, ...).

Standard library only: copy it anywhere and run it with plain `python3`.

    python3 harness/chat.py                                   # vLLM on :8000 (serving/vllm/serve.sh)
    python3 harness/chat.py --prompt "What is an LLM?"        # one question, then exit
    python3 harness/chat.py --base-url http://127.0.0.1:11434/v1 --model qwen3.5:9b   # Ollama

In the chat, type /reset to clear the conversation, /quit (or Ctrl-D) to leave.
Ctrl-C while the model is answering stops the answer early.

After every answer a stats line shows time-to-first-token (TTFT), output tokens
and generation speed, which is a quick sanity check on server performance.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request

DEFAULT_SYSTEM = (
    "You are a friendly, patient assistant for people at a university, most of whom "
    "are not technical experts. Answer in clear, plain language. Avoid jargon; if you "
    "must use a technical term, explain it in a few words. Keep answers concise and "
    "well organized, and say so honestly when you are not sure about something."
)

TTY = sys.stdout.isatty()


def dim(text):
    return f"\033[2m{text}\033[0m" if TTY else text


def bold(text):
    return f"\033[1m{text}\033[0m" if TTY else text


def parse_args():
    p = argparse.ArgumentParser(description="Chat with an OpenAI-compatible LLM server.")
    p.add_argument("--base-url", default="http://127.0.0.1:8000/v1",
                   help="server API root (default: %(default)s)")
    p.add_argument("--model", default="chat", help="model name the server exposes (default: %(default)s)")
    p.add_argument("--system", default=DEFAULT_SYSTEM, help="system prompt (default: plain-language assistant)")
    p.add_argument("--prompt", help="ask a single question, print the answer and exit")
    p.add_argument("--think", action="store_true",
                   help="let reasoning models think before answering (slower; off by default)")
    p.add_argument("--show-thinking", action="store_true", help="print the model's reasoning, if any, dimmed")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top-p", type=float, default=0.8)
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--presence-penalty", type=float, default=1.5)
    p.add_argument("--max-tokens", type=int, default=2048)
    p.add_argument("--timeout", type=float, default=600, help="seconds to wait for the server (default: %(default)s)")
    p.add_argument("--no-stats", action="store_true", help="hide the stats line after each answer")
    return p.parse_args()


def build_request(args, messages):
    body = {
        "model": args.model,
        "messages": messages,
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "presence_penalty": args.presence_penalty,
        "max_tokens": args.max_tokens,
    }
    if not args.think:
        # Qwen3/3.5-style templates honour this; servers/models that don't simply ignore it.
        body["chat_template_kwargs"] = {"enable_thinking": False}
    return urllib.request.Request(
        args.base_url.rstrip("/") + "/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    )


def stream_answer(args, messages):
    """Send the conversation, print the reply as it streams. Returns (answer_text, stats)."""
    start = time.perf_counter()
    first = None
    answer, chunks, usage = [], 0, None
    thinking_open = False

    with urllib.request.urlopen(build_request(args, messages), timeout=args.timeout) as resp:
        try:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                event = json.loads(data)
                if event.get("usage"):
                    usage = event["usage"]
                for choice in event.get("choices") or []:
                    delta = choice.get("delta") or {}
                    reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                    content = delta.get("content")
                    if (reasoning or content) and first is None:
                        first = time.perf_counter()
                    if reasoning:
                        chunks += 1
                        if args.show_thinking:
                            thinking_open = True
                            print(dim(reasoning), end="", flush=True)
                    if content:
                        chunks += 1
                        if thinking_open:
                            print("\n")
                            thinking_open = False
                        answer.append(content)
                        print(content, end="", flush=True)
        except KeyboardInterrupt:
            print(dim(" [stopped]"), end="")
    end = time.perf_counter()
    print()

    tokens = (usage or {}).get("completion_tokens") or chunks
    stats = {
        "ttft": (first - start) if first else None,
        "tokens": tokens,
        "approx": not usage,
        "tok_s": tokens / (end - first) if first and end > first else None,
        "total": end - start,
    }
    return "".join(answer), stats


def format_stats(s):
    parts = []
    if s["ttft"] is not None:
        parts.append(f"TTFT {s['ttft']:.2f}s")
    parts.append(f"{'~' if s['approx'] else ''}{s['tokens']} tokens")
    if s["tok_s"]:
        parts.append(f"{s['tok_s']:.1f} tok/s")
    parts.append(f"total {s['total']:.1f}s")
    return dim("[" + " | ".join(parts) + "]")


def ask(args, messages):
    """Run one turn; on success append the assistant reply to messages. Returns True if OK."""
    try:
        answer, stats = stream_answer(args, messages)
    except KeyboardInterrupt:  # pressed before the first token arrived
        print(dim(" [cancelled]"))
        return False
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:500]
        print(f"\nServer returned HTTP {e.code}: {detail}", file=sys.stderr)
        return False
    except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
        reason = getattr(e, "reason", e)
        print(f"\nCan't reach the model server at {args.base_url} ({reason}).\n"
              "Is it running? Start vLLM with serving/vllm/serve.sh", file=sys.stderr)
        return False
    messages.append({"role": "assistant", "content": answer})
    if not args.no_stats:
        print(format_stats(stats))
    return True


def main():
    args = parse_args()
    history = [{"role": "system", "content": args.system}] if args.system else []
    messages = list(history)

    if args.prompt:
        messages.append({"role": "user", "content": args.prompt})
        sys.exit(0 if ask(args, messages) else 1)

    print(bold(f"Chatting with '{args.model}' at {args.base_url}"))
    print(dim("Type your question and press Enter. /reset starts over, /quit exits.\n"))
    while True:
        try:
            text = input(bold("You: ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text:
            continue
        if text in ("/quit", "/exit", "/q"):
            break
        if text == "/reset":
            messages = list(history)
            print(dim("(conversation cleared)\n"))
            continue
        messages.append({"role": "user", "content": text})
        print(bold("Assistant: "), end="", flush=True)
        if not ask(args, messages):
            messages.pop()  # drop the unanswered question so the user can retry
        print()


if __name__ == "__main__":
    main()
