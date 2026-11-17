"""Benchmark 3 Qwen models x 3 prompt lengths on one GB10 through Ollama.

    uv run architecture/bench_qwen.py

Measures, per request:
  prefill tok/s  how fast the model READS the prompt
  decode tok/s   how fast it WRITES the answer
  TTFT           seconds until the first output token appears
  $/1M tokens    cost to generate a million output tokens (hardware + electricity)

Pull the models first (ollama pull <tag>); missing ones are skipped.
Results are printed and appended to architecture/bench_results.csv.
"""

import csv
import json
import socket
import subprocess
import time
import urllib.request
import uuid
from datetime import datetime
from pathlib import Path

OLLAMA = "http://localhost:11434"
MODELS = [
    "qwen3.5:9b-q4_K_M",         # small dense   (~6 GB)
    "qwen3.5:27b-q4_K_M",        # medium dense  (~17 GB)
    "qwen3.5:122b-a10b-q4_K_M",  # large MoE: 122B total, ~10B active (~81 GB)
]
RUNS = 3            # repeats per (model, prompt)
NUM_PREDICT = 256   # output tokens per request
NUM_CTX = 8192      # context window

# Cost assumptions. Edit these to match reality.
NODE_PRICE_USD = 4000          # price of one GB10 node (assumption)
NODE_LIFETIME_YEARS = 3        # spread the price over this long
ELECTRICITY_USD_PER_KWH = 0.12

# Three prompt lengths. Long ones repeat a paragraph; the exact token count
# comes back from Ollama as prompt_eval_count, so it doesn't need to be exact.
PARAGRAPH = (
    "The university library extended its hours during finals week. Students used the "
    "study rooms late into the night, and staff added extra printers and a coffee cart "
    "near the entrance. Feedback forms showed most students wanted the change to be "
    "permanent, though some asked for quieter areas on the upper floors. "
)
QUESTION = "Summarize the text above in three sentences."
PROMPTS = {
    "short": "Explain how a hash table works and when you would use one.",  # ~30 tokens
    "medium": PARAGRAPH * 16 + QUESTION,   # ~1,000 tokens
    "long": PARAGRAPH * 64 + QUESTION,     # ~4,000 tokens
}

OUT_CSV = Path(__file__).parent / "bench_results.csv"


def ollama(path, body=None):
    """GET (no body) or POST JSON to the Ollama API and return the decoded JSON."""
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(OLLAMA + path, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        return json.loads(r.read())


def generate(model, prompt):
    return ollama("/api/generate", {
        "model": model, "prompt": prompt, "stream": False,
        "options": {"temperature": 0, "seed": 42, "num_predict": NUM_PREDICT, "num_ctx": NUM_CTX},
    })


def start_power_meter():
    """nvidia-smi prints GPU watts every 500 ms until we stop it."""
    return subprocess.Popen(
        ["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits", "-lms", "500"],
        stdout=subprocess.PIPE, text=True)


def stop_power_meter(proc):
    """Stop nvidia-smi and return the average watts it saw."""
    proc.terminate()
    out, _ = proc.communicate()
    watts = [float(x) for x in out.split() if x.replace(".", "", 1).isdigit()]
    return sum(watts) / len(watts) if watts else float("nan")


def cost_per_million_tokens(decode_tps, watts):
    """What one node costs per hour, divided by how many tokens it writes per hour.

    Assumes one user at a time. With several users at once, the node writes more
    tokens per hour, so this cost goes down.
    """
    hardware_per_hour = NODE_PRICE_USD / (NODE_LIFETIME_YEARS * 365 * 24)
    power_per_hour = watts / 1000 * ELECTRICITY_USD_PER_KWH
    tokens_per_hour = decode_tps * 3600
    return (hardware_per_hour + power_per_hour) / tokens_per_hour * 1e6


def main():
    pulled = {m["name"] for m in ollama("/api/tags")["models"]}
    version = ollama("/api/version")["version"]
    host = socket.gethostname()
    new_file = not OUT_CSV.exists()
    out = OUT_CSV.open("a", newline="")
    writer = csv.writer(out)
    if new_file:
        writer.writerow(["timestamp", "host", "ollama_version", "model", "prompt", "run",
                         "input_tokens", "output_tokens", "prefill_tps", "decode_tps",
                         "ttft_s", "avg_gpu_watts", "usd_per_1m_tokens"])

    print(f"{'model':26} {'prompt':6}  {'in':>5}  {'prefill t/s':>11}  {'decode t/s':>10}"
          f"  {'TTFT s':>6}  {'watts':>5}  {'$/1M tok':>8}")

    for model in MODELS:
        if model not in pulled:
            print(f"{model}: not pulled, skipping (run: ollama pull {model})")
            continue

        # load weights into memory with bs prompt
        generate(model, "Say hello.")

        for name, text in PROMPTS.items():
            for run in range(RUNS):
                # random tag at the start makes every prompt new so prefill is accurate
                prompt = f"[{uuid.uuid4().hex[:8]}] {text}"

                meter = start_power_meter()
                r = generate(model, prompt)
                watts = stop_power_meter(meter)

                # Ollama reports all durations in nanoseconds.
                prefill_tps = r["prompt_eval_count"] / (r["prompt_eval_duration"] / 1e9)
                decode_tps = r["eval_count"] / (r["eval_duration"] / 1e9)
                ttft = (r["load_duration"] + r["prompt_eval_duration"]) / 1e9
                cost = cost_per_million_tokens(decode_tps, watts)

                print(f"{model:26} {name:6}  {r['prompt_eval_count']:>5}  {prefill_tps:>11.0f}"
                      f"  {decode_tps:>10.1f}  {ttft:>6.2f}  {watts:>5.0f}  {cost:>8.2f}")
                writer.writerow([datetime.now().isoformat(timespec="seconds"), host, version,
                                 model, name, run, r["prompt_eval_count"], r["eval_count"],
                                 round(prefill_tps, 1), round(decode_tps, 2), round(ttft, 3),
                                 round(watts, 1), round(cost, 3)])
                out.flush()

        # Unload so the next model has the whole memory pool to itself.
        ollama("/api/generate", {"model": model, "keep_alive": 0})
        time.sleep(2)

    out.close()
    print(f"\nSaved to {OUT_CSV}")


if __name__ == "__main__":
    main()
