"""
Base eval script. Copy this file to make your own evals.

Setup (one time):
    ollama pull <model_name>      # see week2.md for which models

Run:
    uv run architecture/base_eval.py
    uv run architecture/base_eval.py qwen3.5:9b      # pick a model from the command line

Adding your own eval:
    1. Write a function in the EVAL FUNCTIONS section that takes `model`
       and returns a dict of metrics, e.g. {"tokens_per_sec": 42.0}
    2. Add it to the EVALS list at the bottom
"""

import sys
import time

import ollama

MODEL = "llama3.2"  # change this, or pass a model name on the command line


# SETUP #


def load_model(model):
    print(f"Loading {model}...")

    # Empty prompt = load the model without generating anything
    start = time.perf_counter()
    ollama.generate(model=model, prompt="", keep_alive="10m")
    load_time = time.perf_counter() - start

    print(f"Loaded in {load_time:.2f}s\n")
    return load_time


def unload_model(model):
    ollama.generate(model=model, prompt="", keep_alive=0)


# EVAL FUNCTIONS 
# Use this sections to write your own evals


def eval_tokens_per_second(model):
    prompt = "Write a short paragraph explaining how a CPU works."
    response = ollama.generate(model=model, prompt=prompt)

    # Ollama durations are in nanoseconds
    prompt_tokens = response.prompt_eval_count
    prompt_seconds = response.prompt_eval_duration / 1e9
    generated_tokens = response.eval_count
    generated_seconds = response.eval_duration / 1e9
    total_seconds = response.total_duration / 1e9

    prompt_tokens_per_sec = prompt_tokens / prompt_seconds
    gen_tokens_per_sec = generated_tokens / generated_seconds

    return {
        "prompt_tokens": prompt_tokens,
        "prompt_tokens_per_sec": round(prompt_tokens_per_sec, 2),
        "generated_tokens": generated_tokens,
        "gen_tokens_per_sec": round(gen_tokens_per_sec, 2),
        "total_time_sec": round(total_seconds, 2),
    }


# TODO: add your own evals here, e.g.
#   def eval_time_to_first_token(model): ...
#   def eval_long_context(model): ...
#   def eval_memory_usage(model): ...


# Main Execution #

EVALS = [
    eval_tokens_per_second,
    # add your eval functions here
]


def main():
    if len(sys.argv) > 1:
        model = sys.argv[1]
    else:
        model = MODEL

    results = {"model": model}
    results["load_time_sec"] = round(load_model(model), 2)

    for eval_fn in EVALS:
        print(f"Running {eval_fn.__name__}...")
        results[eval_fn.__name__] = eval_fn(model)

    unload_model(model)

    print(f"\n===== Results: {model} =====")
    print(f"load_time_sec: {results['load_time_sec']}")

    for eval_fn in EVALS:
        print(f"\n{eval_fn.__name__}")
        for metric, value in results[eval_fn.__name__].items():
            print(f"  {metric:<24} {value}")


if __name__ == "__main__":
    main()
