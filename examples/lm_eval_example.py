"""
A simple example of benchmarking an Ollama model with lm-evaluation-harness.

lm-evaluation-harness (lm_eval) is a standard tool for testing LLMs on
benchmark tasks. Here we point it at Ollama, which is running on this machine.

Setup (one time):
    1. Download a model:         ollama pull llama3.2

Run:
    uv run lm_eval_example.py

Note: Ollama can only generate text, so use generation tasks like "gsm8k"
(grade-school math). Multiple-choice tasks like "mmlu" or "hellaswag" need
token probabilities, which Ollama doesn't provide.
"""

import lm_eval
from lm_eval.utils import make_table

MODEL = "llama3.2"  # Change this to any model you've pulled (see `ollama list`)

results = lm_eval.simple_evaluate(
    # Talk to Ollama through its OpenAI-compatible chat API
    model="local-chat-completions",
    model_args={
        "model": MODEL,
        "base_url": "http://localhost:11434/v1/chat/completions",
        "tokenizer_backend": None,  # Ollama handles tokenizing for us
    },
    tasks=["gsm8k"],           # The benchmark to run
    apply_chat_template=True,  # Send questions as chat messages
    limit=10,                  # Only test 10 questions so it finishes fast
)

# Print a summary table of the scores
print(make_table(results))

# Or grab a single score yourself
score = results["results"]["gsm8k"]["exact_match,flexible-extract"]
print(f"{MODEL} got {score:.0%} of the math questions right.")
