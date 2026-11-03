"""
A simple example of using Ollama from Python.

Setup (one time):
    1. Download a model:         ollama pull llama3.2

Run:
    uv run ollama_example.py
"""

import ollama

MODEL = "llama3.2"  # Change this to any model you've pulled (see `ollama list`)


# --- Example 1: Ask a single question ---
response = ollama.chat(
    model=MODEL,
    messages=[{"role": "user", "content": "Explain photosynthesis in one sentence."}],
)
print("Answer:", response["message"]["content"])
print()


# --- Example 2: Stream the answer word by word (like ChatGPT) ---
print("Streaming answer: ", end="")
stream = ollama.chat(
    model=MODEL,
    messages=[{"role": "user", "content": "Give me 3 tips for grading essays faster."}],
    stream=True,
)
for chunk in stream:
    print(chunk["message"]["content"], end="", flush=True)
print("\n")


# --- Example 3: A conversation with memory ---
# The model only "remembers" what you send it, so we keep a list of messages.
messages = [
    {"role": "system", "content": "You are a helpful assistant for teachers."},
    {"role": "user", "content": "My name is Ms. Smith and I teach biology."},
]
reply = ollama.chat(model=MODEL, messages=messages)
messages.append(reply["message"])  # save the model's reply

messages.append({"role": "user", "content": "What subject do I teach?"})
reply = ollama.chat(model=MODEL, messages=messages)
print("Follow-up answer:", reply["message"]["content"])
