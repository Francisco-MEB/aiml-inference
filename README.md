# aiml-inference

The goal of this project is to provide an ***evaluation*** and ***analysis*** of potential architectures for the university to use when they scale up compute. 

## Use Cases

1. Faculty/Staff: Professors want to be able to create tools for their classroom which might require access to sensitive information. Localized LLMs provide a way for them to avoid violating copyright and FERPA laws. **PRIMARY USE: Chatbot like architectures, tool calling, RAG (large context windows needed)**

2. Coding Agents: Tokens are EXPENSIVE. We want to give engineers on campus a way to utilize the power of coding agents while not incurring the variable cost from the tokens. **PRIMARY USE: development environments, coding jobs, harness level tasks (the model must interact with its environment)**

## Repository layout

| Path | Contents |
|---|---|
| [`examples/`](examples/) | Minimal "hello world" usage of local inference servers (e.g. Ollama from Python). |
| [`scripts/`](scripts/) | Benchmarking and measurement tools. See [`scripts/README.md`](scripts/README.md) for full docs, including the [Ollama concurrency probe](scripts/README.md#ollama_concurrency_probepy). |
