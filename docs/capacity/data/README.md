# Capacity benchmark raw data

One pair of files per run:

- `<model>_<promptset>_<topology>.csv`: from `scripts/openai_capacity_probe.py`, one row per request
- `<model>_<promptset>_<topology>_samples.csv`: from `scripts/sample_node.py`, one row per second

Example: `35b-a3b_chat_1node.csv`. Each probe row carries the model, revision, engine version, servers, prompt set and `server_note`, so a file is self-describing. Summaries and the verdict go in [`../README.md`](../README.md#results).
