"""Helpers for humaneval_chat: pull the code block out of a chat answer and score pass@1."""
import re

import evaluate as hf_evaluate

compute_ = hf_evaluate.load("code_eval")
FENCE = re.compile(r"```(?:python|py)?[ \t]*\n(.*?)(?:```|\Z)", re.DOTALL)


def pass_at_k(references: list[str], predictions: list[list[str]], k: list[int] | None = None):
    assert k is not None
    if isinstance(k, int):
        k = [k]
    return compute_.compute(references=references, predictions=predictions, k=k)[0]


def extract(resp: str, doc: dict) -> str:
    m = FENCE.search(resp)
    code = m.group(1) if m else resp
    if f"def {doc['entry_point']}" in code:
        # Full function returned. Keep the original prompt in front so its imports survive;
        # the model's definition then replaces the docstring-only one.
        return doc["prompt"] + "\n" + code
    return doc["prompt"] + code   # body only


def build_predictions_chat(resps: list[list[str]], docs: list[dict]) -> list[list[str]]:
    return [[extract(r, doc) for r in resp] for resp, doc in zip(resps, docs, strict=True)]
