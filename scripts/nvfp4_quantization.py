#!/usr/bin/env python3
"""Download a Hugging Face model and quantize it to NVFP4 on a Blackwell GPU (GB10).

    uv run --group quantize scripts/nvfp4_quantization.py Qwen/Qwen3-4B

transformers quantizes the weights to NVFP4 while it loads them. The first run downloads
the full-precision weights to the Hugging Face cache (~/.cache/huggingface/hub); later runs reuse them.
The NVFP4 model lives in GPU memory only: transformers can't save it yet (NVFP4Config is not serializable).

Requires a CUDA build of torch and `kernels>=0.18.2,<0.19`. Gated models (e.g. meta-llama/*)
also need access granted on huggingface.co and `hf auth login`.
"""

import argparse
import sys

from huggingface_hub import HfApi, auth_check
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
from transformers import AutoModelForCausalLM, AutoTokenizer, NVFP4Config


def find_model(model_id: str) -> None:
    """Exit with a clear message if model_id isn't on the Hub or we can't access it."""
    try:
        auth_check(model_id)
    except GatedRepoError:
        sys.exit(
            f"{model_id} is gated. Request access at https://huggingface.co/{model_id}, then run `hf auth login`."
        )
    except RepositoryNotFoundError:
        matches = HfApi().list_models(
            search=model_id.split("/")[-1], sort="downloads", limit=5
        )
        suggestions = (
            "".join(f"\n  {m.id}" for m in matches)
            or " none, search https://huggingface.co/models"
        )
        sys.exit(
            f"No model '{model_id}' on Hugging Face (or it's private). Did you mean:{suggestions}"
        )


def quantize(model_id: str):
    """Download the model (if not cached) and load it onto the GPU with its linear layers in NVFP4."""
    return AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=NVFP4Config(), device_map="cuda"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face model and quantize it to NVFP4."
    )
    parser.add_argument("model", help="Hugging Face model id, e.g. Qwen/Qwen3-4B")
    model_id = parser.parse_args().model

    find_model(model_id)
    model = quantize(model_id)
    print(f"{model_id} loaded in NVFP4: {model.get_memory_footprint() / 1e9:.1f} GB")

    # Smoke test: since the result can't be saved, a short generation is the proof the NVFP4 kernels run.
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    inputs = tokenizer("The capital of France is", return_tensors="pt").to(model.device)
    # transformers' type hints wrongly reject generate() on from_pretrained's result; it works at runtime.
    output = model.generate(**inputs, max_new_tokens=20)  # pyright: ignore[reportAttributeAccessIssue]
    print(tokenizer.decode(output[0], skip_special_tokens=True))


if __name__ == "__main__":
    main()
