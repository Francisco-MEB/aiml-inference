#!/usr/bin/env python3
"""Download a Hugging Face model and save an NVFP4 copy of it with llm-compressor.

    uv run --group quantize scripts/nvfp4_quantization.py Qwen/Qwen3-4B

The first run saves the model's full-precision weights to models/dell/<model_id>/ at the repo
root; later runs reuse them. The NVFP4 checkpoint goes to models/dell/<model_id>-NVFP4/ in
compressed-tensors format, which vLLM serves directly:

    vllm serve models/dell/Qwen/Qwen3-4B-NVFP4 --gpu-memory-utilization 0.3

Why llm-compressor: calibration runs in bf16, so it needs no FP4 kernel on this GPU. transformers'
NVFP4Config depends on a Hub kernel built only for sm_100a / sm_120a, which won't run on the
GB10 (sm_121), and it can't save its result anyway.

Vision-language models (e.g. Qwen3.5/3.8) are loaded as text-only causal LMs, so the NVFP4
copy has no vision tower. Gated models (e.g. meta-llama/*) also need access granted on
huggingface.co and `hf auth login`. The GB10 is shared: check `free -g` before running.
"""

import argparse
import os
import sys
from pathlib import Path

# compressed-tensors packs the FP4 weights with a Triton kernel by default, and Triton compiles
# a C helper against Python.h, which the system Python here lacks (no python3.12-dev). The plain
# PyTorch path gives the same checkpoint and only runs once, at save time. Set before importing.
os.environ.setdefault("CT_ENFORCE_EAGER", "1")

from huggingface_hub import HfApi, auth_check, snapshot_download
from huggingface_hub.errors import GatedRepoError, RepositoryNotFoundError
from compressed_tensors.offload import dispatch_model
from llmcompressor import oneshot
from llmcompressor.modifiers.quantization import QuantizationModifier
from transformers import AutoModelForCausalLM, AutoTokenizer

# The Dell's folder under models/ (the Macs can't serve NVFP4), next to the benchmark's Dell builds.
MODELS_DIR = Path(__file__).resolve().parent.parent / "models" / "dell"
# Config, tokenizer and safetensors weights only. Some repos also ship a second full copy of
# the weights (Meta's original/, Mistral's consolidated.safetensors), so skip those.
MODEL_FILES = ["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"]
DUPLICATE_WEIGHTS = ["original/*", "consolidated*"]

# Layers kept in full precision, from llm-compressor's own NVFP4 examples. The output head and
# embeddings are sensitive to rounding; MoE routers and Qwen3.5's linear-attention (Gated
# DeltaNet) projections are small and lose accuracy in FP4. Patterns that match nothing in a
# given model are simply unused, so one list covers dense, MoE and hybrid models.
KEEP_FULL_PRECISION = [
    "re:.*lm_head",
    "re:.*embed_tokens$",
    "re:.*visual.*",
    "re:.*mlp.gate$",
    "re:.*shared_expert_gate$",
    "re:.*linear_attn.*",
]


def find_model(model_id: str) -> None:
    # Exit with a clear message if model_id isn't on the Hub or we can't access it.
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


def download(model_id: str) -> Path:
    # Save the config, tokenizer and weights to models/dell/<model_id>/, skipping files there.
    model_dir = MODELS_DIR / model_id
    snapshot_download(
        model_id,
        local_dir=model_dir,
        allow_patterns=MODEL_FILES,
        ignore_patterns=DUPLICATE_WEIGHTS,
    )
    return model_dir


def quantize(model_dir: Path, samples: int):
    # Weights go to FP4 in blocks of 16 with an FP8 scale per block. Activations are also
    # quantized to FP4 at run time, so a short calibration pass over chat data measures
    # their range first. llm-compressor moves one layer at a time onto the GPU for this.
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype="auto")
    recipe = QuantizationModifier(
        targets="Linear", scheme="NVFP4", ignore=KEEP_FULL_PRECISION
    )
    oneshot(
        model=model,
        recipe=recipe,
        dataset="perfectblend",
        splits=f"train[:{samples * 2}]",
        num_calibration_samples=samples,
        max_seq_length=2048,
    )
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face model and save an NVFP4 copy with llm-compressor."
    )
    parser.add_argument("model", help="Hugging Face model id, e.g. Qwen/Qwen3-4B")
    parser.add_argument(
        "--samples",
        type=int,
        default=256,
        help="calibration samples (default 256; fewer is faster but less accurate)",
    )
    args = parser.parse_args()
    model_id = args.model

    find_model(model_id)
    model_dir = download(model_id)
    print(f"Saved {model_id} to {model_dir}")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = quantize(model_dir, args.samples)

    # Save before the smoke test, so a failed generation doesn't throw away the quantization.
    out_dir = MODELS_DIR / f"{model_id}-NVFP4"
    model.save_pretrained(out_dir, save_compressed=True)
    tokenizer.save_pretrained(out_dir)
    print(f"Saved the NVFP4 checkpoint to {out_dir}")

    # Smoke test: llm-compressor simulates FP4 in bf16 here, so this checks the quantized
    # weights give sensible text. vLLM runs the real FP4 kernels when it serves the checkpoint.
    dispatch_model(model)
    inputs = tokenizer("The capital of France is", return_tensors="pt").to(model.device)
    # transformers' type hints wrongly reject generate() on from_pretrained's result; it works at runtime.
    output = model.generate(**inputs, max_new_tokens=20)  # pyright: ignore[reportAttributeAccessIssue]
    print(tokenizer.decode(output[0], skip_special_tokens=True))
    print(f"\nServe it with: vllm serve {out_dir} --gpu-memory-utilization 0.3")


if __name__ == "__main__":
    main()
