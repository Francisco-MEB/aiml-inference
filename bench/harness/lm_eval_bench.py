"""`lm-eval` with one extra model type, `bench-chat-completions`: lm-eval's
local-chat-completions without the fixed `seed` it adds to every request.

Why: mlx_lm.server serves any request that carries a seed on its own (no batching), which
would make the quality phase on a Mac many times slower. Temperature is 0, so the seed changes
nothing on either engine; we drop it everywhere so vLLM and MLX get identical requests.

  .venv/bin/python bench/harness/lm_eval_bench.py run --model bench-chat-completions ...
"""
import os
import sys

# Keep bench/harness (hf.py, common.py, ...) off the import path so it can't shadow packages
# lm-eval imports.
_here = os.path.dirname(os.path.abspath(__file__))
sys.path[:] = [p for p in sys.path if os.path.abspath(p or ".") != _here]

from lm_eval.__main__ import cli_evaluate  # noqa: E402
from lm_eval.api.registry import register_model  # noqa: E402
from lm_eval.models.openai_completions import LocalChatCompletion  # noqa: E402


@register_model("bench-chat-completions")
class BenchChatCompletion(LocalChatCompletion):
    def _create_payload(self, messages, generate=False, gen_kwargs=None, seed=1234, eos=None,
                        **kwargs):
        payload = super()._create_payload(messages, generate=generate, gen_kwargs=gen_kwargs,
                                          seed=seed, eos=eos, **kwargs)
        payload.pop("seed", None)
        return payload


if __name__ == "__main__":
    sys.exit(cli_evaluate())
