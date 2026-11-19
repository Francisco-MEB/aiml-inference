"""Quality phase: lm-evaluation-harness against the running server's OpenAI-compatible API.
Generative tasks only (no logprobs), thinking off (the server's chat-template default and the
model's extra_body), greedy decoding. The core suite and its limits are fixed by the protocol
(protocol.QUALITY_TASKS / QUALITY_LIMITS); --quality-tasks / --quality-limit are for tests."""
import glob
import json
import os
import subprocess
import sys

import common
import engines
import protocol


def gen_kwargs(cfg, model):
    q = cfg["quality"]
    kw = [q.get("gen_kwargs") or "temperature=0",
          f"max_gen_toks={q.get('max_gen_toks', protocol.QUALITY_MAX_GEN_TOKS)}"]
    for k, v in (model.get("extra_body") or {}).items():   # e.g. gpt-oss reasoning_effort=low
        kw.append(f"{k}={v}")
    return ",".join(kw)


def task_plan(cfg, limit_override=None):
    """[(tasks, limit)] groups: lm-eval takes one --limit per call, so tasks with different
    limits run as separate calls."""
    q = cfg["quality"]
    tasks = q["tasks"]
    if limit_override:
        return [(tasks, limit_override)]
    limits = q.get("limits") or {}
    groups = {}
    for t in tasks:
        groups.setdefault(limits.get(t), []).append(t)
    return [(ts, lim) for lim, ts in groups.items()]


def run(cfg, model, out_dir, limit=None, log=print):
    q = cfg["quality"]
    skipped, runnable = [], []
    for t in q["tasks"]:
        if t in q.get("gated_tasks", []) and not (os.environ.get("HF_TOKEN")
                                                  or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
            skipped.append({"task": t, "reason": "gated Hugging Face dataset and no HF_TOKEN set "
                                                 "(accept the dataset terms, then export HF_TOKEN)"})
        elif t in q.get("code_exec_tasks", []) and not q.get("allow_code_execution"):
            skipped.append({"task": t, "reason": "runs model-written code; "
                                                 "allow_code_execution is false"})
        else:
            runnable.append(t)
    rec = {"model_key": model["key"], "model_tag": model["tag"], "limit": limit,
           "limits": q.get("limits"), "tasks_requested": q["tasks"], "tasks_skipped": skipped,
           "gen_kwargs": gen_kwargs(cfg, model), "commands": [], "results": [],
           "returncode": 0}
    if not runnable:
        return rec
    base = engines.base_url(cfg) + "/v1/chat/completions"
    model_args = (f"model={engines.request_model(cfg, model)},base_url={base},"
                  f"num_concurrent={q['num_concurrent']},tokenized_requests=False,"
                  f"max_retries=3,timeout=3600")
    env = dict(os.environ, OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY", "none"))
    cfg_q = dict(cfg, quality=dict(q, tasks=runnable))
    for i, (tasks, lim) in enumerate(task_plan(cfg_q, limit)):
        cmd = [sys.executable, str(common.BENCH / "harness/lm_eval_bench.py"), "run",
               "--model", "bench-chat-completions", "--model_args", model_args,
               "--tasks", ",".join(tasks), "--apply_chat_template",
               "--gen_kwargs", rec["gen_kwargs"],
               "--output_path", str(out_dir), "--log_samples"]
        if lim:
            cmd += ["--limit", str(lim)]
        if q.get("include_path"):          # the club's own tasks (humaneval_chat)
            cmd += ["--include_path", str(common.BENCH / q["include_path"])]
        if any(t in q.get("code_exec_tasks", []) for t in tasks):
            cmd.append("--confirm_run_unsafe_code")
            env["HF_ALLOW_CODE_EVAL"] = "1"
        log("  " + " ".join(cmd))
        rec["commands"].append(cmd)
        with open(os.path.join(out_dir, f"lm_eval_{model['key']}_{i}.log"), "w") as lf:
            p = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, check=False,
                               cwd=str(out_dir))
        rec["returncode"] = rec["returncode"] or p.returncode
    for path in sorted(glob.glob(os.path.join(out_dir, "**", "results_*.json"), recursive=True)):
        with open(path) as f:
            data = json.load(f)
        for task, metrics in data.get("results", {}).items():
            n = (data.get("n-samples", {}).get(task) or {}).get("effective")
            for k, v in metrics.items():
                if k == "alias" or "_stderr" in k or not isinstance(v, (int, float)):
                    continue
                name = k.split(",")[0]
                rec["results"].append({"task": task, "metric": k, "value": v, "n": n,
                                       "stderr": metrics.get(k.replace(name, name + "_stderr"))})
    rec["lm_eval_version"] = common.sh([sys.executable, "-c",
                                        "import lm_eval; print(lm_eval.__version__)"])
    return rec
