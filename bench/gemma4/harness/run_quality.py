"""Quality eval stub: lm-evaluation-harness against the OpenAI-compatible endpoint.
Generative tasks only (Ollama exposes no logprobs)."""
import glob
import json
import os
import subprocess

import common


def run(cfg, model, out_dir, limit=None, log=print):
    q = cfg["quality"]
    tasks, skipped = [], []
    for t in q["tasks"]:
        if t in q.get("gated_tasks", []) and not (os.environ.get("HF_TOKEN")
                                                  or os.environ.get("HUGGING_FACE_HUB_TOKEN")):
            skipped.append({"task": t,
                            "reason": "gated Hugging Face dataset and no HF_TOKEN set"})
        elif t in q.get("code_exec_tasks", []) and not q.get("allow_code_execution"):
            skipped.append({"task": t, "reason": ("runs model-written code; "
                                                  "allow_code_execution is false")})
        else:
            tasks.append(t)
    limit = limit if limit is not None else q.get("limit")
    lm_eval = common.REPO / ".venv/bin/lm_eval"
    model_args = (f"model={model['tag']},base_url={cfg['backend']['base_url']}"
                  f"/chat/completions,num_concurrent={q['num_concurrent']},"
                  f"tokenized_requests=False,max_retries=3")
    cmd = [str(lm_eval), "run", "--model", "local-chat-completions",
           "--model_args", model_args,
           "--tasks", ",".join(tasks), "--apply_chat_template",
           "--gen_kwargs", q["gen_kwargs"], "--seed", str(cfg["request"]["seed"]),
           "--output_path", str(out_dir), "--log_samples"]
    if limit:
        cmd += ["--limit", str(limit)]
    if q.get("include_path"):
        cmd += ["--include_path", str(common.BENCH / q["include_path"])]
    env = dict(os.environ, OPENAI_API_KEY=os.environ.get("OPENAI_API_KEY", "none"))
    if any(t in q.get("code_exec_tasks", []) for t in tasks):
        cmd.append("--confirm_run_unsafe_code")
        env["HF_ALLOW_CODE_EVAL"] = "1"
    log("  " + " ".join(cmd))
    rec = {"model_key": model["key"], "model_tag": model["tag"], "command": cmd, "limit": limit,
           "tasks_requested": q["tasks"], "tasks_skipped": skipped, "results": []}
    if not tasks:
        return rec
    with open(os.path.join(out_dir, f"lm_eval_{model['key']}.log"), "w") as lf:
        p = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT, check=False)
    rec["returncode"] = p.returncode
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
    return rec
