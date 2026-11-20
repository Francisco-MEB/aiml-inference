#!/usr/bin/env python3
"""The club's model catalog (bench/catalog/models.json) and the study planner.

  python3 bench/harness/catalog.py list                     # families and models
  python3 bench/harness/catalog.py plan qwen                # what a Qwen study runs on THIS machine
  python3 bench/harness/catalog.py plan gpt-oss-120b        # a single model
  python3 bench/harness/catalog.py plan qwen --machine mac:64 --write   # pretend 64 GB Mac; write config
  python3 bench/harness/catalog.py plan qwen --machine mini # our 32 GB Mac minis (studio: 128 GB)
  python3 bench/harness/catalog.py pin                      # maintainers: fill revisions etc. from the Hub
  python3 bench/harness/catalog.py verify                   # maintainers: every pin still resolves

A study is one family x one machine x one quantization (protocol section 1). Each family lists
its builds per engine from highest to lowest precision. plan_study() takes the largest model of
the family that fits this machine in any build, then the highest-precision build in which that
model fits (or the family's preferred_build for this engine, when it names one), and runs every
family model that build has and that fits. Models without a build in that format are reported,
never substituted."""
import argparse
import copy
import json
import os
import platform
import re
import sys
from datetime import date

import common
import protocol

CATALOG = common.BENCH / "catalog" / "models.json"
KV_RESERVE = 0.15          # share of the engine's memory budget kept free for the KV cache
MLX_MEMORY_FRACTION = 0.75  # macOS lets the GPU wire about 3/4 of RAM by default


def load():
    return common.load_json(CATALOG)


def save(cat):
    common.write_json(CATALOG, cat)


# ---- machine ------------------------------------------------------------------------------

def detect_machine():
    """{"engine", "hostname", "mem_bytes", "hardware"} for this machine."""
    host = platform.node().split(".")[0]
    if platform.system() == "Darwin":
        mem = int(common.sh("sysctl -n hw.memsize") or 0)
        chip = common.sh("sysctl -n machdep.cpu.brand_string") or "Apple Silicon"
        model = common.sh("sysctl -n hw.model")
        # "Model Name: Mac Studio" picks this Mac's folder under models/ (mac/studio, mac/mini).
        hw = common.sh("system_profiler SPHardwareDataType")
        name = next((line.split(":", 1)[1].strip() for line in hw.splitlines()
                     if line.strip().startswith("Model Name:")), "")
        return {"engine": "mlx", "hostname": host, "mem_bytes": mem,
                "hardware": f"{model} {chip}, {mem / 2**30:.0f} GiB unified".strip(),
                "models_dir": mac_models_dir(name)}
    mem = 0
    try:
        with open("/proc/meminfo") as f:
            mem = int(re.search(r"MemTotal:\s+(\d+)", f.read()).group(1)) * 1024
    except (OSError, AttributeError):
        pass
    gpu = common.sh("nvidia-smi --query-gpu=name --format=csv,noheader").splitlines()
    return {"engine": "vllm", "hostname": host, "mem_bytes": mem,
            "hardware": f"{gpu[0] if gpu else 'unknown GPU'}, {mem / 2**30:.0f} GiB "
                        f"{'unified' if gpu and 'GB10' in gpu[0] else 'system'} memory",
            "models_dir": "dell"}


def mac_models_dir(model_name):
    # A Mac's folder under models/, from its model name: "Mac Studio" -> mac/studio,
    # "Mac mini" -> mac/mini, "MacBook Pro" -> mac/macbook-pro.
    name = model_name.strip().lower()
    name = name[len("mac "):] if name.startswith("mac ") else name
    return "mac/" + (re.sub(r"[^a-z0-9]+", "-", name).strip("-") or "unknown")


def parse_machine(spec):
    """'auto', 'dell' (a GB10 like ours), 'studio' / 'mini' (our 128 GB Mac Studio and 32 GB
    Mac minis) or 'mac:<GiB>' (for planning on another machine)."""
    if spec in (None, "auto"):
        return detect_machine()
    if spec == "dell":
        return {"engine": "vllm", "hostname": "gb10", "mem_bytes": 127533184 * 1024,
                "hardware": "Dell Pro Max GB10, 122 GiB unified memory", "models_dir": "dell"}
    if spec in ("studio", "mini"):
        gib, label = (128, "Mac Studio") if spec == "studio" else (32, "Mac mini")
        return {"engine": "mlx", "hostname": f"mac-{spec}", "mem_bytes": gib * 2**30,
                "hardware": f"{label}, {gib} GiB unified memory", "models_dir": f"mac/{spec}"}
    if spec.startswith("mac:"):
        gib = float(spec.split(":")[1])
        return {"engine": "mlx", "hostname": f"mac-{gib:.0f}g", "mem_bytes": int(gib * 2**30),
                "hardware": f"Mac, {gib:.0f} GiB unified memory",
                "models_dir": f"mac/{gib:.0f}gb"}
    raise SystemExit(f"unknown machine {spec!r}: use auto, dell, studio, mini or mac:<GiB>")


def weight_budget_bytes(machine):
    """Largest weights a model may have on this machine and still leave KV_RESERVE of the
    engine's memory budget for the KV cache."""
    frac = protocol.VLLM_GPU_MEMORY_UTILIZATION if machine["engine"] == "vllm" \
        else MLX_MEMORY_FRACTION
    return machine["mem_bytes"] * frac * (1 - KV_RESERVE)


# ---- lookup -------------------------------------------------------------------------------

def find(cat, name):
    """(family_key, model_key or None) for a family name, model key or repo id."""
    n = name.strip().lower()
    for fk, fam in cat["families"].items():
        if n in (fk, fam["label"].lower()) or n in fam["label"].lower().split():
            return fk, None
        for mk in fam["models"]:
            if n == mk:
                return fk, mk
        for builds in fam["builds"].values():
            for b in builds:
                for mk, x in b["models"].items():
                    if n == x["repo"].lower():
                        return fk, mk
    return None, None


def _fits(build_model, budget):
    gb = build_model.get("weights_gb")
    return gb is not None and gb * 1e9 <= budget


def plan_study(name, machine=None, cat=None):
    """The study a member gets for a family (or one model) on this machine.
    Returns {"family", "build", "models": [...], "left_out": [...], "budget_gb", ...}."""
    cat = cat or load()
    machine = machine or detect_machine()
    fk, only = find(cat, name)
    if fk is None:
        raise SystemExit(f"{name!r} is not in the catalog ({CATALOG}). Families: "
                         f"{', '.join(cat['families'])}. Ask the benchmarks lead to add it.")
    fam = cat["families"][fk]
    builds = fam["builds"].get(machine["engine"], [])
    # A family can prefer one build over the highest-precision default, with the reason in its
    # notes (Gemma: NVFP4 on Dells, 4-bit on Macs). The stable sort keeps the rest in order.
    preferred = (fam.get("preferred_build") or {}).get(machine["engine"])
    builds = sorted(builds, key=lambda b: b["format"] != preferred)
    budget = weight_budget_bytes(machine)
    study_set = fam.get("study") or list(fam["models"])     # the family's small/medium/large
    wanted = [only] if only else list(study_set)
    if any(not x.get("revision") for b in builds for x in b["models"].values()):
        raise SystemExit("catalog has unpinned builds: run `catalog.py pin` first")

    def size(mk):
        return fam["models"][mk]["params_b"]

    def fits_any(mk):
        return any(mk in b["models"] and _fits(b["models"][mk], budget) for b in builds)

    left = []
    # A study model that fits this machine at no build is swapped for the largest other family
    # model that does, so the family keeps three sizes where the catalog allows.
    study = list(study_set)
    extras = sorted((mk for mk in fam["models"] if mk not in study_set and fits_any(mk)),
                    key=size)
    for mk in list(study):
        if not fits_any(mk) and extras:
            sub = extras.pop()
            study.append(sub)
            left.append({"key": mk, "reason": f"too large for this machine at every build; "
                                              f"{sub} runs in its place"})
    if not only:
        wanted = study
    for mk in wanted:
        if not fits_any(mk) and not any(x["key"] == mk for x in left):
            have = [f"{b['format']} {b['models'][mk].get('weights_gb')} GB" for b in builds
                    if mk in b["models"]]
            left.append({"key": mk, "reason": ("too large for this machine at every build ("
                                               + ", ".join(have) + ")") if have
                         else f"no {machine['engine']} build in the catalog"})
    # The build is always the family study's (decided by the study's models), so one model
    # named on its own is measured in the same quantization as its family's study: the
    # highest-precision build that holds every study model that fits; failing that, the one in
    # which the largest of them fits.
    fitting = [mk for mk in dict.fromkeys([*study, *wanted]) if fits_any(mk)]
    deciders = [mk for mk in fitting if mk in study] or fitting
    if not fitting:
        return {"family": fk, "build": None, "models": [], "left_out": left,
                "budget_gb": round(budget / 1e9, 1), "machine": machine}
    build = next((b for b in builds
                  if all(mk in b["models"] and _fits(b["models"][mk], budget)
                         for mk in deciders)), None)
    if build is None:
        largest = max(deciders, key=size)
        build = next(b for b in builds if largest in b["models"]
                     and _fits(b["models"][largest], budget))
    models = []
    for mk in sorted(wanted, key=size):
        if mk not in fitting:          # already listed in left_out
            continue
        x = build["models"].get(mk)
        if x is None:
            if mk in fitting:
                left.append({"key": mk, "reason": f"no {build['format']} build (one "
                                                  f"quantization per study)"})
            continue
        if not _fits(x, budget):
            left.append({"key": mk, "reason": f"{build['format']} build is {x['weights_gb']} "
                                              f"GB, over the {budget / 1e9:.0f} GB budget"})
            continue
        models.append(model_entry(fam, mk, x, build, machine["engine"]))
    return {"family": fk, "family_label": fam["label"], "build": build["format"],
            "publisher": build.get("publisher", ""), "deviation": build.get("deviation", ""),
            "models": models, "left_out": left, "budget_gb": round(budget / 1e9, 1),
            "machine": machine, "notes": fam.get("notes", "")}


def model_entry(fam, mk, x, build, engine):
    """One model as it appears in a study config."""
    meta = fam["models"][mk]
    th = fam.get("thinking", {})
    args = list(meta.get(f"{engine}_args", fam.get(f"{engine}_args", [])))
    if engine == "vllm" and x.get("multimodal"):
        args.append("--language-model-only")
    return {"key": mk, "label": meta["label"], "size_class": meta["size_class"],
            "arch": meta["arch"], "params_b": meta["params_b"],
            "active_params_b": meta["active_params_b"],
            "tag": x["repo"], "repo": x["repo"], "revision": x["revision"],
            "quantization": build["format"], "weight_format": x.get("detected_format", ""),
            # Effective stored bits per parameter (includes unquantized embeddings and scales).
            "bits_per_weight": round(x["weights_gb"] * 8 / meta["params_b"], 2),
            "model_blob_gb": x.get("weights_gb"),
            "gated": x.get("gated", False), "download_exclude": x.get("download_exclude", []),
            "server_args": args,
            "chat_template_kwargs": th.get("chat_template_kwargs") or {},
            "extra_body": dict(th.get("request_extra") or {}),
            "thinking_off": th.get("note", "not set"),
            "thinking_cannot_disable": bool(th.get("cannot_disable"))}


def study_config(plan, member, today=None):
    """A complete study config from a plan (the template's fixed sections + the plan)."""
    m = plan["machine"]
    today = today or date.today().strftime("%Y%m%d")
    cfg = copy.deepcopy(common.load_json(common.BENCH / "configs" / "TEMPLATE.json"))
    sid = f"{plan['family']}_{plan['build']}_{m['hostname']}_{today}"
    cfg["study"] = {"id": sid, "member": member, "machine": m["hostname"],
                    "hardware": m["hardware"], "family": plan["family"],
                    "notes": plan.get("notes", ""), "left_out": plan["left_out"]}
    cfg["quantization"] = plan["build"]
    cfg["quantization_publisher"] = plan.get("publisher", "")
    if plan.get("deviation"):
        cfg["quantization_deviation"] = plan["deviation"]
    cfg["backend"]["type"] = m["engine"]
    cfg["models"] = plan["models"]
    return cfg


# ---- maintainers: pin and verify ------------------------------------------------------------

def pin(cat, only_missing=True, log=print):
    import hf
    n = 0
    for fk, fam in cat["families"].items():
        for engine, builds in fam["builds"].items():
            for b in builds:
                for mk, x in b["models"].items():
                    if only_missing and x.get("revision"):
                        continue
                    try:
                        d = hf.describe(x["repo"], x.get("revision") or "main")
                    except Exception as e:  # noqa: BLE001
                        log(f"  FAILED {x['repo']}: {e}")
                        x["pin_error"] = repr(e)
                        continue
                    x.pop("pin_error", None)
                    x.update({k: d[k] for k in ("revision", "weights_gb", "multimodal",
                                                "gated", "download_exclude")})
                    x["detected_format"] = d["format"]
                    n += 1
                    log(f"  {engine:4} {b['format']:6} {mk:22} {x['repo']:55} "
                        f"{d['revision'][:10]} {d['weights_gb']:7.2f} GB {d['format']}"
                        f"{' (gated)' if d['gated'] else ''}")
    return n


def verify(cat, log=print):
    import hf
    bad = 0
    for fam in cat["families"].values():
        for engine, builds in fam["builds"].items():
            for b in builds:
                for mk, x in b["models"].items():
                    try:
                        d = hf.describe(x["repo"], x["revision"])
                        ok = d["revision"] == x["revision"] and \
                            abs(d["weights_gb"] - x["weights_gb"]) < 0.01
                    except Exception as e:  # noqa: BLE001
                        ok, d = False, {"error": repr(e)}
                    bad += not ok
                    log(f"  {'ok  ' if ok else 'FAIL'} {engine} {b['format']} {x['repo']}"
                        f"@{x.get('revision', '')[:10]} {'' if ok else d}")
    return bad


def print_plan(p):
    m = p["machine"]
    print(f"machine: {m['hostname']} ({m['hardware']}), engine {m['engine']}, "
          f"weights budget {p['budget_gb']} GB")
    if not p["models"]:
        print(f"nothing in {p['family']} fits this machine")
    else:
        print(f"study: {p['family']} at {p['build']} ({p['publisher']})"
              + (f"  DEVIATION: {p['deviation']}" if p.get("deviation") else ""))
        for x in p["models"]:
            print(f"  {x['key']:22} {x['size_class']:6} {x['arch']:5} {x['repo']:55} "
                  f"{x['model_blob_gb']:7.2f} GB  thinking: {x['thinking_off']}")
    for x in p["left_out"]:
        print(f"  left out: {x['key']}: {x['reason']}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list")
    pl = sub.add_parser("plan")
    pl.add_argument("name", help="family (qwen), model key (gpt-oss-120b) or repo id")
    pl.add_argument("--machine", default="auto", help="auto | dell | studio | mini | mac:<GiB>")
    pl.add_argument("--member", default=os.environ.get("USER", ""))
    pl.add_argument("--write", action="store_true", help="write bench/configs/<study_id>.json")
    pl.add_argument("--json", action="store_true")
    pi = sub.add_parser("pin")
    pi.add_argument("--all", action="store_true", help="re-pin everything to the latest revision")
    sub.add_parser("verify")
    a = ap.parse_args()
    cat = load()
    if a.cmd == "list":
        for fk, fam in cat["families"].items():
            print(f"{fk}: {fam['label']}")
            for mk, meta in fam["models"].items():
                engines = [e for e, bs in fam["builds"].items()
                           if any(mk in b["models"] for b in bs)]
                print(f"  {mk:22} {meta['size_class']:6} {meta['arch']:5} "
                      f"{meta['params_b']:6.1f}B  engines: {', '.join(engines) or 'none'}")
    elif a.cmd == "plan":
        p = plan_study(a.name, parse_machine(a.machine), cat)
        if a.json:
            print(json.dumps(p, indent=2))
        else:
            print_plan(p)
        if a.write:
            if not p["models"]:
                raise SystemExit("nothing to write")
            cfg = study_config(p, a.member)
            path = common.BENCH / "configs" / f"{cfg['study']['id']}.json"
            common.write_json(path, cfg)
            print(f"wrote {os.path.relpath(path, common.REPO)}")
    elif a.cmd == "pin":
        n = pin(cat, only_missing=not a.all)
        save(cat)
        print(f"pinned {n} builds into {CATALOG}")
    elif a.cmd == "verify":
        sys.exit(1 if verify(cat) else 0)


if __name__ == "__main__":
    main()
