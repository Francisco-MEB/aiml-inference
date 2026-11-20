#!/usr/bin/env python3
"""Downloads each configured model at its pinned revision into the Hugging Face cache (this
machine's folder under the repo's models/, or HF_HUB_CACHE) and verifies the weights against the
pinned sha256 of every file (fill_digests.py writes them).
Writes models.json (what was actually on disk) for env.json.

  python3 harness/pull_models.py --config configs/<study>.json [--models k1 k2] [--no-pull]

Downloads go through a pinned huggingface_hub CLI (HF_HUB_VERSION), skip subfolders such as
original/ and metal/ (no server loads them), and need HF_TOKEN for gated repos (Llama)."""
import argparse
import os
import subprocess
import sys
import tempfile
import time

import common
import hf

HF_HUB_VERSION = "2.4.0"          # same version as serving/vllm/uv.lock
PROGRESS_EVERY_S = 60             # a progress line in the run's log this often while downloading


def pull(m, log=print, root=None):
    cmd = ["uvx", "--from", f"huggingface_hub=={HF_HUB_VERSION}", "hf", "download", m["repo"],
           "--revision", m["revision"]]
    for pat in m.get("download_exclude") or hf.DOWNLOAD_EXCLUDE:
        cmd += ["--exclude", pat]
    log(f"  downloading {m['repo']}@{m['revision'][:10]} ({m.get('model_blob_gb')} GB) ...")
    if m.get("gated") and not (os.environ.get("HF_TOKEN") or os.environ.get(
            "HUGGING_FACE_HUB_TOKEN")):
        log(f"  {m['repo']} is gated: accept its license on huggingface.co and set HF_TOKEN")
    # hf download writes where the harness reads: models/<machine>/ unless HF_HUB_CACHE is set
    # (root: another machine's folder, for bench/download_models.py --machine).
    root = root or hf.cache_root()
    env = {**os.environ, "HF_HUB_CACHE": str(root)}
    t0 = time.time()
    with tempfile.TemporaryFile(mode="w+") as err:
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=err, text=True, env=env)
        while True:
            try:
                p.wait(timeout=PROGRESS_EVERY_S)
                break
            except subprocess.TimeoutExpired:
                log("  " + progress(m, t0, root))
        if p.returncode != 0:
            err.seek(0)
            raise SystemExit(f"download of {m['repo']} failed: {err.read().strip()[-600:]}")
    log("  " + progress(m, t0, root).replace("downloading", "downloaded"))


def progress(m, t0, root=None):
    """One line: how many of the pinned weight files are on disk, in GB, and the rate."""
    files = m.get("weight_files") or {}
    snap = hf.snapshot_dir(m["repo"], m["revision"], root)
    have = [n for n in files if (snap / n).exists()]
    gb = sum(files[n]["size"] or 0 for n in have) / 1e9
    total = sum(f["size"] or 0 for f in files.values()) / 1e9 or m.get("model_blob_gb") or 0
    mins = (time.time() - t0) / 60
    rate = f", {gb * 1e3 / max(time.time() - t0, 1):.0f} MB/s" if gb else ""
    return (f"downloading {m['repo']}: {len(have)}/{len(files)} weight files, "
            f"{gb:.1f}/{total:.1f} GB after {mins:.0f} min{rate}")


def verify(m, rehash=True):
    """Compare the cached snapshot with the pins. Returns a record; record['ok'] is the verdict."""
    files = m.get("weight_files") or {}
    rec = {"key": m["key"], "tag": m["tag"], "repo": m["repo"], "revision": m["revision"],
           "pinned_weight_files": len(files)}
    if not files:
        rec.update(ok=False, checks={"pinned_checksums": False},
                   note="no weight_files in the config: run fill_digests.py")
        return rec
    v = hf.verify_cache(m["repo"], m["revision"], files, rehash=rehash)
    rec.update(ok=v["ok"], checks=v["checks"], missing=v["missing"],
               size_bytes=v["size_bytes"], snapshot=v["snapshot"], rehashed=rehash)
    return rec


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--models", nargs="+", help="model keys (default: all)")
    ap.add_argument("--out", default="models.json")
    ap.add_argument("--no-pull", action="store_true", help="verify only")
    ap.add_argument("--no-rehash", action="store_true", help="check names and sizes only")
    a = ap.parse_args()
    cfg = common.load_json(a.config)
    recs = []
    for m in cfg["models"]:
        if a.models and m["key"] not in a.models:
            continue
        if not a.no_pull:
            pull(m)
        rec = verify(m, rehash=not a.no_rehash)
        print(f"  {m['repo']}: {'VERIFIED' if rec['ok'] else 'MISMATCH'} "
              f"({rec.get('size_bytes', 0) / 1e9:.2f} GB, missing {rec.get('missing')})")
        recs.append(rec)
    common.write_json(a.out, recs)
    sys.exit(0 if all(r["ok"] for r in recs) else 1)


if __name__ == "__main__":
    main()
