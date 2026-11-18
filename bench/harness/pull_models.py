#!/usr/bin/env python3
"""Pulls each configured model through the Ollama API and verifies it against the pinned
digests. Writes models.json (what was actually on disk) for env.json."""
import argparse
import json
import os
import pathlib
import sys
import urllib.request

import common


def pull(native_url, tag):
    req = urllib.request.Request(native_url + "/api/pull",
                                 data=json.dumps({"model": tag, "stream": True}).encode(),
                                 headers={"Content-Type": "application/json"})
    last = ""
    with urllib.request.urlopen(req, timeout=24 * 3600) as r:
        for line in r:
            msg = json.loads(line)
            if msg.get("error"):
                raise SystemExit(f"pull {tag} failed: {msg['error']}")
            status = msg.get("status", "")
            if status != last and not status.startswith("pulling "):
                print(f"  {tag}: {status}")
            last = status


def verify(cfg, m, hash_blob=True):
    """Compare what the server has with the pins. Returns a record; record['ok'] is the verdict."""
    native = cfg["backend"]["native_url"]
    rec = {"key": m["key"], "tag": m["tag"], "pinned_manifest_sha256": m["manifest_sha256"],
           "pinned_model_blob_sha256": m["model_blob_sha256"], "checks": {}}
    tags = {t["name"]: t for t in common.http_json(native + "/api/tags").get("models", [])}
    t = tags.get(m["tag"])
    if not t:
        rec["ok"] = False
        rec["checks"]["present"] = False
        return rec
    rec["ollama_digest"] = t.get("digest")
    rec["size_bytes"] = t.get("size")
    rec["details"] = t.get("details")
    rec["checks"]["manifest_digest"] = t.get("digest") == m["manifest_sha256"]
    # If we can read the model store (native mode), check the weights file itself.
    name, _, tagpart = m["tag"].partition(":")
    store = pathlib.Path(os.path.expanduser(cfg["server"]["models_dir"]))
    mf = store / "manifests/registry.ollama.ai/library" / name / tagpart
    if mf.is_file():
        man = json.loads(mf.read_text())
        layers = {layer["mediaType"].rsplit(".", 1)[-1]: layer["digest"] for layer in man["layers"]}
        rec["layers"] = layers
        blob = layers.get("model", "")
        rec["checks"]["model_layer_digest"] = blob == "sha256:" + m["model_blob_sha256"]
        bf = store / "blobs" / blob.replace(":", "-")
        if hash_blob and bf.is_file():
            rec["model_blob_sha256_on_disk"] = common.sha256_file(bf)
            rec["checks"]["model_blob_bytes"] = \
                rec["model_blob_sha256_on_disk"] == m["model_blob_sha256"]
    else:
        rec["checks"]["store_readable"] = "no (digest from /api/tags only)"
    rec["ok"] = all(v for v in rec["checks"].values() if isinstance(v, bool))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(common.BENCH / "configs/TEMPLATE.json"))
    ap.add_argument("--models", nargs="+", help="model keys (default: all non-optional)")
    ap.add_argument("--out", default="models.json")
    ap.add_argument("--no-pull", action="store_true", help="verify only")
    a = ap.parse_args()
    cfg = common.load_json(a.config)
    keys = a.models or [m["key"] for m in cfg["models"] if not m.get("optional")]
    recs = []
    for m in cfg["models"]:
        if m["key"] not in keys:
            continue
        if not a.no_pull:
            print(f"pulling {m['tag']} ...")
            pull(cfg["backend"]["native_url"], m["tag"])
        rec = verify(cfg, m)
        print(f"  {m['tag']}: {'VERIFIED' if rec['ok'] else 'MISMATCH'} {rec['checks']}")
        recs.append(rec)
    common.write_json(a.out, recs)
    sys.exit(0 if all(r["ok"] for r in recs) else 1)


if __name__ == "__main__":
    main()
