#!/usr/bin/env python3
"""Looks up each model tag in the Ollama registry and writes its checksums into the config, so
nobody has to copy them by hand. Fails (and changes nothing) if a tag does not exist.

  python3 harness/fill_digests.py configs/<study>.json

Writes per model: manifest_sha256 (what `ollama list` shows as the ID), model_blob_sha256 (the
weights file, checked on disk at every run), and file_type / parameter count as the registry
reports them, so you can confirm the quantization you chose is what the tag really holds."""
import hashlib
import json
import sys
import urllib.error
import urllib.request

REGISTRY = "https://registry.ollama.ai/v2"
ACCEPT = "application/vnd.docker.distribution.manifest.v2+json"


def get(url, accept=None):
    req = urllib.request.Request(url, headers={"Accept": accept} if accept else {})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def lookup(tag):
    name, _, ref = tag.partition(":")
    ref = ref or "latest"
    repo = name if "/" in name else f"library/{name}"
    raw = get(f"{REGISTRY}/{repo}/manifests/{ref}", ACCEPT)
    man = json.loads(raw)
    layers = {la["mediaType"].rsplit(".", 1)[-1]: la for la in man["layers"]}
    cfg = json.loads(get(f"{REGISTRY}/{repo}/blobs/{man['config']['digest']}"))
    return {"manifest_sha256": hashlib.sha256(raw).hexdigest(),
            "model_blob_sha256": layers["model"]["digest"].split(":", 1)[1],
            "model_blob_gb": round(layers["model"]["size"] / 1e9, 2),
            "registry_file_type": cfg.get("file_type", ""),
            "registry_model_type": cfg.get("model_type", "")}


def main():
    path = sys.argv[1]
    with open(path) as f:
        cfg = json.load(f)
    found, missing = {}, []
    for m in cfg["models"]:
        try:
            found[m["key"]] = lookup(m["tag"])
        except (urllib.error.HTTPError, KeyError) as e:
            missing.append(f"{m['tag']}: {e}")
    if missing:
        raise SystemExit("not found in the registry (nothing written):\n  " + "\n  ".join(missing))
    for m in cfg["models"]:
        m.update(found[m["key"]])
        print(f"{m['tag']}: manifest {m['manifest_sha256'][:12]}..., weights "
              f"{m['model_blob_gb']} GB, file type {m['registry_file_type']}, "
              f"{m['registry_model_type']} (your quantization: {m.get('quantization')})")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    print(f"updated {path}")


if __name__ == "__main__":
    main()
