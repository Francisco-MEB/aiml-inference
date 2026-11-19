#!/usr/bin/env python3
"""Looks up each model's pinned revision on the Hugging Face Hub and writes the sha256 and size
of every weight file into the config, so every run can prove which bytes it served. Fails (and
changes nothing) if a repo or revision does not exist.

  python3 harness/fill_digests.py configs/<study>.json

Also records the weights size and what format the Hub reports, so you can confirm the build
is the quantization the study claims."""
import json
import sys

import hf


def main():
    path = sys.argv[1]
    with open(path) as f:
        cfg = json.load(f)
    found, missing = {}, []
    for m in cfg["models"]:
        try:
            d = hf.describe(m["repo"], m["revision"])
            if d["revision"] != m["revision"]:
                missing.append(f"{m['repo']}: revision {m['revision']} resolves to "
                               f"{d['revision']} (pin a full commit SHA)")
            found[m["key"]] = d
        except Exception as e:  # noqa: BLE001
            missing.append(f"{m['repo']}@{m.get('revision')}: {e}")
    if missing:
        raise SystemExit("not found on the Hub (nothing written):\n  " + "\n  ".join(missing))
    for m in cfg["models"]:
        d = found[m["key"]]
        m["weight_files"] = d["weight_files"]
        m["model_blob_gb"] = d["weights_gb"]
        m["weight_format"] = d["format"]
        print(f"{m['repo']}@{m['revision'][:10]}: {len(d['weight_files'])} weight files, "
              f"{d['weights_gb']} GB, Hub reports {d['format']} "
              f"(study quantization: {m.get('quantization')})")
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
        f.write("\n")
    print(f"updated {path}")


if __name__ == "__main__":
    main()
