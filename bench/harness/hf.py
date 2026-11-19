"""Hugging Face model identity: the exact weights a run used, without any server's help.
Standard library only. A model is pinned by repo + revision (commit SHA), and its weights by the
sha256 of every root-level *.safetensors file, which the Hugging Face cache uses as blob names.

Only root-level files count: some repos also ship an `original/` or `metal/` copy that no
server loads, and downloads skip every subfolder (DOWNLOAD_EXCLUDE)."""
import json
import os
import pathlib
import urllib.parse
import urllib.request

import common

API = "https://huggingface.co/api/models"
DOWNLOAD_EXCLUDE = ["*/*"]          # anything in a subfolder (original/, metal/, ...)


def _get(url, timeout=60):
    headers = {"User-Agent": "aiml-bench"}
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        headers["Authorization"] = "Bearer " + token
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers),
                                timeout=timeout) as r:
        return json.loads(r.read())


def model_info(repo, revision="main"):
    """The Hub's record of repo at revision, with file sizes and LFS checksums."""
    rev = urllib.parse.quote(revision or "main", safe="")
    return _get(f"{API}/{repo}/revision/{rev}?blobs=true")


def weight_files(info):
    """{filename: {"sha256", "size"}} for the root-level *.safetensors files a server loads.
    Mistral repos ship the same weights twice (Hugging Face `model-*.safetensors` and Mistral's
    `consolidated*.safetensors`); servers load the Hugging Face set, so only that one counts."""
    out = {}
    for s in info.get("siblings", []):
        name = s["rfilename"]
        if "/" in name or not name.endswith(".safetensors"):
            continue
        lfs = s.get("lfs") or {}
        out[name] = {"sha256": lfs.get("sha256", ""), "size": lfs.get("size") or s.get("size")}
    if any(not n.startswith("consolidated") for n in out):
        out = {n: v for n, v in out.items() if not n.startswith("consolidated")}
    return out


def download_excludes(info):
    """Glob patterns `hf download --exclude` needs so only what a server loads is fetched."""
    names = [s["rfilename"] for s in info.get("siblings", [])]
    ex = list(DOWNLOAD_EXCLUDE)
    if any(n.startswith("consolidated") for n in names) and \
            any(n.startswith("model") and n.endswith(".safetensors") for n in names):
        ex.append("consolidated*")
    return ex


def weight_format(info):
    """Short description of how the weights are stored, from the Hub's config summary and
    safetensors dtypes. 'bits' is the stored bits per weight of the quantized layers."""
    qc = (info.get("config") or {}).get("quantization_config") or {}
    dtypes = sorted(((info.get("safetensors") or {}).get("parameters") or {}))
    method = qc.get("quant_method", "")
    bits = qc.get("bits")
    for g in (qc.get("config_groups") or {}).values():
        bits = bits or (g.get("weights") or {}).get("num_bits")
    if method == "mxfp4":
        bits = 4
    mode = qc.get("mode") or ("affine" if "bits" in qc and not method else "")
    if not qc:
        float_types = [d for d in dtypes if d in ("BF16", "F16", "F32")]
        fmt = (float_types[0].lower() if float_types else "unknown")
        if "F8_E4M3" in dtypes:
            fmt = "fp8"
        return {"format": fmt, "quant_method": "", "bits": 8 if fmt == "fp8" else 16,
                "dtypes": dtypes}
    return {"format": method or f"mlx-{mode}", "quant_method": method or "mlx", "bits": bits,
            "mode": mode, "group_size": qc.get("group_size"), "dtypes": dtypes}


def describe(repo, revision="main"):
    """Everything the catalog and a study config need to pin one model build."""
    info = model_info(repo, revision)
    files = weight_files(info)
    archs = (info.get("config") or {}).get("architectures") or []
    return {"repo": repo, "revision": info["sha"], "gated": info.get("gated") or False,
            "weights_gb": round(sum(f["size"] or 0 for f in files.values()) / 1e9, 2),
            "weight_files": files, "architectures": archs,
            # Vision-language checkpoints: serve the text model only (saves memory).
            "multimodal": any(a.endswith("ForConditionalGeneration") for a in archs),
            "download_exclude": download_excludes(info), **weight_format(info)}


# ---- local cache -------------------------------------------------------------------------

def cache_root():
    hub = os.environ.get("HF_HUB_CACHE")
    if hub:
        return pathlib.Path(hub)
    home = os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface")
    return pathlib.Path(home) / "hub"


def snapshot_dir(repo, revision, root=None):
    return (pathlib.Path(root or cache_root()) / ("models--" + repo.replace("/", "--"))
            / "snapshots" / revision)


def verify_cache(repo, revision, files, root=None, rehash=False):
    """Check the downloaded snapshot against the pinned checksums. Hugging Face stores each
    LFS file as blobs/<sha256>, linked from snapshots/<revision>/<name>, so a file whose link
    points at the right blob name (with the right size) holds the pinned bytes; rehash=True
    also recomputes each sha256. Returns {"ok", "checks", "missing", "size_bytes"}."""
    snap = snapshot_dir(repo, revision, root)
    checks, missing, total = {}, [], 0
    for name, want in files.items():
        p = snap / name
        if not p.exists():
            missing.append(name)
            checks[name] = False
            continue
        # The snapshot entry links to the repo's blobs/<sha256>; newer huggingface_hub
        # versions link that blob on into a shared store under another name, so the sha256
        # is the name one link deep, and the bytes are at the end of the chain.
        blob_name = pathlib.Path(os.readlink(p)).name if p.is_symlink() else ""
        data = pathlib.Path(os.path.realpath(p))
        size = data.stat().st_size
        total += size
        ok = size == want["size"] and (blob_name == want["sha256"] or not want["sha256"])
        if ok and rehash:
            ok = common.sha256_file(data) == want["sha256"]
        checks[name] = ok
    return {"ok": bool(files) and all(checks.values()), "checks": checks, "missing": missing,
            "size_bytes": total, "snapshot": str(snap)}
