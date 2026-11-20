#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Download the club benchmark's models ahead of time, so a study doesn't stop to fetch
20-80 GB per model. Needs only uv (https://docs.astral.sh/uv/).

    uv run bench/download_models.py                    # Qwen, Gemma, GLM and Kimi for THIS machine
    uv run bench/download_models.py --dry-run          # only list what would download
    uv run bench/download_models.py --verify           # only check the downloads' checksums
    uv run bench/download_models.py qwen glm           # just these families (or model keys)
    uv run bench/download_models.py --machine studio   # another machine's builds (dell, studio, mini)

It fetches what the benchmark runs on that machine (the catalog's build at its pinned commit)
into its folder: models/dell, models/mac/studio or models/mac/mini (HF_HUB_CACHE overrides).
Re-running resumes. Every run ends by checking each file against the sha256 pinned on the Hub.
"""
import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "harness"))
import catalog  # noqa: E402
import hf  # noqa: E402
import pull_models  # noqa: E402

# The families in the single-node lineup (the catalog has more; name them to fetch those).
DEFAULT_FAMILIES = ["qwen", "gemma", "glm", "kimi"]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("names", nargs="*", default=DEFAULT_FAMILIES,
                    help="families or model keys (default: %(default)s)")
    ap.add_argument("--dry-run", action="store_true", help="list the downloads, fetch nothing")
    ap.add_argument("--verify", action="store_true", help="only check what is already here")
    ap.add_argument("--machine", default="auto", help="auto | dell | studio | mini | mac:<GiB>")
    args = ap.parse_args()
    sys.stdout.reconfigure(line_buffering=True)   # so `tail -f` on a log shows progress live

    # Plan each name the way the benchmark will on that machine, one entry per repo.
    machine = catalog.parse_machine(args.machine)
    print(f"machine: {machine['hardware']} ({machine['engine']})")
    models = {}
    for name in args.names:
        plan = catalog.plan_study(name, machine)
        print(f"\n{plan['family']}: {plan['build'] or 'nothing fits this machine'} "
              f"({plan.get('publisher', '')})")
        for m in plan["models"]:
            print(f"  {m['repo']:55} {m['model_blob_gb']:6.1f} GB")
            models.setdefault(m["repo"], m)
        for x in plan["left_out"]:
            print(f"  left out: {x['key']}: {x['reason']}")

    # Check the disk can hold what isn't downloaded yet before starting hours of downloads.
    root = hf.cache_root(machine)
    root.mkdir(parents=True, exist_ok=True)
    new = [m for m in models.values()
           if not hf.snapshot_dir(m["repo"], m["revision"], root).exists()]
    need_gb = sum(m["model_blob_gb"] for m in new)
    free_gb = shutil.disk_usage(root).free / 1e9
    print(f"\n{len(models)} models, {len(new)} not downloaded yet ({need_gb:.0f} GB); "
          f"{free_gb:.0f} GB free in {root}")
    if args.dry_run:
        return
    if not args.verify:
        if need_gb > free_gb:
            sys.exit("not enough free disk space")
        # One model at a time at its pinned commit; a failure stops here and a re-run resumes.
        for m in models.values():
            pull_models.pull(m, root=root)

    # The same check the benchmark runs before a study: every weight file's sha256 (this reads
    # each file once, a few minutes for a few hundred GB).
    print("\nchecking checksums ...")
    bad = []
    for m in models.values():
        files = hf.describe(m["repo"], m["revision"])["weight_files"]
        v = hf.verify_cache(m["repo"], m["revision"], files, root=root, rehash=True)
        print(f"  {'OK      ' if v['ok'] else 'MISMATCH'} {m['repo']:55} "
              f"{v['size_bytes'] / 1e9:6.1f} GB" + (" (missing files)" if v["missing"] else ""))
        bad += [] if v["ok"] else [m["repo"]]
    if bad:
        sys.exit(f"{len(bad)} model(s) failed: re-run to fetch missing files; for a mismatch, "
                 f"delete that model's models--<org>--<name> folder in {root}, then re-run")
    print(f"\nall {len(models)} models verified in {root}")


if __name__ == "__main__":
    main()
