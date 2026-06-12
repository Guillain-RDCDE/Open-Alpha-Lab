#!/usr/bin/env python
"""Build the third-party-verification bundle: every data cache behind the published numbers.

The desk's reproducibility chain (see ``docs/reproducibility.md``) ends in a release
artefact: the ``_cache/*.parquet`` files are gitignored (heavy, and some upstream licences
forbid redistribution in-tree), yet every ``docs/results*.md`` quotes fingerprints computed
from them. This script collects all existing parquet caches -- the root ``_cache/`` and each
study's ``_cache/`` -- into a single ``repro_bundle_<date>.zip`` to attach to a GitHub
Release, so a third party can verify the published fingerprints byte-for-byte instead of
re-fetching (where vendor drift is possible).

The zip carries a ``manifest.json`` listing, for every file: its repo-relative path, size in
bytes and sha256 -- so the bundle itself is checkable without unpacking trust.

Stale snapshots (any path under a ``_stale`` directory) are excluded. Pure stdlib.

    python tools/make_repro_bundle.py --dry-run   # list what would be bundled, write nothing
    python tools/make_repro_bundle.py             # write repro_bundle_<YYYY-MM-DD>.zip + manifest
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def collect_cache_files() -> list[str]:
    """Repo-relative paths (POSIX separators) of every cached parquet, sorted.

    Scope: the root ``_cache/`` tree plus every ``_cache/`` tree under ``studies/``
    (including nested ones such as ``studies/01-*/notebooks/_cache``). ``_stale``
    snapshots are skipped -- they are not behind any published number.
    """
    out: list[str] = []
    roots = [os.path.join(ROOT, "_cache")]
    studies = os.path.join(ROOT, "studies")
    if os.path.isdir(studies):
        roots.append(studies)
    for top in roots:
        if not os.path.isdir(top):
            continue
        for dirpath, dirnames, filenames in os.walk(top):
            dirnames[:] = [d for d in dirnames if d != "_stale"]
            rel_dir = os.path.relpath(dirpath, ROOT).replace(os.sep, "/")
            if "_cache" not in rel_dir.split("/"):
                continue
            for f in filenames:
                if f.endswith(".parquet"):
                    out.append(f"{rel_dir}/{f}")
    return sorted(set(out))


def sha256_of(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_manifest(rel_paths: list[str], hash_files: bool = True) -> list[dict]:
    entries = []
    for rel in rel_paths:
        full = os.path.join(ROOT, rel.replace("/", os.sep))
        entries.append({
            "path": rel,
            "size": os.path.getsize(full),
            "sha256": sha256_of(full) if hash_files else None,
        })
    return entries


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dry-run", action="store_true",
                    help="list the files that would be bundled (no hashing, no zip)")
    ap.add_argument("-o", "--output", default=None,
                    help="output zip path (default: repro_bundle_<YYYY-MM-DD>.zip in the repo root)")
    args = ap.parse_args(argv)

    rel_paths = collect_cache_files()
    if not rel_paths:
        print("No _cache/*.parquet files found -- nothing to bundle.")
        return 1

    total = sum(os.path.getsize(os.path.join(ROOT, p.replace("/", os.sep))) for p in rel_paths)
    out_zip = args.output or os.path.join(
        ROOT, f"repro_bundle_{dt.date.today().isoformat()}.zip")

    if args.dry_run:
        for p in rel_paths:
            size = os.path.getsize(os.path.join(ROOT, p.replace("/", os.sep)))
            print(f"  {size:>12,d}  {p}")
        print(f"\nDry-run: {len(rel_paths)} file(s), {total / 1e6:,.1f} MB "
              f"-> would write {out_zip} (+ manifest.json inside).")
        return 0

    manifest = build_manifest(rel_paths)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, indent=2) + "\n")
        for rel in rel_paths:
            zf.write(os.path.join(ROOT, rel.replace("/", os.sep)), arcname=rel)
    print(f"Wrote {out_zip}: {len(rel_paths)} file(s), {total / 1e6:,.1f} MB raw "
          "(manifest.json inside). Attach it to the GitHub Release.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
