#!/usr/bin/env python
"""Decide what a CI run tests: everything, or only the studies a change touched.

    python tools/ci_plan.py --event push --base <sha> --head <sha>
    python tools/ci_plan.py --event schedule

Full run when any of these holds:

* the run is the nightly schedule or a manual ``workflow_dispatch``;
* the base commit is unknown (first push of a branch, force-push, shallow history);
* any changed path lies outside a study directory ``studies/<id>/`` -- the engine,
  the tools, the workflow, ``pyproject.toml``, the docs, the template... anything that
  can move more than one study.

Otherwise the run covers only the changed studies: ``studies/<id>/tests`` for a study
with a suite, its case in ``tests/test_study_imports.py`` for one without, and its
offline demo if it has one in ``tools/run_smoke_demos.py``. A deleted study contributes
nothing (the repo-wide gates still run).

Writes ``mode``, ``targets``, ``studies`` and ``demos`` to ``$GITHUB_OUTPUT`` when set,
and prints them either way. Pure stdlib.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDY_RE = re.compile(r"^studies/(\d+-[a-z0-9-]+)/")
FULL_EVENTS = {"schedule", "workflow_dispatch"}


def changed_paths(base: str, head: str) -> list[str] | None:
    """Paths changed between two commits, or None if the diff cannot be computed."""
    if not base or set(base) == {"0"}:
        return None
    try:
        out = subprocess.run(["git", "diff", "--name-only", "--no-renames", base, head],
                             cwd=ROOT, capture_output=True, text=True, check=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return [p for p in out.splitlines() if p.strip()]


def has_suite(study: str) -> bool:
    d = os.path.join(ROOT, "studies", study, "tests")
    return os.path.isdir(d) and any(f.startswith("test_") and f.endswith(".py")
                                    for f in os.listdir(d))


def plan(event: str, paths: list[str] | None) -> dict:
    if event in FULL_EVENTS:
        return {"mode": "full", "why": f"{event} run"}
    if paths is None:
        return {"mode": "full", "why": "no usable base commit"}
    outside = [p for p in paths if not STUDY_RE.match(p)]
    if outside:
        return {"mode": "full", "why": f"{len(outside)} path(s) outside studies/<id>/, "
                                       f"e.g. {outside[0]}"}
    studies = sorted({STUDY_RE.match(p).group(1) for p in paths})
    present = [s for s in studies if os.path.isdir(os.path.join(ROOT, "studies", s))]
    targets = []
    for s in present:
        if has_suite(s):
            targets.append(f"studies/{s}/tests")
        else:
            targets.append(f"tests/test_study_imports.py::test_study_smoke[{s}]")
    sys.path.insert(0, os.path.join(ROOT, "tools"))
    from run_smoke_demos import DEMOS  # noqa: E402

    ids = [s.split("-")[0] for s in present]
    demos = [d.split("/")[1].split("-")[0] for d in DEMOS
             if d.split("/")[1].split("-")[0] in ids]
    return {"mode": "changed", "why": f"{len(present)} study dir(s) changed",
            "targets": targets, "studies": present, "demos": demos}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", default=os.environ.get("GITHUB_EVENT_NAME", "push"))
    ap.add_argument("--base", default="")
    ap.add_argument("--head", default="HEAD")
    args = ap.parse_args(argv)

    paths = None if args.event in FULL_EVENTS else changed_paths(args.base, args.head)
    p = plan(args.event, paths)
    out = {
        "mode": p["mode"],
        "targets": " ".join(p.get("targets", [])),
        "studies": " ".join(p.get("studies", [])),
        "demos": " ".join(p.get("demos", [])),
    }
    print(f"plan: {p['mode']} ({p['why']})")
    for k, v in out.items():
        print(f"  {k} = {v}")
    gh_out = os.environ.get("GITHUB_OUTPUT")
    if gh_out:
        with open(gh_out, "a", encoding="utf-8") as fh:
            for k, v in out.items():
                fh.write(f"{k}={v}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
