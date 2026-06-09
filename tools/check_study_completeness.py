#!/usr/bin/env python
"""Study-construction gate -- the house standard, enforced in CI (not just documented).

Every study under ``studies/`` must be built the same way (see ``METHODOLOGY.md``). This
script turns the parts that were silently skippable -- and *were* found broken in an audit --
into a hard check, so they cannot regress:

  1. **Notebooks ship executed.** A published study has the two narrative notebooks, and every
     code cell must carry outputs (a non-null ``execution_count``). A rebuilt-but-not-re-run
     skeleton -- no figures, no numbers -- is a broken deliverable and is a FAIL. (This is
     exactly how studies 04 and 14 shipped dead before the gate existed.)
  2. **Tests run in CI.** If a study has a ``tests/test_*.py`` suite, that suite must be wired
     into ``.github/workflows/tests.yml`` -- otherwise regressions in it go undetected.
  3. **docs/ is not empty.** A finished study carries a literature map at ``docs/references.md``.

Which studies are gated? A study is "published" -- and therefore held to the standard -- the
moment it is linked in the **root ``README.md`` studies table**. That is the desk's own
done-signal: you list a study on the landing page when it's ready. A study that is not yet in
the table is treated as work-in-progress and skipped, so day-to-day scaffolding commits don't
turn CI red. The discipline lands exactly when you publish.

Recommended-but-not-blocking items (a results doc, requirements files) print as warnings.

Exit code 0 if every published study passes the hard checks; 1 otherwise. Pure stdlib.

    python tools/check_study_completeness.py
"""

from __future__ import annotations

import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDIES = os.path.join(ROOT, "studies")
CI_FILE = os.path.join(ROOT, ".github", "workflows", "tests.yml")
README = os.path.join(ROOT, "README.md")


def published_studies() -> set[str]:
    """Study dir names linked in the root README (the desk's 'this is ready' signal)."""
    text = open(README, encoding="utf-8").read() if os.path.exists(README) else ""
    return set(re.findall(r"studies/(\d+-[a-z0-9-]+)/", text))


def _notebook_executed(path: str) -> bool | None:
    """True/False whether all code cells ran; None if the notebook is absent."""
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        nb = json.load(fh)
    code = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    if not code:
        return True
    return all(c.get("execution_count") is not None for c in code)


def check_study(name: str, ci_text: str) -> tuple[list[str], list[str]]:
    """Return (hard_failures, warnings) for one published study."""
    p = os.path.join(STUDIES, name)
    fails: list[str] = []
    warns: list[str] = []

    # 1) Both notebooks present AND executed (the broken-skeleton guard).
    for fname in ("01_for_the_curious.ipynb", "02_for_the_quants.ipynb"):
        st = _notebook_executed(os.path.join(p, "notebooks", fname))
        if st is None:
            fails.append(f"missing notebook {fname}")
        elif not st:
            fails.append(f"notebook {fname} has unexecuted code cells "
                         "(run: jupyter nbconvert --to notebook --execute --inplace)")

    # 2) A test-suite, if present, must be wired into CI.
    tests_dir = os.path.join(p, "tests")
    has_tests = os.path.isdir(tests_dir) and any(
        f.startswith("test_") and f.endswith(".py") for f in os.listdir(tests_dir)
    )
    if has_tests and f"studies/{name}/tests" not in ci_text:
        fails.append("has a test-suite not run in .github/workflows/tests.yml "
                     f"(add: pytest -q studies/{name}/tests)")

    # 3) A literature map must exist (no empty docs/).
    if not os.path.exists(os.path.join(p, "docs", "references.md")):
        fails.append("missing docs/references.md (the literature map)")

    # --- recommended, non-blocking ---
    docs = os.path.join(p, "docs")
    has_results = os.path.isdir(docs) and any(
        f.startswith("results") and f.endswith(".md") for f in os.listdir(docs)
    )
    if not has_results and not os.path.isdir(os.path.join(p, "paper")):
        warns.append("no docs/results*.md (a fingerprinted reproducible-run doc is the standard)")
    if not os.path.exists(os.path.join(p, "requirements.txt")):
        warns.append("no requirements.txt")

    return fails, warns


def main() -> int:
    ci_text = open(CI_FILE, encoding="utf-8").read() if os.path.exists(CI_FILE) else ""
    published = published_studies()
    studies = sorted(d for d in os.listdir(STUDIES)
                     if re.match(r"\d+-", d) and os.path.isdir(os.path.join(STUDIES, d)))

    n_ok = n_wip = n_fail = 0
    failed = False
    for s in studies:
        if s not in published:
            print(f"  ~  {s:24s} not in README yet -- WIP, skipped")
            n_wip += 1
            continue
        fails, warns = check_study(s, ci_text)
        if fails:
            print(f"  X  {s:24s} {len(fails)} problem(s)")
            for f in fails:
                print(f"       FAIL: {f}")
            failed = True
            n_fail += 1
        else:
            print(f"  ok {s:24s} complete")
            n_ok += 1
        for w in warns:
            print(f"       warn: {w}")

    print(f"\n{n_ok} complete | {n_wip} WIP (skipped) | {n_fail} with hard failures.")
    if failed:
        print("\nStudy-construction gate FAILED. Fix the items above (see METHODOLOGY.md, "
              "section 'Definition of Done'). Studies not yet listed in the README are exempt.")
        return 1
    print("Study-construction gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
