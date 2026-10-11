#!/usr/bin/env python
"""Study-construction gate -- the house standard, enforced in CI (not just documented).

Every study under ``studies/`` must be built the same way (see ``METHODOLOGY.md``). This
script turns the parts that were silently skippable -- and *were* found broken in an audit --
into a hard check, so they cannot regress:

  1. **Notebooks ship executed.** A published study has the two narrative notebooks; each
     must contain at least one code cell, every code cell must have been run (a non-null
     ``execution_count``), and at least one code cell must carry non-empty outputs -- a
     rebuilt-but-not-re-run skeleton (no figures, no numbers) is a broken deliverable and is
     a FAIL. (This is exactly how studies 04 and 14 shipped dead before the gate existed.)
  2. **Notebooks ship clean.** No code cell may carry an ``error`` output: a notebook built
     with ``nbconvert --execute --allow-errors`` that swallowed a traceback is a FAIL.
  3. **Tests run in CI.** If a study has a ``tests/test_*.py`` suite, pytest must *discover*
     it: CI runs ``pytest`` over the configured ``testpaths``, so ``pyproject.toml`` must list
     ``studies`` there and exclude nothing (no ``--ignore``/``--deselect`` in ``addopts``, no
     ``norecursedirs`` pattern that hides a study or its ``tests/`` dir, no ``collect_ignore``
     in a conftest). With ``--collected FILE`` (a ``pytest --collect-only -q`` listing or a
     JUnit XML report) the gate also checks that every study with a suite really had at
     least one test collected -- the proof that discovery worked, not just that it should.
     (This used to mean "has a step in ``.github/workflows/tests.yml``"; 673 hand-written
     steps were replaced by discovery, so a new suite can no longer be forgotten.)
  4. **The front card exists.** A published study must carry a ``README.md`` with verdict
     badges: the ledger advertises two stamps on the study's behalf, and with no front card
     nothing in the study backs them up. 17 studies were published without one while this
     gate reported them complete.
  5. **docs/ is not empty.** A finished study carries a literature map at
     ``docs/references.md`` with real content (more than 5 non-blank lines).

A study that publishes a working paper instead of a results doc is exempt from the
``docs/results*.md`` recommendation: a ``paper/`` directory at the study root stands in for
the fingerprinted reproducible-run doc (study 01 is the canonical example).

Which studies are gated? A study is "published" -- and therefore held to the standard -- the
moment it is linked in the **studies table in ``docs/REFERENCE.md``**. That table used to live
in the root README; it moved, and the root README is now a front door that links to it rather
than carrying it. Add a newly-published study as a row in ``docs/REFERENCE.md`` -- appending it
to the root README instead leaves an orphan table row stranded on the landing page.

That is the desk's own done-signal: you list a study in the ledger when it's ready. A study
that is not yet in the table is treated as work-in-progress and skipped, so day-to-day
scaffolding commits don't turn CI red. The discipline lands exactly when you publish.

Recommended-but-not-blocking items (a results doc, requirements files) print as warnings.

Exit code 0 if every published study passes the hard checks; 1 otherwise. Pure stdlib
(``tomllib``; on Python 3.10 the ``tomli`` backport that pytest itself depends on).

    python tools/check_study_completeness.py
    python tools/check_study_completeness.py --collected junit.xml
    python tools/check_study_completeness.py --collected ids.txt --only 1015-analogue-overlay
"""

from __future__ import annotations

import argparse
import fnmatch
import json
import os
import re
import sys
import xml.etree.ElementTree as ET

try:
    import tomllib
except ImportError:  # Python 3.10
    import tomli as tomllib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDIES = os.path.join(ROOT, "studies")
PYPROJECT = os.path.join(ROOT, "pyproject.toml")
README = os.path.join(ROOT, "README.md")
REFERENCE = os.path.join(ROOT, "docs", "REFERENCE.md")
# The studies table used to live in the root README and now lives in docs/REFERENCE.md.
# Read both: a gate that silently stops finding any study still reports "passed", so
# looking in only one place turns this check into a no-op the day the table moves.
INDEX_FILES = (README, REFERENCE)


def published_studies() -> set[str]:
    """Study dir names linked from the desk's index — its 'this is ready' signal."""
    out: set[str] = set()
    for path in INDEX_FILES:
        if not os.path.exists(path):
            continue
        text = open(path, encoding="utf-8").read()
        out |= set(re.findall(r"studies/(\d+-[a-z0-9-]+)/", text))
    return out


def _notebook_problems(path: str) -> list[str] | None:
    """Problems with one notebook ([] if clean); None if the notebook is absent.

    Hard requirements: at least one code cell; every code cell executed (non-null
    ``execution_count``); at least one code cell with non-empty outputs; and no cell
    carrying an ``error`` output (catches ``nbconvert --allow-errors`` artefacts).
    """
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        nb = json.load(fh)
    code = [c for c in nb.get("cells", []) if c.get("cell_type") == "code"]
    problems: list[str] = []
    if not code:
        problems.append("has no code cells")
        return problems
    if not all(c.get("execution_count") is not None for c in code):
        problems.append("has unexecuted code cells "
                        "(run: jupyter nbconvert --to notebook --execute --inplace)")
    if not any(c.get("outputs") for c in code):
        problems.append("has no code cell with outputs -- no figures, no numbers")
    n_err = sum(1 for c in code
                for out in (c.get("outputs") or [])
                if out.get("output_type") == "error")
    if n_err:
        problems.append(f"carries {n_err} error output(s) -- a traceback shipped "
                        "(--allow-errors is not an acceptable build)")
    return problems


def pytest_config() -> dict:
    """``[tool.pytest.ini_options]`` from pyproject.toml ({} if absent)."""
    if not os.path.exists(PYPROJECT):
        return {}
    with open(PYPROJECT, "rb") as fh:
        return tomllib.load(fh).get("tool", {}).get("pytest", {}).get("ini_options", {})


# addopts flags that silently shrink what a plain `pytest` run collects.
_EXCLUDING_OPTS = ("--ignore", "--ignore-glob", "--deselect", "-k", "-m", "--co",
                   "--collect-only", "--lf", "--last-failed", "--sw", "--stepwise")


def discovery_problems(cfg: dict) -> list[str]:
    """Repo-wide reasons a study suite could be missed by a plain ``pytest`` run."""
    problems: list[str] = []
    testpaths = cfg.get("testpaths", [])
    if isinstance(testpaths, str):
        testpaths = testpaths.split()
    if "studies" not in [t.rstrip("/") for t in testpaths]:
        problems.append("pyproject.toml [tool.pytest.ini_options] testpaths does not include "
                        "'studies' -- study suites would not be discovered")
    addopts = cfg.get("addopts", "")
    tokens = addopts.split() if isinstance(addopts, str) else list(addopts)
    for tok in tokens:
        if tok.split("=")[0] in _EXCLUDING_OPTS:
            problems.append(f"pyproject.toml addopts carries {tok!r}, which excludes tests "
                            "from discovery")
    if "--import-mode=importlib" not in tokens:
        problems.append("pyproject.toml addopts lost --import-mode=importlib; the study "
                        "suites share module names (tests/test_strategy.py) and only that "
                        "mode collects them side by side")
    for conftest in (os.path.join(ROOT, "conftest.py"), os.path.join(STUDIES, "conftest.py")):
        if os.path.exists(conftest):
            text = open(conftest, encoding="utf-8").read()
            if re.search(r"^\s*collect_ignore(_glob)?\s*=", text, re.M):
                problems.append(f"{os.path.relpath(conftest, ROOT)} sets collect_ignore")
    return problems


def _hidden_by_norecurse(name: str, cfg: dict) -> str | None:
    """The norecursedirs pattern hiding ``studies/<name>/tests``, if any."""
    patterns = cfg.get("norecursedirs", [])
    if isinstance(patterns, str):
        patterns = patterns.split()
    for part in (name, "tests"):
        for pat in patterns:
            if fnmatch.fnmatch(part, pat):
                return pat
    return None


def collected_studies(path: str) -> set[str]:
    """Study dir names with at least one collected test in a node-id listing or JUnit XML."""
    found: set[str] = set()
    if path.endswith(".xml"):
        # A collection error is reported with an empty classname, so it never counts.
        for case in ET.parse(path).getroot().iter("testcase"):
            ref = case.get("file") or case.get("classname", "").replace(".", "/")
            m = re.match(r"studies/(\d+-[a-z0-9-]+)/", ref)
            if m:
                found.add(m.group(1))
    else:
        with open(path, encoding="utf-8") as fh:
            for ln in fh:
                m = re.match(r"studies/(\d+-[a-z0-9-]+)/tests/\S+::", ln.strip())
                if m:
                    found.add(m.group(1))
    return found


def _references_useful_lines(path: str) -> int:
    """Number of non-blank lines in docs/references.md (0 if absent)."""
    if not os.path.exists(path):
        return 0
    with open(path, encoding="utf-8") as fh:
        return sum(1 for ln in fh if ln.strip())


def has_test_suite(name: str) -> bool:
    tests_dir = os.path.join(STUDIES, name, "tests")
    return os.path.isdir(tests_dir) and any(
        f.startswith("test_") and f.endswith(".py") for f in os.listdir(tests_dir))


def check_study(name: str, cfg: dict,
                collected: set[str] | None = None) -> tuple[list[str], list[str]]:
    """Return (hard_failures, warnings) for one published study."""
    p = os.path.join(STUDIES, name)
    fails: list[str] = []
    warns: list[str] = []

    # 1+2) Both notebooks present, executed, with real outputs, and error-free.
    for fname in ("01_for_the_curious.ipynb", "02_for_the_quants.ipynb"):
        problems = _notebook_problems(os.path.join(p, "notebooks", fname))
        if problems is None:
            fails.append(f"missing notebook {fname}")
        else:
            fails.extend(f"notebook {fname} {prob}" for prob in problems)

    # 3) The front card. The house Definition of Done opens with it, and the ledger
    #    publishes a verdict on every study's behalf -- so a study with no README is one
    #    whose advertised stamps no file in the study actually backs up. This check was
    #    absent until 17 published studies turned out to have no front card at all while
    #    this gate reported them complete.
    readme = os.path.join(p, "README.md")
    if not os.path.exists(readme):
        fails.append("missing README.md (the front card the ledger's verdict points at)")
    elif "img.shields.io/badge/" not in open(readme, encoding="utf-8").read():
        fails.append("README.md carries no verdict badges -- nothing backs its ledger row")

    # 4) A test-suite, if present, must be discovered by the plain `pytest` run CI makes
    #    (and, given a collection report, must actually have been collected).
    if has_test_suite(name):
        pat = _hidden_by_norecurse(name, cfg)
        if pat:
            fails.append(f"its test-suite is hidden from discovery by norecursedirs {pat!r}")
        if collected is not None and name not in collected:
            fails.append("has a test-suite but no test of it was collected "
                         f"(run: pytest --collect-only -q studies/{name}/tests)")

    # 5) A literature map must exist and carry real content (no empty docs/).
    refs = os.path.join(p, "docs", "references.md")
    n_useful = _references_useful_lines(refs)
    if n_useful == 0 and not os.path.exists(refs):
        fails.append("missing docs/references.md (the literature map)")
    elif n_useful <= 5:
        fails.append(f"docs/references.md is a stub ({n_useful} non-blank line(s); "
                     "a literature map needs more than 5)")

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


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--collected", metavar="FILE",
                    help="pytest --collect-only -q listing or JUnit XML of the run: every "
                         "study with a suite must have >= 1 collected test in it")
    ap.add_argument("--only", nargs="*", metavar="STUDY",
                    help="restrict the --collected check to these study dirs (a CI run "
                         "that tested only the changed studies)")
    args = ap.parse_args(argv)

    cfg = pytest_config()
    published = published_studies()
    studies = sorted(d for d in os.listdir(STUDIES)
                     if re.match(r"\d+-", d) and os.path.isdir(os.path.join(STUDIES, d)))
    collected = collected_studies(args.collected) if args.collected else None
    only = set(args.only) if args.only is not None else None

    failed = False
    for prob in discovery_problems(cfg):
        print(f"  X  discovery: {prob}")
        failed = True

    n_ok = n_wip = n_fail = 0
    for s in studies:
        if s not in published:
            print(f"  ~  {s:24s} not in README yet -- WIP, skipped")
            n_wip += 1
            continue
        in_scope = only is None or s in only
        fails, warns = check_study(s, cfg, collected if in_scope else None)
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
    if collected is not None:
        scope = [s for s in studies if has_test_suite(s) and (only is None or s in only)]
        print(f"Collection check: {sum(s in collected for s in scope)}/{len(scope)} "
              f"studies with a suite had tests collected ({args.collected}).")
    if failed:
        print("\nStudy-construction gate FAILED. Fix the items above (see METHODOLOGY.md, "
              "section 'Definition of Done'). Studies not yet listed in the README are exempt.")
        return 1
    print("Study-construction gate passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
