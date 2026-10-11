"""Smoke coverage for every study that ships without a test-suite of its own.

About a third of the bench (the later lots, mostly built from one template) has no
``tests/`` directory, so nothing in CI ever imported their code: a syntax error, a
renamed helper or a dependency the package no longer finds would only surface when a
reader opened the notebook. This file is generated from the tree at collection time --
one case per such study, nothing to keep in sync by hand -- and checks the minimum:

1. the study package and **every** module in it import cleanly;
2. each zero-argument synthetic generator in its ``data`` module (``synthetic_panel``,
   ``synthetic_world``, ...) runs offline and returns something non-empty.

It is a floor, not a test-suite: it says the code loads and the offline world builds,
not that any number is right. A study that grows a real ``tests/`` suite drops out of
this file automatically. Studies that cannot be smoke-tested this way are listed in
``EXPECTED_SKIPS`` with the reason; a stale entry there fails the run.
"""

from __future__ import annotations

import importlib
import inspect
import os
import re
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STUDIES = os.path.join(ROOT, "studies")

# study dir -> why it is not smoke-tested here (each one is checked to still apply)
EXPECTED_SKIPS = {
    "01-overnight-anomaly": "no study package: study 01 is the quantlab engine itself, "
                            "covered by the tests/ suite",
}


def _has_suite(study: str) -> bool:
    d = os.path.join(STUDIES, study, "tests")
    return os.path.isdir(d) and any(f.startswith("test_") and f.endswith(".py")
                                    for f in os.listdir(d))


def _untested_studies() -> list[str]:
    return sorted(
        (d for d in os.listdir(STUDIES)
         if re.match(r"\d+-", d) and os.path.isdir(os.path.join(STUDIES, d))
         and not _has_suite(d)),
        key=lambda d: int(d.split("-")[0]))


def _packages(study: str) -> list[str]:
    p = os.path.join(STUDIES, study)
    return sorted(e for e in os.listdir(p) if os.path.isfile(os.path.join(p, e, "__init__.py")))


def _non_empty(obj) -> bool:
    if isinstance(obj, (pd.DataFrame, pd.Series, pd.Index)):
        return len(obj) > 0
    if isinstance(obj, np.ndarray):
        return obj.size > 0
    if isinstance(obj, (tuple, list)):
        return len(obj) > 0 and _non_empty(obj[0])
    if isinstance(obj, dict):
        return len(obj) > 0
    return obj is not None


UNTESTED = _untested_studies()


@pytest.mark.parametrize("study", UNTESTED)
def test_study_smoke(study):
    if study in EXPECTED_SKIPS:
        pytest.skip(EXPECTED_SKIPS[study])
    pkgs = _packages(study)
    assert pkgs, f"{study}: no importable package (add it to EXPECTED_SKIPS with a reason)"
    path = os.path.join(STUDIES, study)
    sys.path.insert(0, path)
    try:
        ran = 0
        for pk in pkgs:
            importlib.import_module(pk)
            for f in sorted(os.listdir(os.path.join(path, pk))):
                if f.endswith(".py") and f != "__init__.py":
                    importlib.import_module(f"{pk}.{f[:-3]}")
            try:
                data = importlib.import_module(f"{pk}.data")
            except ModuleNotFoundError:
                continue
            for name, fn in sorted(vars(data).items()):
                if not (name.startswith(("synthetic", "make_synthetic"))
                        and inspect.isfunction(fn) and fn.__module__ == data.__name__):
                    continue
                required = [p for p in inspect.signature(fn).parameters.values()
                            if p.default is p.empty
                            and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
                if required:
                    continue
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    out = fn()
                assert _non_empty(out), f"{pk}.data.{name}() returned nothing"
                ran += 1
        assert ran, (f"{study}: no zero-argument synthetic generator in its data module "
                     "(add it to EXPECTED_SKIPS with a reason)")
    finally:
        sys.path.remove(path)


def test_expected_skips_are_current():
    """A skip that no longer applies must be removed, not carried forward."""
    stale = [s for s in EXPECTED_SKIPS if s not in UNTESTED]
    assert not stale, f"EXPECTED_SKIPS lists studies that are gone or now tested: {stale}"
