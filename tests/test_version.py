"""The package, its metadata and its citation must announce the same version."""

import os
import re

import quantlab

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _read(name):
    with open(os.path.join(ROOT, name), encoding="utf-8") as fh:
        return fh.read()


def test_one_version_everywhere():
    pyproject = re.search(r'^version = "([^"]+)"', _read("pyproject.toml"), re.M).group(1)
    citation = re.search(r"^version: (\S+)", _read("CITATION.cff"), re.M).group(1)
    assert quantlab.__version__ == pyproject == citation
