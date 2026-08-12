"""Shared fixtures for Study 844 — Madden-Cover-Curse.

Fully offline: everything runs on the seeded synthetic world and on pure functions. No
network, no real cache required.
"""
from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from madden_curse import data as dt  # noqa: E402


@pytest.fixture(scope="session")
def null_world():
    """The null world — publisher tracks the benchmark, no planted launch drift."""
    return dt.synthetic_world(drift=0.0, seed=844)


@pytest.fixture(scope="session")
def edge_world():
    """A planted +4% launch-week drift the detector must recover."""
    return dt.synthetic_world(drift=0.04, seed=844)
