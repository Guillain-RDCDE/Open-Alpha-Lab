"""Shared fixtures for Study 909 (Preferred Reset Premium) — deterministic, offline.

Two synthetic worlds, one knob (``edge``): a planted regime-contingent reset premium (the
variable sleeve out-carries the fixed sleeve in the high-rate months — the positive control)
and a null world (both sleeves the same asset up to noise). Tests never touch the network."""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from pref_reset import data  # noqa: E402


@pytest.fixture(scope="session")
def planted_world():
    """edge=0.30%/mo + a duration hit on the fixed leg in the high-rate regime (pos control)."""
    return data.synthetic_world(edge=0.0030, dur_hit=0.010, seed=909)


@pytest.fixture(scope="session")
def null_world():
    """edge=0, dur_hit=0 — the two sleeves are the same asset up to noise (the null)."""
    return data.synthetic_world(edge=0.0, dur_hit=0.0, seed=909)
