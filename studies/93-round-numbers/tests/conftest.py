"""Shared fixtures for Study 93 (Round-Numbers) — deterministic, offline."""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from round_numbers import data  # noqa: E402


@pytest.fixture
def null_walk():
    """A constant-drift random walk (magnet 0) — no magnetism; the fade must not bank an edge."""
    return data.synthetic_daily(magnet=0.0, seed=93)


@pytest.fixture
def planted_magnet():
    """A tape with a reflective barrier at each round level (magnet 0.8) — the fade should bank an edge."""
    return data.synthetic_daily(magnet=0.8, seed=93)
