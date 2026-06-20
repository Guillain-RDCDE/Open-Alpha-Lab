"""Shared fixtures for Study 339 (Convertible-Bonds) — deterministic, offline."""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from convertible_bonds import data  # noqa: E402


@pytest.fixture
def convex():
    """A planted convex 'convertible' (gamma 0.6) — the harness must recover positive gamma."""
    return data.synthetic_convex(n_days=4000, gamma=0.6, seed=339)


@pytest.fixture
def linear():
    """A linear blend (gamma 0) — no convexity to find: the null."""
    return data.synthetic_convex(n_days=4000, gamma=0.0, seed=339)
