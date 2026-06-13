"""Shared fixtures for Study 96 (New-Year-Pop) — deterministic, offline."""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from new_year_pop import data  # noqa: E402


@pytest.fixture
def null_tape():
    """No planted seasonal (jan_bump 0) — the harness must NOT find a January effect."""
    return data.synthetic_monthly(n_years=36, jan_bump=0.0, seed=96)


@pytest.fixture
def planted_tape():
    """A planted +4%/January bump in the small leg — the harness must detect it."""
    return data.synthetic_monthly(n_years=36, jan_bump=0.04, seed=96)
