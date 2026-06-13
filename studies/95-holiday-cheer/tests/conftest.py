"""Shared fixtures for Study 95 (Holiday-Cheer) - deterministic, offline."""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from holiday_cheer import data  # noqa: E402


@pytest.fixture
def planted_tape():
    """An engineered tape with a real pre-holiday bump - the harness must detect it."""
    return data.synthetic_daily(planted=True, seed=95)


@pytest.fixture
def flat_tape():
    """A flat i.i.d. tape with engineered holidays but no bump - must NOT be detected."""
    return data.synthetic_daily(planted=False, seed=95)
