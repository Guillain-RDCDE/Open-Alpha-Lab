"""Shared fixtures for Study 88 (Dogs-of-the-Dow) -- deterministic, offline."""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from dogs_of_the_dow import data  # noqa: E402


@pytest.fixture
def planted_panel():
    """A synthetic panel where high yield genuinely forecasts return -- the timer must win."""
    return data.synthetic_panel(planted=True, seed=88)


@pytest.fixture
def null_panel():
    """A synthetic panel where yield is noise -- the timer must NOT find an edge."""
    return data.synthetic_panel(planted=False, seed=88)
