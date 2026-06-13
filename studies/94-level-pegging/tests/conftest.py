"""Shared fixtures for Study 94 (Level-Pegging) — deterministic, offline."""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from level_pegging import data  # noqa: E402


@pytest.fixture
def broadening_panel():
    """A panel where small names carry the premium — equal-weight must win."""
    return data.synthetic_panel(mode="broadening", seed=94)


@pytest.fixture
def megacap_panel():
    """A panel where a few mega names dominate — cap-weight must win, equal-weight lags."""
    return data.synthetic_panel(mode="megacap", seed=94)
