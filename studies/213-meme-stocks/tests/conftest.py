"""Shared fixtures for Study 213 (Meme Stocks) — deterministic, offline."""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from meme_stocks import data  # noqa: E402


@pytest.fixture
def planted_panel():
    """Synthetic panel where the meme basket has a genuine planted premium — the engine must detect it."""
    return data.synthetic_panel(premium=0.30, seed=213)


@pytest.fixture
def null_panel():
    """Synthetic panel where the basket has NO premium — the engine must NOT find an edge."""
    return data.synthetic_panel(premium=0.0, seed=213)
