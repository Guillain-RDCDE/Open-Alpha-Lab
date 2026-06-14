"""Shared fixtures for Study 135 (FOMC-Cycle) tests.

All fixtures are deterministic and offline — no network calls. The synthetic
generator (``data.synthetic_cycle``) bakes the even-week premium directly into
the return series so tests can assert the even-vs-odd gap appears when and only
when we plant it.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from fomc_cycle import data  # noqa: E402


@pytest.fixture
def null_world():
    """No even-week premium (pure random walk) — the null in a bottle."""
    frame, truth = data.synthetic_cycle(n_days=2000, even_premium=0.0, seed=135)
    return frame, truth


@pytest.fixture
def signal_world():
    """A planted even-week premium of 0.10% per day — the effect should be visible."""
    frame, truth = data.synthetic_cycle(n_days=2000, even_premium=0.001, seed=135)
    return frame, truth
