"""Shared fixtures — deterministic synthetic drawdown-duration panels (no network)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from drawdown_duration import data  # noqa: E402


@pytest.fixture(scope="session")
def edge_world():
    """A planted relation: low-drift names stay underwater and keep sinking, so a high
    time-underwater goes with a *lower* forward return (a negative long-high/short-low
    spread)."""
    return data.synthetic_panel(knob=0.0010, seed=816, n_assets=40, n_days=1500)


@pytest.fixture(scope="session")
def null_world():
    """The null — time-underwater varies across names (path noise) but carries no
    forward information."""
    return data.synthetic_panel(knob=0.0, seed=816, n_assets=40, n_days=1500)
