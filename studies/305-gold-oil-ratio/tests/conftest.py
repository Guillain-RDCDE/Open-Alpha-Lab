"""Shared fixtures — deterministic synthetic gold/oil/equity tapes with a *known* amount of
forward ratio→equity linkage, so tests never touch the network and the only thing the timing
switch can harvest (a lead from the lagged ratio to the next return) is baked in or absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from gold_oil_ratio import data  # noqa: E402


@pytest.fixture
def null():
    """A null tape (pred_r=0) — the switch must be a fair coin vs random timing here."""
    daily, truth = data.synthetic_daily(n_days=4000, pred_r=0.0, seed=305)
    return daily, truth


@pytest.fixture
def planted():
    """A folklore-consistent tape (pred_r<0: high ratio precedes lower returns) — the switch
    should clearly beat both buy-and-hold and random timing here."""
    daily, truth = data.synthetic_daily(n_days=4000, pred_r=-0.01, seed=305)
    return daily, truth
