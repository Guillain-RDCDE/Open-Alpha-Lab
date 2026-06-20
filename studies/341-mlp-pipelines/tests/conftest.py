"""Shared fixtures — deterministic synthetic MLP tapes so tests never touch the network. One
world plants the yield trap (high energy beta + fat distribution on an eroding NAV → return of
capital on a leveraged energy bet); one is the null (beta 0, no payout, no drift → flat,
market-neutral, nothing to find)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from mlp_pipelines import data  # noqa: E402


@pytest.fixture
def trap():
    """Yield-trap world: beta 1.10, dist 0.65%/mo, NAV bleed −0.35%/mo (the positive control)."""
    panel, truth = data.synthetic_mlp(beta=1.10, dist=0.0065, nav_drift=-0.0035, seed=341)
    return panel, truth


@pytest.fixture
def null():
    """The null: beta 0, no distribution, no NAV drift → a flat, market-neutral fund."""
    panel, truth = data.synthetic_mlp(beta=0.0, dist=0.0, nav_drift=0.0, idio_vol=0.0, seed=341)
    return panel, truth
