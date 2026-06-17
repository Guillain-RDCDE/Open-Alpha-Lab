"""Shared fixtures -- deterministic synthetic (mvrv, price) paths with a *known*
contrarian MVRV->price link, so tests never touch the network and the only thing the
predictive regression can recover is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from mvrv_ratio import data  # noqa: E402


@pytest.fixture
def null_series():
    """No MVRV->price contrarian link (beta=0). Predictive regression should read ~zero."""
    df, truth = data.synthetic_series(beta=0.0, seed=293)
    return df, truth


@pytest.fixture
def live_series():
    """A planted contrarian MVRV->price link (beta=0.50). Regression should recover a
    significantly *negative* slope (high MVRV -> low forward return)."""
    df, truth = data.synthetic_series(beta=0.50, seed=293)
    return df, truth
