"""Shared fixtures -- deterministic synthetic (hashrate, price) paths with a *known*
hashrate->price lead-lag, so tests never touch the network and the only thing the
predictive regression can recover is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from bitcoin_hashrate import data  # noqa: E402


@pytest.fixture
def null_series():
    """No hashrate->price lead-lag (beta=0). Predictive regression should read ~zero."""
    df, truth = data.synthetic_series(beta=0.0, seed=292)
    return df, truth


@pytest.fixture
def live_series():
    """A planted hashrate->price lead-lag (beta=0.60). Regression should recover t>2."""
    df, truth = data.synthetic_series(beta=0.60, seed=292)
    return df, truth
