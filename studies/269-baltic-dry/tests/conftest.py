"""Shared fixtures -- deterministic synthetic (bdi, sp500) monthly frames with a
*known* BDI->equity lead-lag, so tests never touch the network and the only
predictability the regression can find is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from baltic_dry import data  # noqa: E402


@pytest.fixture
def null_frame():
    """No BDI->equity lead-lag (beta=0). Regression slope should be ~0, |t| small."""
    df, truth = data.synthetic_series(n_months=360, beta=0.0, seed=269)
    return df, truth


@pytest.fixture
def live_frame():
    """A planted beta=0.6 BDI->equity lead-lag. Regression slope should be strongly positive."""
    df, truth = data.synthetic_series(n_months=360, beta=0.6, seed=269)
    return df, truth
