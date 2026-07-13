"""Shared fixtures -- deterministic synthetic (sopr, price) paths with a *known*
momentum SOPR->price link, so tests never touch the network and the only thing the
predictive regression can recover is baked in or deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from sopr import data  # noqa: E402


@pytest.fixture
def null_series():
    """No SOPR->price momentum link (beta=0). Predictive regression should read ~zero."""
    df, truth = data.synthetic_series(beta=0.0, seed=764)
    return df, truth


@pytest.fixture
def live_series():
    """A planted momentum SOPR->price link (beta=2.0). Regression should recover a
    significantly *positive* slope (SOPR > 1 -> higher forward return)."""
    df, truth = data.synthetic_series(beta=2.0, seed=764)
    return df, truth
