"""Shared fixtures — deterministic synthetic bull/bear-regime worlds (no network, no
real data). The whole suite runs offline; the single real-cache test is skipif-gated."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from rp_trend import data  # noqa: E402


@pytest.fixture(scope="session")
def null_world():
    """edge=0 — no persistent downtrend; the 200d trend gate must NOT improve the Sharpe."""
    return data.synthetic_world(edge=0.0, seed=896, n_days=5000)


@pytest.fixture(scope="session")
def planted_world():
    """edge=1 — sustained bear regimes; a 200d gate MUST cut drawdown and lift the Sharpe."""
    return data.synthetic_world(edge=1.0, seed=896, n_days=5000)
