"""Shared fixtures -- deterministic synthetic BTC tapes with a *known* amount of
short-horizon mean reversion (the only thing a contrarian fear gauge can harvest),
so tests never touch the network and the control is either baked in or absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from crypto_fear_greed import data  # noqa: E402


@pytest.fixture
def null_tape():
    """No reversion (pure random walk + vol clustering). Spread should NOT clear t>2."""
    df, truth = data.synthetic_btc(n_days=2500, reversion=0.0, seed=325)
    return df, truth


@pytest.fixture
def control_tape():
    """A strongly planted mean reversion. Contrarian spread should clear t>2."""
    df, truth = data.synthetic_btc(n_days=2500, reversion=0.5, seed=325)
    return df, truth
