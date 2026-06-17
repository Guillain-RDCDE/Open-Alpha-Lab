"""Shared fixtures — deterministic synthetic BTC tapes with a *known* relationship
between the Mayer Multiple and forward returns, so tests never touch the network.

- ``null_tape`` — pure random walk; MM carries no forward-return information.
- ``signal_tape`` — price has mild mean-reversion when MM is elevated; MM genuinely
  predicts negative forward returns when above the sell threshold.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from mayer_multiple import data  # noqa: E402


@pytest.fixture
def null_tape():
    """A pure random-walk price — no Mayer-Multiple signal (signal_strength=0)."""
    df, truth = data.synthetic_btc(n_days=2_000, signal_strength=0.0, seed=221)
    return df, truth


@pytest.fixture
def signal_tape():
    """A tape where price genuinely mean-reverts when MM is elevated (signal_strength=1)."""
    df, truth = data.synthetic_btc(n_days=2_000, signal_strength=1.0, seed=221)
    return df, truth
