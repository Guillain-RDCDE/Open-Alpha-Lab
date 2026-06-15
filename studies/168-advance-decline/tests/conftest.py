"""Shared fixtures — deterministic synthetic A/D tapes with a *known* amount of
planted divergence (or none), so tests never touch the network."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from advance_decline import data  # noqa: E402


@pytest.fixture
def null_tape():
    """A tape with no planted divergence — the null hypothesis."""
    panel, spy, truth = data.synthetic_daily(
        n_stocks=100, n_days=504, divergence=0.0, seed=168
    )
    ad = data.build_ad_line(panel)
    return spy, ad, truth


@pytest.fixture
def divergence_tape():
    """A tape with planted bearish divergences — the A/D line should lag the index."""
    panel, spy, truth = data.synthetic_daily(
        n_stocks=100, n_days=504, divergence=2.0,
        n_div_episodes=4, div_len=15, seed=168
    )
    ad = data.build_ad_line(panel)
    return spy, ad, truth
