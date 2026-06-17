"""Shared fixtures -- deterministic synthetic (supply, btc_return) panels with a
*known* amount of dry-powder predictability, so tests never touch the network and
the only thing the supply-timing signal can harvest is baked in or deliberately
absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from stablecoin_supply import data  # noqa: E402


@pytest.fixture
def null_panel():
    """No dry-powder premium (supply growth = noise). Timing must NOT beat buy-and-hold."""
    df, truth = data.synthetic_panel(n_months=200, premium=0.0, seed=295)
    return df, truth


@pytest.fixture
def live_panel():
    """A planted +0.04 dry-powder premium. Lagged supply growth should predict BTC."""
    df, truth = data.synthetic_panel(n_months=200, premium=0.04, seed=295)
    return df, truth
