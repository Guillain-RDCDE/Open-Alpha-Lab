"""Shared fixtures — deterministic synthetic CPPI worlds (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from cppi import data  # noqa: E402

FIXTURE_SEED = 897


@pytest.fixture(scope="session")
def bear_world():
    """A falling tape (drift < 0): CPPI must de-risk and pin its drawdown near the floor."""
    return data.synthetic_prices(seed=FIXTURE_SEED, n_days=2500, drift=-0.0004, sigma=0.016)


@pytest.fixture(scope="session")
def calm_world():
    """A gentle bull, low vol: little drawdown, so CPPI and buy-and-hold nearly coincide."""
    return data.synthetic_prices(seed=FIXTURE_SEED, n_days=2500, drift=0.0004, sigma=0.006)


@pytest.fixture(scope="session")
def gap_world():
    """Crash gaps of 25% > 1/m (m=5 ⇒ 20%): the floor must break."""
    return data.synthetic_prices(seed=FIXTURE_SEED, n_days=4000, drift=0.0003, sigma=0.011,
                                 gap_prob=0.01, gap_size=0.25)


@pytest.fixture(scope="session")
def nogap_world():
    """Crash gaps of 15% < 1/m (m=5 ⇒ 20%): the floor holds through every gap."""
    return data.synthetic_prices(seed=FIXTURE_SEED, n_days=4000, drift=0.0003, sigma=0.011,
                                 gap_prob=0.01, gap_size=0.15)
