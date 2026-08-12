"""Shared fixtures — deterministic synthetic 90/10 worlds (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from cash_call import data  # noqa: E402

FIXTURE_SEED = 899


@pytest.fixture(scope="session")
def bear_world():
    """A falling tape (drift < 0): the 90/10 rule must de-risk and protect capital (shallow DD)."""
    return data.synthetic_prices(seed=FIXTURE_SEED, n_days=2500, drift=-0.0004, sigma=0.016)


@pytest.fixture(scope="session")
def calm_world():
    """A steady low-vol bull: little to protect, so 90/10 gives up upside to the premium bleed."""
    return data.synthetic_prices(seed=FIXTURE_SEED, n_days=2500, drift=0.0005, sigma=0.006)


@pytest.fixture(scope="session")
def jump_world():
    """A positively-skewed tape (up-jumps): the convex call sleeve rides the rallies, floor intact."""
    return data.synthetic_prices(seed=FIXTURE_SEED, n_days=3000, drift=0.0003, sigma=0.011,
                                 up_jump_prob=0.02, up_jump_size=0.06)
