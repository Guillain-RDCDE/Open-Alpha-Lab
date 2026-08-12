"""Shared fixtures — deterministic synthetic 60/40 worlds (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from vt6040 import data  # noqa: E402

# A regime draw (seed 904) whose clustering is strong enough for the thermostat to bite on
# and whose flat-vol twin is quiet — chosen once, then frozen.
FIXTURE_SEED = 904
FIXTURE_DAYS = 6000


@pytest.fixture(scope="session")
def edge_world():
    """Clustered portfolio vol + regime-independent drift — vol-targeting SHOULD pay here."""
    frame, truth = data.synthetic_prices(seed=FIXTURE_SEED, n_days=FIXTURE_DAYS)
    return frame, truth


@pytest.fixture(scope="session")
def null_world():
    """Flat vol (sigma_lo == sigma_hi) — nothing to forecast, the overlay must add nothing."""
    frame, truth = data.synthetic_prices(seed=FIXTURE_SEED, n_days=FIXTURE_DAYS, sigma_hi=0.006)
    return frame, truth
