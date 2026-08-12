"""Shared fixtures — deterministic synthetic CLO-carry worlds (no network, no real data)."""
import os
import sys

import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from clo_aaa import data  # noqa: E402


def _as_returns(world: pd.DataFrame) -> pd.DataFrame:
    """Rename a synthetic world's columns to the strategy's excess-of-cash convention:
    the cash leg becomes ``BIL`` so ``carry_stats(..., cash='BIL')`` works unchanged."""
    return world.rename(columns={"cash": "BIL", "carry": "CARRY", "dur": "DUR"})


@pytest.fixture(scope="session")
def carry_world():
    """A planted +1.2%/yr excess-of-cash carry (AAA-CLO analogue) + a high-vol decoy."""
    return _as_returns(data.synthetic_world(carry_annual=0.012, seed=888))


@pytest.fixture(scope="session")
def null_world():
    """The null — the carry leg is just cash + noise; its excess-of-cash must NOT fire."""
    return _as_returns(data.synthetic_world(carry_annual=0.0, seed=888))
