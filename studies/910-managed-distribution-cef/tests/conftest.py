"""Shared fixtures — deterministic synthetic CEF worlds (no network, no real data)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from md_cef import data  # noqa: E402


@pytest.fixture(scope="session")
def null_world():
    """The null — the CEF is pure levered beta; no structural carry, no leak."""
    return data.synthetic_world(carry_annual=0.0, roc_leak_annual=0.0, seed=910)


@pytest.fixture(scope="session")
def planted_world():
    """A genuine planted net carry (+5 %/yr) on top of levered beta."""
    return data.synthetic_world(carry_annual=0.05, roc_leak_annual=0.0, seed=910)


@pytest.fixture(scope="session")
def roc_trap_world():
    """The mREIT trap: a fat payout that is ALL return-of-capital (carry == leak == net 0)."""
    return data.synthetic_world(carry_annual=0.05, roc_leak_annual=0.05, seed=910)
