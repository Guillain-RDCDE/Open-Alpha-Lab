"""Shared fixtures — deterministic synthetic tapes for Study 934 (Lump Sum vs DCA).

Three worlds, all offline and deterministic (fixed seeds; no network):

- ``rising`` — a fat planted excess drift (``signal_strength=1``): every dollar waiting
  in cash forgoes a real risk premium, so the lump sum must win.
- ``flat`` — the null (``signal_strength=0``): the asset earns exactly the cash rate in
  expectation, so the mean gap must sit on zero and the win rate on one half.
- ``falling`` — a negative excess drift (``signal_strength=-1``): DCA must win. This is
  the second half of the two-sided control; without it a harness that always crowned
  the lump sum would look "validated".
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from lump_vs_dca import data  # noqa: E402


@pytest.fixture(scope="session")
def rising():
    """A tape with a genuine planted risk premium — the lump sum should win."""
    return data.synthetic_daily(signal_strength=1.0, seed=934)


@pytest.fixture(scope="session")
def flat():
    """The null: the asset earns the cash rate in expectation."""
    return data.synthetic_daily(signal_strength=0.0, seed=934)


@pytest.fixture(scope="session")
def falling():
    """A falling tape — DCA should win (the other side of the control)."""
    return data.synthetic_daily(signal_strength=-1.0, seed=934)


@pytest.fixture(scope="session")
def panel():
    """Equity + bond + cash, for the bond-heavy variant."""
    return data.synthetic_panel(signal_strength=1.0, seed=934)
