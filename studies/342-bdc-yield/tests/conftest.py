"""Shared fixtures for Study 342 (BDC-Yield) — deterministic, offline.

Two synthetic worlds, one knob (``bdc_beta``): a high-beta tape where the BDC leg is
levered equity (the positive control), and a low-beta tape where it is genuinely
bond-like (the null). Tests never touch the network."""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bdc_yield import data  # noqa: E402


@pytest.fixture
def equity_like():
    """bdc_beta=1.15 — the BDC leg crashes WITH stocks, amplified (the positive control)."""
    return data.synthetic_three_asset(n_days=3300, bdc_beta=1.15, seed=342)


@pytest.fixture
def bond_like():
    """bdc_beta=0.05 — the BDC leg cushions like a bond (the null)."""
    return data.synthetic_three_asset(n_days=3300, bdc_beta=0.05, seed=342)
