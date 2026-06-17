"""Shared fixtures — a deterministic synthetic QMJ panel with a *known* quality premium,
so tests never touch the network and the inference machinery can be validated against a
planted signal (null and non-null).
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from quality_minus_junk import data  # noqa: E402


@pytest.fixture
def null_panel():
    """No QMJ premium (qmj_premium=0) — the long-short must earn ~0 in expectation."""
    signal, fwd_ret, truth = data.synthetic_panel(
        n_firms=200, n_years=20, qmj_premium=0.0, seed=242
    )
    return signal, fwd_ret, truth


@pytest.fixture
def live_panel():
    """Positive QMJ premium (qmj_premium=0.08) — the long-short should clearly outperform."""
    signal, fwd_ret, truth = data.synthetic_panel(
        n_firms=300, n_years=20, qmj_premium=0.08, seed=242
    )
    return signal, fwd_ret, truth
