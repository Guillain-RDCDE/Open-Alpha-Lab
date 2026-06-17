"""Shared fixtures for Study 229 (Beneish M-score) tests.

All tests run on the synthetic panel — deterministic, offline, no network,
no EDGAR dependency. The ``has_premium`` fixture plants a real M-score → return
relationship (negative premium: high-M earns less); the ``no_premium`` fixture
is the null.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from beneish_m_score import data  # noqa: E402


@pytest.fixture
def has_premium():
    """Panel with a planted M-score → return premium (m_premium = -0.08).

    Negative m_premium means high-M (manipulator) firms earn less next year,
    which is the paper's prediction and the strategy's positive control.
    """
    m, fwd, truth = data.synthetic_panel(
        n_firms=200, n_years=20, m_premium=-0.08, seed=229
    )
    return m, fwd, truth


@pytest.fixture
def no_premium():
    """Panel under the null: M has zero relationship with next-year returns."""
    m, fwd, truth = data.synthetic_panel(
        n_firms=200, n_years=20, m_premium=0.0, seed=229
    )
    return m, fwd, truth
