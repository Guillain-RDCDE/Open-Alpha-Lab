"""Shared fixtures — deterministic synthetic effect batteries, no network.

Three tapes: a pure NULL (n_true=0 -> nothing real, only luck can 'win'), a STRONG
positive control (10 real of 100, an edge the corrections must recover), and a WEAK
control (20 real of 200, faint edges where the FWER-vs-FDR trade-off shows)."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from multiple_testing import data  # noqa: E402


@pytest.fixture
def null_battery():
    """100 effects, none real — every |t|>2 'winner' is a false positive."""
    return data.synthetic_battery(n_days=2520, m_effects=100, n_true=0, seed=346)


@pytest.fixture
def strong_battery():
    """10 real of 100, strong edge — the corrections must recover all 10, 0 false."""
    return data.synthetic_battery(n_days=2520, m_effects=100, n_true=10, seed=346)


@pytest.fixture
def weak_battery():
    """20 real of 200, faint edge — BH should out-power Bonferroni/Holm."""
    return data.synthetic_battery(n_days=2520, m_effects=200, n_true=20, edge=0.0025, seed=346)
