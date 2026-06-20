"""Shared fixtures for Study 338 (Preferred-Stocks) — deterministic, offline.

Two synthetic worlds, one knob (``pref_beta``): a high-beta tape where the preferred
leg is equity-in-disguise (the positive control), and a low-beta tape where it is
genuinely bond-like (the null). Tests never touch the network."""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from preferred_stocks import data  # noqa: E402


@pytest.fixture
def equity_like():
    """pref_beta=0.90 — the preferred leg crashes WITH stocks (the positive control)."""
    return data.synthetic_three_asset(n_days=4500, pref_beta=0.90, seed=338)


@pytest.fixture
def bond_like():
    """pref_beta=0.05 — the preferred leg cushions like a bond (the null)."""
    return data.synthetic_three_asset(n_days=4500, pref_beta=0.05, seed=338)
