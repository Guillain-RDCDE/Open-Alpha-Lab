"""Shared fixtures for Study 98 (High-Noon) — deterministic, offline."""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from high_noon import data  # noqa: E402


@pytest.fixture
def null_walk():
    """A driftful random walk (ath_reversal 0) — highs are benign; no bearishness to find."""
    return data.synthetic_daily(n_days=8000, ath_reversal=0.0, seed=98)


@pytest.fixture
def planted_reversal():
    """A tape with a mean-reversion-at-highs drag (ath_reversal 1.0) — ATH is genuinely bearish."""
    return data.synthetic_daily(n_days=8000, ath_reversal=1.0, seed=98)
