"""Shared fixtures — deterministic synthetic daily return panels with tunable
correlation and drift, so tests never touch the network and the only thing a
global blend can improve (variance reduction via low correlation) is either
baked in or deliberately absent."""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from home_bias import data  # noqa: E402


@pytest.fixture
def equal_mu_low_corr():
    """Same expected return for US and international, low correlation — diversification should help."""
    ret, truth = data.synthetic_panel(n_days=3000, mu_us=0.07 / 252, mu_intl=0.07 / 252,
                                       corr_intl=0.40, seed=145)
    return ret, truth


@pytest.fixture
def us_dominant():
    """US outperforms international by 4%/yr — the post-2009 regime; blend likely lags on return."""
    ret, truth = data.synthetic_panel(n_days=3000, mu_us=0.11 / 252, mu_intl=0.07 / 252,
                                       corr_intl=0.70, seed=145)
    return ret, truth


@pytest.fixture
def full_correlation():
    """Perfect correlation — diversification has zero benefit by construction."""
    ret, truth = data.synthetic_panel(n_days=3000, mu_us=0.07 / 252, mu_intl=0.07 / 252,
                                       corr_intl=1.0, seed=145)
    return ret, truth
