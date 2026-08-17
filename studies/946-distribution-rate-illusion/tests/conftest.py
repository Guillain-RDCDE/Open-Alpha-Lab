"""Shared fixtures — deterministic synthetic fund panels for Study 946.

Three offline, fixed-seed worlds (seed 946, no network anywhere):

- ``null_panel`` — ``signal_strength=0``: the payout is pure return-of-capital. It is
  perfectly forecastable and it erodes the price leg one-for-one, but it carries **no**
  information about total return. This is the null the estimator must not fire on.
- ``planted_panel`` — ``signal_strength=1``: a genuine one-for-one yield-to-total-return
  bonus is planted. The estimator must find it.
- ``beta_confound_panel`` — ``signal_strength=0`` with ``beta_slope=0.5``: no alpha, but
  market beta falls across the yield sort, so the raw high-minus-low return is negative in
  an up-market. The CAPM control must absorb it back to zero.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from dist_illusion import data  # noqa: E402


@pytest.fixture
def null_panel():
    """Pure return-of-capital: payout predicts erosion, never total return."""
    return data.synthetic_panel(signal_strength=0.0, seed=946)


@pytest.fixture
def planted_panel():
    """A genuine planted yield-to-total-return link the sort must recover."""
    return data.synthetic_panel(signal_strength=1.0, seed=946)


@pytest.fixture
def beta_confound_panel():
    """No alpha, but beta declines across the yield sort — the CAPM control's job."""
    return data.synthetic_panel(signal_strength=0.0, beta_slope=0.5, seed=946)
