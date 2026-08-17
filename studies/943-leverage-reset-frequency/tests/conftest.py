"""Shared fixtures — deterministic synthetic tapes for Study 943 (Reset Frequency).

Three worlds, all offline and deterministic (fixed seed 943; no network, no cache):

- ``choppy`` — daily log returns with a *negative* AR(1) (``phi = -0.15``): moves partly
  reverse, so resetting leverage every day systematically buys high and sells low and the
  monthly reset should win.
- ``trending`` — a *positive* AR(1) (``phi = +0.15``): moves persist, so the daily reset
  compounds the trend and should win.
- ``null`` — ``signal_strength = 0``: iid log returns. The average monthly-minus-daily
  reset gap must be centred at zero here.

Twelve years each, which keeps the whole suite comfortably fast.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from reset_freq import data  # noqa: E402

N_YEARS = 12


@pytest.fixture
def choppy():
    """A mean-reverting (reversal-rich) world — the monthly reset should win here."""
    return data.synthetic_daily(phi=-0.15, signal_strength=1.0, n_years=N_YEARS, seed=943)


@pytest.fixture
def trending():
    """A momentum-rich world — the daily reset should win here."""
    return data.synthetic_daily(phi=+0.15, signal_strength=1.0, n_years=N_YEARS, seed=943)


@pytest.fixture
def null():
    """An iid random walk — no reset frequency may claim an edge here."""
    return data.synthetic_daily(phi=-0.15, signal_strength=0.0, n_years=N_YEARS, seed=943)
