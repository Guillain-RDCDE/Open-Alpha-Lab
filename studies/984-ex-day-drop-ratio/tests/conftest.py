"""Shared fixtures for Study 984 — deterministic, offline, no network.

The planted world drops the price by a controllable fraction of the dividend on
each ex-date and buries that drop under realistic daily volatility. Because the dividend is
small and the noise is not, the generator is also the study's power analysis: it shows how far
a naive average of per-event ratios can be from the truth even when the truth is planted.

Both worlds come from the same generator with one knob moved, which is the whole point:
a test that passes on the planted world and *also* passes on the null is not testing
anything.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from exday import data  # noqa: E402

N_DAYS = 3000
N_TICKERS = 10


@pytest.fixture
def planted():
    """A tape whose ex-day drop is exactly the dividend — the textbook world."""
    return data.synthetic_panel(n=N_DAYS, n_tickers=N_TICKERS, drop_fraction=1.0, seed=984)


@pytest.fixture
def partial():
    """The Elton-Gruber world: the price gives up only 78% of the dividend."""
    return data.synthetic_panel(n=N_DAYS, n_tickers=N_TICKERS,
                                drop_fraction=data.ELTON_GRUBER_DEFAULT, seed=984)


@pytest.fixture
def null_panel():
    """The matching null: dividends are paid and the price does not move for them at all."""
    return data.synthetic_panel(n=N_DAYS, n_tickers=N_TICKERS, drop_fraction=0.0, seed=984)


@pytest.fixture
def noiseless():
    """The arithmetic world: no volatility, so every estimator must agree exactly."""
    return data.synthetic_panel(n=N_DAYS, n_tickers=4, drop_fraction=0.8,
                                daily_vol=1e-9, market_beta=0.0, seed=984)

