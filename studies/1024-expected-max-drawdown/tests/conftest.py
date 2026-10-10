"""Shared fixtures for Study 1024 — deterministic, offline, no network.

The synthetic world is a GARCH(1,1)-t whose ``signal_strength`` knob dials volatility
clustering and fat tails in and out at a fixed unconditional mean and variance. At ``0.0`` the
returns are i.i.d. Gaussian — the one world in which the textbook drawdown formula is exactly
right — so the test-suite can check the machinery is calibrated there *and* that it reports a
failure once the planted departures are switched on. A detector that fires in both worlds is
not testing anything.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from promised import strategy as st  # noqa: E402

N_YEARS = 50
N_TAPES = 4
SEED = 7


@pytest.fixture(scope="session")
def null_cov():
    """1-year windows on four i.i.d. Gaussian tapes (signal_strength = 0)."""
    return st.synthetic_coverage(0.0, n_years=N_YEARS, models=("gauss", "gauss_oracle"),
                                 n_tapes=N_TAPES, n_sims=2000, seed=SEED)


@pytest.fixture(scope="session")
def planted_cov():
    """The same windows with clustering and t(4) tails planted (signal_strength = 1)."""
    return st.synthetic_coverage(1.0, n_years=N_YEARS, models=("gauss", "gauss_oracle"),
                                 n_tapes=N_TAPES, n_sims=2000, seed=SEED)
