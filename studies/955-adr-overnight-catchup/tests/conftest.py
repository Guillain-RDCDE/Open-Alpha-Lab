"""Shared fixtures — deterministic synthetic ADR panels for Study 955 (ADR Catch-Up).

Two fixtures, both offline and deterministic (fixed seed 955; no network, no cache):

- ``planted`` — a panel in which 40% of each ADR's loading on its home market arrives a
  day late (``signal_strength=1``): a genuine, detectable stale-price effect the whole
  battery must find.
- ``null_panel`` — the same world with the lag switched off (``signal_strength=0``): the
  ADR prices the home move in full the same day, so nothing may fire.

The panels are session-scoped because they are read-only and each one costs a few tenths
of a second to build.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from adr_catchup import data, strategy as st  # noqa: E402

# Small enough to keep the suite quick, long enough for a 252-day rolling beta plus
# several years of out-of-sample residual.
N_YEARS = 8
N_NAMES = 6


@pytest.fixture(scope="session")
def planted():
    """A panel with a genuine planted catch-up lag (40% of the loading arrives late)."""
    panel, truth = data.synthetic_panel(
        n_names=N_NAMES, n_years=N_YEARS, signal_strength=1.0, seed=955
    )
    return st.add_residual(panel), truth


@pytest.fixture(scope="session")
def null_panel():
    """The null: the ADR prices its home market's move in full, same day."""
    panel, truth = data.synthetic_panel(
        n_names=N_NAMES, n_years=N_YEARS, signal_strength=0.0, seed=955
    )
    return st.add_residual(panel), truth


@pytest.fixture(scope="session")
def raw_planted():
    """The planted panel *before* the residual is attached (for data-layer tests)."""
    return data.synthetic_panel(n_names=N_NAMES, n_years=N_YEARS,
                                signal_strength=1.0, seed=955)
