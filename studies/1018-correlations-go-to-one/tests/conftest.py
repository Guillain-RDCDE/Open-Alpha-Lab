"""Shared fixtures for Study 1018 — deterministic, offline, no network.

Both synthetic worlds come from the same generator with one knob moved. In the **null**
(``signal_strength=0``) every pair of assets has a true correlation of exactly 0.30 on every
day while volatility clusters and spikes — the artefact alone. In the **planted** world the
true correlation jumps to 0.60 whenever the latent volatility state is in its top decile. A
detector that fires on both is measuring the artefact; one that fires on neither is blind.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from goestoone import data  # noqa: E402

N_ASSETS = 8
N_YEARS = 12


@pytest.fixture(scope="session")
def planted():
    """A panel with the crisis-correlation jump planted at full size (signal_strength=1)."""
    return data.synthetic_panel(n_assets=N_ASSETS, n_years=N_YEARS, signal_strength=1.0,
                                seed=7)


@pytest.fixture(scope="session")
def null_panel():
    """The matching null: constant correlation, the same stochastic volatility."""
    return data.synthetic_panel(n_assets=N_ASSETS, n_years=N_YEARS, signal_strength=0.0,
                                seed=7)
