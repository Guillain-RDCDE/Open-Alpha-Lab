"""Shared fixtures — deterministic synthetic panels (no network, no real data).

A null panel with a common time effect (β = 0 — nothing to find, the whole demonstration) and
an edge panel with a genuinely planted slope (the positive control), so tests never touch the
network and the only thing an honest estimator can reward (a real slope) is either baked in or
deliberately absent."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from clustered_se import data  # noqa: E402

# Modest sizes so the suite runs fast but the pitfall is unmistakable.
N_REPS = 800
N_PERIODS = 40
N_FIRMS = 40


@pytest.fixture(scope="session")
def null_panel():
    """β = 0 with a common time effect (ρ_x = ρ_e = 0.5): every rejection is a false positive."""
    return data.panel(N_REPS, N_PERIODS, N_FIRMS, rho_x=0.5, rho_e=0.5, beta=0.0, seed=840)


@pytest.fixture(scope="session")
def iid_panel():
    """ρ_x = ρ_e = 0: no common time factor — the pitfall must vanish (all SEs calibrated)."""
    return data.panel(N_REPS, N_PERIODS, N_FIRMS, rho_x=0.0, rho_e=0.0, beta=0.0, seed=840)


@pytest.fixture(scope="session")
def edge_panel():
    """A genuinely planted slope — Fama-MacBeth must fire on it (the control)."""
    return data.panel(N_REPS, N_PERIODS, N_FIRMS, rho_x=0.5, rho_e=0.5, beta=0.06, seed=840)
