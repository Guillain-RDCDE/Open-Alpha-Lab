"""Shared fixtures — deterministic synthetic feature+return panels (no network, no real data).

Three worlds: a stationary null (no leak — the contrast), a non-stationary random-walk null (the
trap the study is about), and a planted real edge (the positive control the honest method must
recover). All offline and fixed-seed."""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from lookahead_standardization import data  # noqa: E402


@pytest.fixture(scope="session")
def stationary_null():
    """Stationary AR(1) feature, iid returns — a genuine null where full-sample z barely leaks."""
    return data.null_stationary(seed=data.BASE_SEED, n_names=60, n_days=1000)


@pytest.fixture(scope="session")
def nonstationary_null():
    """Random-walk feature, forward-change returns — the trap: full-sample z manufactures a fake IC."""
    return data.null_nonstationary(seed=data.BASE_SEED, n_names=60, n_days=1000, horizon=10)


@pytest.fixture(scope="session")
def planted():
    """A real, point-in-time-tradeable edge — the honest expanding method MUST recover it."""
    return data.planted_edge(seed=data.BASE_SEED, n_names=60, n_days=1000, beta=0.10)
