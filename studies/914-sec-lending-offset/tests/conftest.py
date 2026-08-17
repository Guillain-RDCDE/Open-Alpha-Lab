"""Shared fixtures — deterministic synthetic ETF pairs for Study 914.

Three fixtures, all offline and deterministic (fixed seed 914; no network, no cache):

- ``planted_pair`` — a same-index ETF pair where fund A is handed a genuine 60 bp/yr of
  securities-lending revenue on top of its (dearer) fee (``signal_strength=1``). The
  fee-adjusted residual estimator must recover that +60 bp with |t| >= 2.
- ``null_pair`` — the same world with the lending differential switched off
  (``signal_strength=0``): the two funds differ only by fees and composition noise, so the
  residual is exactly zero in expectation and the estimator must stay quiet.
- ``planted_panel`` — five independent planted pairs, for the pooling power check.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from lending_offset import data  # noqa: E402


@pytest.fixture
def planted_pair():
    """A pair with a real, planted +60 bp/yr lending passthrough on the dearer leg."""
    return data.synthetic_daily(signal_strength=1.0, seed=914)


@pytest.fixture
def null_pair():
    """The null: fees and composition noise only, no lending differential."""
    return data.synthetic_daily(signal_strength=0.0, seed=914)


@pytest.fixture
def planted_panel():
    """Five independent planted pairs — the pooling check."""
    return data.synthetic_panel(signal_strength=1.0, seed=914)
