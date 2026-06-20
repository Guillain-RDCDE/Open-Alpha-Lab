"""Shared fixtures for Study 340 (Bank-Loans) — deterministic, offline.

Two synthetic worlds, two knobs (``dur``, ``credit_beta``): a floating/credit-loaded tape
where the loan leg has near-zero duration but loads on equities (the positive control), and
a duration-bond tape where it is a plain long bond (the null). Tests never touch the
network."""

import os
import sys

import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bank_loans import data  # noqa: E402


@pytest.fixture
def floating_like():
    """dur=0.05, credit_beta=0.65 — near-zero rate risk, equity/credit-loaded (pos control)."""
    return data.synthetic_four_asset(dur=0.05, credit_beta=0.65, seed=340)


@pytest.fixture
def duration_like():
    """dur=12.0, credit_beta=0.05 — a plain long-duration bond (the null)."""
    return data.synthetic_four_asset(dur=12.0, credit_beta=0.05, seed=340)
