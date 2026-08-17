"""Shared fixtures — deterministic synthetic tapes for Study 956 (the ADR custody fee).

Every fixture is offline and deterministic (fixed seed 956; no network, no cache):

- ``planted`` — one ADR / home-line / FX triple carrying a **known** 25 bp/yr custody fee
  on top of a 15 % withholding leak on a 3.5 % gross yield.
- ``null_pair`` — the same world with ``signal_strength=0``: no fee, no withholding. The
  estimator must report a drag indistinguishable from zero.
- ``planted_panel`` / ``null_panel`` — ten independent pairs of each, for the pooled
  estimator and its cross-name inference.
- ``broken_pair`` — a planted world with a 0.70-log ADS-ratio step at mid-sample, for the
  level-shift detector.
- ``no_dividend_pair`` — a world whose *home* leg carries no dividend adjustment at all,
  reproducing the London Stock Exchange defect the coverage screen exists to catch.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from adr_drag import data  # noqa: E402


@pytest.fixture
def planted():
    return data.synthetic_pair(signal_strength=1.0, seed=956)


@pytest.fixture
def null_pair():
    return data.synthetic_pair(signal_strength=0.0, seed=956)


@pytest.fixture
def planted_panel():
    return data.synthetic_panel(n_names=10, drag_bps_per_year=25.0,
                                signal_strength=1.0, seed=956)


@pytest.fixture
def null_panel():
    return data.synthetic_panel(n_names=10, drag_bps_per_year=25.0,
                                signal_strength=0.0, seed=956)


@pytest.fixture
def broken_pair():
    return data.synthetic_pair(signal_strength=1.0, ratio_break=0.70, seed=956)


@pytest.fixture
def no_dividend_pair():
    """A pair whose home leg is split-adjusted only — the London defect, reproduced.

    The ADR keeps its dividends; the home line's total-return column is overwritten with
    its price-only column, exactly as Yahoo reports an LSE listing.
    """
    df, truth = data.synthetic_pair(signal_strength=1.0, seed=956)
    df = df.copy()
    df["loc_tr"] = df["loc_px"]
    df["loc_tr_usd"] = df["loc_px_usd"]
    return df, truth
