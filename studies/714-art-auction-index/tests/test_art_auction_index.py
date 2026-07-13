"""Fully-offline, deterministic tests for Study 714 (contemporary-art auction index).

No network: exercises the hardcoded art index, the pure inference functions on synthetic
data, and the buyer's-premium haircut identity. Run with ``pytest -q``.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from art_auction_index import data, strategy as st


def test_art_index_shape_and_anchors():
    idx = data.load_art_index()
    # Base 100 @ 2000, a real 2000->2025 span, and the two peaks/two round-trips.
    assert idx.iloc[0] == 100.0
    assert idx.index[0].year == 2000 and idx.index[-1].year == 2025
    assert len(idx) == 26
    # 2007 pre-crisis peak > 2009 trough by ~44% (financial-crisis crash).
    crash = idx.loc["2009-12-31"] / idx.loc["2007-12-31"] - 1.0
    assert -0.50 < crash < -0.38
    # 2022 records peak, then a 2023-24 correction.
    assert idx.loc["2022-12-31"] > idx.loc["2024-12-31"]


def test_index_underperforms_spy_synthetic():
    # Deterministic: a steady 8.76%/yr benchmark beats the art index's ~5.7%/yr.
    idx = data.load_art_index()
    years = pd.to_datetime([f"{y}-12-31" for y in range(2000, 2026)])
    bench = pd.Series([100 * (1.0876) ** i for i in range(len(years))], index=years)
    ae = st.annual_excess_t(idx, bench)
    assert ae["n"] == 25
    assert ae["mean_excess"] < 0.0          # art loses to a stock-like benchmark
    assert abs(ae["t"]) < 2.0               # not significant -> Signal NONE


def test_premium_haircut_is_a_real_drag():
    # A ~28% round-trip over 7y + carry must reduce the net CAGR below the gross.
    h = st.net_of_premium_cagr(0.057, buyers_premium=0.25, sellers_commission=0.10,
                               hold_years=7.0, insure_per_year=0.01)
    assert abs(h["round_trip_mult"] - 0.90 / 1.25) < 1e-9
    assert h["spread_drag_annual"] < 0.0
    assert h["net_cagr"] < h["gross_cagr"]
    # With these inputs the gross ~5.7% is wiped out to ~zero/negative.
    assert h["net_cagr"] < 0.01


def test_control_recovers_planted_sign():
    syn = data.synthetic_bubble()
    cr = st.control_recovers(syn, planted_sign=1)
    assert cr["sign_ok"] == 1
    assert np.isfinite(cr["sharpe"])
    # It is a genuine bubble-and-round-trip: a real drawdown from a higher peak.
    assert st.max_drawdown(syn) < -0.30


def test_cagr_and_drawdown_primitives():
    lv = pd.Series([100.0, 150.0, 75.0, 150.0],
                   index=pd.to_datetime(["2000-12-31", "2001-12-31", "2002-12-31", "2003-12-31"]))
    # 100 -> 150 over ~3 years -> (1.5)^(1/3) - 1 ~= 0.1447 (day-count 365.25, so ~1e-3 tol).
    assert abs(st.cagr(lv) - ((1.5) ** (1 / 3) - 1)) < 1e-3
    # Worst peak-to-trough is 150 -> 75 = -50%.
    assert abs(st.max_drawdown(lv) - (-0.5)) < 1e-9
