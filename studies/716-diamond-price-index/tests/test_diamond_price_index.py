"""Fully offline, deterministic tests for Study 716 ("short the diamond").

No network: the price index is hardcoded and the synthetic collapse control is fixed-seed.
We assert pure-function results on that offline data — never on the yfinance cache — so the
suite runs anywhere. Run: ``pytest -q studies/716-diamond-price-index/tests``.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from diamond_price_index import data, strategy as st


def test_price_index_shape():
    """The hardcoded index is base-100 @ 2018, peaks in 2022, and falls into 2025."""
    idx = data.load_price_index()
    assert idx.iloc[0] == 100.0
    assert idx.index[0].year == 2018 and idx.index[-1].year == 2025
    # boom into 2021-22, then a decline to a lower terminal level
    assert idx.loc["2021-12-31"] > idx.loc["2018-12-31"]
    assert idx.loc["2025-12-31"] < idx.loc["2022-12-31"]
    assert idx.loc["2025-12-31"] < 100.0  # net a loser vs base


def test_peak_and_roundtrip():
    """The reported early-2022 top is above the terminal level (a real round-trip)."""
    pk_d, pk_l = data.price_peak()
    assert pk_d.year == 2022 and pk_d.month <= 3
    idx = data.load_price_index()
    roundtrip = idx.loc["2025-12-31"] / pk_l - 1.0
    assert roundtrip < -0.25  # ~ -32% peak-to-terminal


def test_summarize_negative_cagr():
    """As an asset the index has a negative CAGR (it lost money)."""
    idx = data.load_price_index()
    s = st.summarize(idx, periods_per_year=1.0)
    assert s["cagr"] < 0.0
    assert s["mdd"] < 0.0
    assert s["n"] == 7


def test_short_book_volatility_drag():
    """A short of a steadily-falling but volatile series need not compound well; borrow hurts.

    Synthetic collapse: level falls overall, so a *gross* short book is >= 1, but charging a
    fat borrow drags its CAGR strictly below the no-borrow version.
    """
    syn = data.synthetic_collapse()
    r = syn.pct_change().dropna()
    sb0 = st.short_book_from_returns(r, borrow_annual=0.0)
    sb1 = st.short_book_from_returns(r, borrow_annual=0.30)
    # borrow strictly reduces the short's CAGR
    assert sb1["cagr"] < sb0["cagr"]
    # and 30%/yr borrow pushes a modestly-positive gross short negative
    assert sb1["cagr"] < 0.0


def test_resale_haircut_is_deeply_negative():
    """The retail->resale round-trip turns any modest gross CAGR sharply negative."""
    h = st.net_of_resale_cagr(0.0, round_trip_spread=0.55, hold_years=5.0, insure_per_year=0.005)
    assert h["spread_drag_annual"] < -0.10       # big annual drag
    assert h["net_cagr"] < -0.10                  # net deeply negative even from 0% gross


def test_control_recovers_down_sign():
    """Positive control: the engine recovers the planted post-peak DOWN drift."""
    syn = data.synthetic_collapse()
    cr = st.control_recovers(syn, planted_sign=-1)
    assert cr["sign_ok"] == 1
    assert cr["back_cagr"] < 0.0


def test_annual_excess_t_direction():
    """Index vs a synthetic rising benchmark: excess is negative (diamonds underperform)."""
    idx = data.load_price_index()
    # a deterministic 15%/yr benchmark on the same year-end grid
    bench = pd.Series(
        [100.0 * (1.15 ** i) for i in range(len(idx))], index=idx.index, name="bench"
    )
    ae = st.annual_excess_t(idx, bench)
    assert ae["mean_excess"] < 0.0
    assert ae["n"] == 7
    assert ae["t"] < 0.0 and np.isfinite(ae["t"])
