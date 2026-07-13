"""Offline, deterministic tests for Study 712 — "CGC-graded key comics".

No network: every test uses the hardcoded comic index or the fixed-seed synthetic
control. Pure-function assertions on the engine.

    pytest -q studies/712-comic-book-index/tests
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from comic_book_index import data, strategy as st


def test_index_shape_and_base():
    idx = data.load_comic_index()
    assert idx.iloc[0] == 100.0                      # base 100 @ 2018
    assert idx.index[0].year == 2018 and idx.index[-1].year == 2025
    # the load-bearing shape: a 2021 peak then a giveback into 2024
    assert idx.loc["2021-12-31"] == idx.max()
    assert idx.loc["2024-12-31"] < idx.loc["2021-12-31"]


def test_cagr_and_drawdown_signs():
    idx = data.load_comic_index()
    s = st.summarize(idx, periods_per_year=1.0)
    # modest positive gross CAGR, and a real (negative) drawdown
    assert 0.0 < s["cagr"] < 0.12
    assert s["mdd"] < 0.0


def test_costs_turn_return_negative():
    """The headline: grading + spread + carry flips the gross return negative."""
    idx = data.load_comic_index()
    g = st.summarize(idx, periods_per_year=1.0)["cagr"]
    h = st.net_of_costs_cagr(g, round_trip_spread=0.25, hold_years=3.0,
                             grading_fee_pct=0.02, carry_per_year=0.01)
    assert h["gross_cagr"] > 0.0
    assert h["net_cagr"] < 0.0                        # the whole point
    assert h["spread_drag_annual"] < 0.0
    assert h["grading_drag_annual"] < 0.0


def test_annual_excess_is_negative_vs_synthetic_benchmark():
    """Against a benchmark that compounds faster, the excess mean is negative."""
    idx = data.load_comic_index()
    # a deterministic 17%/yr year-end benchmark on the same clock
    yrs = idx.index
    bench = pd.Series([100 * (1.17) ** i for i in range(len(yrs))], index=yrs)
    ae = st.annual_excess_t(idx, bench)
    assert ae["n"] == 7
    assert ae["mean_excess"] < 0.0


def test_newey_west_recovers_planted_beta():
    """On synthetic data with a known beta, NW alpha is ~0 and beta ~ planted."""
    rng = np.random.default_rng(712)
    n = 120
    x = pd.Series(rng.normal(0.01, 0.04, n))
    y = 1.3 * x + pd.Series(rng.normal(0.0, 0.01, n))   # beta 1.3, zero alpha
    nw = st.newey_west_alpha_t(y, x, lags=6)
    assert abs(nw["beta"] - 1.3) < 0.15
    assert abs(nw["alpha_m"]) < 0.01
    assert abs(nw["t_alpha"]) < 2.0                     # no spurious alpha


def test_synthetic_control_recovers_sign():
    syn = data.synthetic_bubble()
    cr = st.control_recovers(syn, planted_sign=1)
    assert cr["sign_ok"] == 1
    assert np.isfinite(cr["sharpe"])
    # a real bubble-and-round-trip: big peak, big drawdown
    assert syn.max() > 300
    assert st.summarize(syn)["mdd"] < -0.3
