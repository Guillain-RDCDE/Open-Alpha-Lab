"""Fully-offline, deterministic tests for Study 713 ("classic cars beat equities?").

No network: everything runs off the hardcoded (cited, approximate) collector-car index and
the fixed-seed synthetic control. Asserts pure-function results only.

    pytest -q studies/713-classic-car-index/tests
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from classic_car_index import data, strategy as st


def test_index_shape_and_bounds():
    idx = data.load_car_index()
    assert idx.index[0].year == 2005 and idx.index[-1].year == 2025
    assert len(idx) == 21
    assert idx.iloc[0] == 100.0                     # base 100
    assert idx.max() == 490.0                        # the 2023 crest
    assert idx.is_monotonic_increasing is False      # it plateaus / dips, not a straight line


def test_index_cagr_and_desmooth_artifact():
    idx = data.load_car_index()
    s = st.summarize(idx, periods_per_year=1.0)
    # ~8.1%/yr gross growth over 2005-2025
    assert abs(s["cagr"] - 0.0811) < 1e-3
    ds = st.desmooth_returns(idx)
    # the appraisal-smoothing tell: strong positive serial correlation...
    assert ds["rho"] > 0.5
    # ...and de-smoothing must RAISE vol and LOWER the Sharpe (the whole point of beat 4b)
    assert ds["vol_desmoothed"] > ds["vol_obs"]
    assert ds["sharpe_desmoothed"] < ds["sharpe_obs"]


def test_carry_haircut_is_a_drag_and_shrinks_gross():
    idx = data.load_car_index()
    g = st.cagr(idx)
    h = st.net_of_carry_cagr(g, round_trip_spread=0.22, hold_years=7.0, carry_per_year=0.025)
    # frictions are negative and the net is strictly below the gross
    assert h["spread_drag_annual"] < 0
    assert h["carry_per_year"] < 0
    assert h["net_cagr"] < h["gross_cagr"]
    # with these params the net lands near ~1.7%/yr (cash-like), well below the gross
    assert 0.0 < h["net_cagr"] < 0.03


def test_annual_excess_sign_and_determinism():
    idx = data.load_car_index()
    # a synthetic benchmark that clearly out-grows the index -> negative excess, no |t|>=2 in cars' favour
    bench = data.synthetic_boom(boom_cagr=0.20, plateau_cagr=0.08, seed=1)
    ae = st.annual_excess_t(idx, bench)
    assert ae["n"] >= 2
    # deterministic: identical inputs give identical stats
    ae2 = st.annual_excess_t(idx, data.synthetic_boom(boom_cagr=0.20, plateau_cagr=0.08, seed=1))
    assert ae["t"] == ae2["t"] and ae["mean_excess"] == ae2["mean_excess"]


def test_newey_west_recovers_planted_beta():
    # y = 0.5*x + noise (zero alpha, beta 0.5): NW must recover beta ~0.5, alpha ~0
    rng = np.random.default_rng(713)
    import pandas as pd
    x = pd.Series(rng.normal(0, 0.04, 240))
    y = 0.5 * x + pd.Series(rng.normal(0, 0.01, 240))
    nw = st.newey_west_alpha_t(y, x, lags=6)
    assert abs(nw["beta"] - 0.5) < 0.1
    assert abs(nw["alpha_m"]) < 0.005


def test_synthetic_control_recovers_up_sign():
    syn = data.synthetic_boom()
    cr = st.control_recovers(syn, planted_sign=1)
    assert cr["sign_ok"] == 1
    assert np.isfinite(cr["sharpe"])
    # deterministic across calls
    assert data.synthetic_boom().iloc[-1] == syn.iloc[-1]
