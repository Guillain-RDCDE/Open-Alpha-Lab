"""Fully-offline, deterministic tests for Study 711 ("a Birkin beats the S&P and gold").

No network: exercises the hardcoded resale index, the pure return/inference primitives on
synthetic/cached-independent data, the consignment haircut algebra, and the fixed-seed
synthetic positive control. Run: ``pytest -q studies/711-birkin-index/tests``.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from birkin_index import data, strategy as st


def test_resale_index_shape():
    idx = data.load_resale_index()
    assert idx.iloc[0] == 100.0                      # base 100 @ 2015
    assert idx.index[0].year == 2015 and idx.index[-1].year == 2025
    assert idx.is_monotonic_increasing is False       # it has the 2024 dip
    # honest, modest CAGR — nowhere near the 14.2% myth
    c = st.cagr(idx)
    assert 0.03 < c < 0.07


def test_baghunter_myth_is_labelled():
    m = data.baghunter_myth()
    assert abs(m["cagr"] - 0.142) < 1e-9
    assert m["window"] == "1980-2015"


def test_cagr_and_drawdown_identities():
    # a clean doubling over exactly 2 years -> CAGR = sqrt(2)-1
    lvl = pd.Series([100.0, 141.42135, 200.0],
                    index=pd.to_datetime(["2015-12-31", "2016-12-31", "2017-12-31"]))
    assert abs(st.cagr(lvl) - (2 ** 0.5 - 1)) < 1e-3
    # a monotone-up series has zero drawdown
    assert st.max_drawdown(lvl) == 0.0
    # a 20% dip is a -0.2 drawdown
    dip = pd.Series([100.0, 80.0, 90.0],
                    index=pd.to_datetime(["2015-12-31", "2016-12-31", "2017-12-31"]))
    assert abs(st.max_drawdown(dip) - (-0.2)) < 1e-12


def test_annual_excess_zero_when_identical():
    idx = data.load_resale_index()
    ae = st.annual_excess_t(idx, idx)                 # index vs itself
    assert abs(ae["mean_excess"]) < 1e-12
    assert ae["n"] == len(idx) - 1


def test_newey_west_recovers_planted_alpha():
    # y = 0.004 + 1.2*x + tiny noise -> alpha_m ~ 0.004, beta ~ 1.2, big |t|
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(0, 0.05, 400))
    y = 0.004 + 1.2 * x + rng.normal(0, 1e-4, 400)
    nw = st.newey_west_alpha_t(y, x, lags=6)
    assert abs(nw["alpha_m"] - 0.004) < 5e-4
    assert abs(nw["beta"] - 1.2) < 1e-2
    assert nw["t_alpha"] > 2                           # planted alpha is detectable


def test_consignment_haircut_turns_gross_negative():
    # the load-bearing result: +5% gross -> negative net after a 30% spread
    h = st.net_of_carry_cagr(0.05, round_trip_spread=0.30, hold_years=3.0, insure_per_year=0.005)
    assert h["spread_drag_annual"] < 0
    assert h["net_cagr"] < 0
    # no spread, no carry -> net == gross
    h0 = st.net_of_carry_cagr(0.05, round_trip_spread=0.0, hold_years=3.0, insure_per_year=0.0)
    assert abs(h0["net_cagr"] - 0.05) < 1e-12


def test_synthetic_control_is_deterministic_and_recovered():
    a = data.synthetic_compounder()
    b = data.synthetic_compounder()
    assert np.allclose(a.values, b.values)             # fixed seed -> identical
    cr = st.control_recovers(a, planted_sign=1)
    assert cr["sign_ok"] == 1                           # engine recovers the up-drift
    assert 0.06 < cr["cagr"] < 0.16                     # near the planted +12%
