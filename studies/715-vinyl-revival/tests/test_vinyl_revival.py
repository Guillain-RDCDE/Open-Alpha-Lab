"""Fully offline, deterministic tests for Study 715 (vinyl revival).

No network: the RIAA vinyl-revenue series is hardcoded, and the synthetic control is
fixed-seed. We assert pure-function results on that in-memory data only. The equity
proxies (which need the cache) are deliberately NOT exercised here.

    pytest -q studies/715-vinyl-revival/tests
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from vinyl_revival import data, strategy as st


def test_vinyl_index_shape_and_growth():
    idx = data.load_vinyl_index()
    # base 100 @ 2010, 15 annual points through 2024, and a real multi-year climb.
    assert idx.index[0].year == 2010 and idx.index[-1].year == 2024
    assert len(idx) == 15
    assert abs(idx.iloc[0] - 100.0) < 1e-9
    # revenue climbed from 89M to 1400M -> CAGR well above 20%/yr (the trend is real).
    c = st.cagr(idx)
    assert 0.20 < c < 0.24


def test_annual_excess_growth_is_positive():
    # A noisy ~13%/yr benchmark, aligned to year-end (variance keeps the test honest).
    import pandas as pd
    rng = np.random.default_rng(715)
    idx = data.load_vinyl_index()
    yrs = pd.to_datetime([f"{y}-12-31" for y in range(2010, 2025)])
    lvl, cur = [], 100.0
    for i in range(len(yrs)):
        lvl.append(cur)
        cur *= (1.0 + 0.13 + rng.normal(0.0, 0.17))
    bench = pd.Series(lvl, index=yrs)
    res = st.annual_excess_t(idx, bench)
    # vinyl out-GREW a ~13%/yr bench on average -> positive excess, finite t, n == 14.
    assert res["mean_excess"] > 0.0
    assert res["n"] == 14
    assert np.isfinite(res["t"])


def test_newey_west_recovers_planted_beta():
    # Regress a series that is exactly 1.5*bench + noise -> beta ~1.5, alpha ~0, |t|<2.
    import pandas as pd
    rng = np.random.default_rng(0)
    n = 120
    bench = pd.Series(rng.normal(0.01, 0.04, n))
    proxy = 1.5 * bench + pd.Series(rng.normal(0.0, 0.02, n))
    nw = st.newey_west_alpha_t(proxy, bench, lags=6)
    assert abs(nw["beta"] - 1.5) < 0.15
    assert abs(nw["t_alpha"]) < 2.0


def test_collector_carry_turns_flat_resale_negative():
    # The realistic case: flat per-record resale -> net is negative after spread + storage.
    h = st.net_of_collector_carry(0.0, round_trip_spread=0.30, hold_years=5.0,
                                  storage_per_year=0.01)
    assert h["net_cagr"] < 0.0
    # And the charitable (full revenue-growth) case still keeps *some* of it.
    hc = st.net_of_collector_carry(0.2175, round_trip_spread=0.30, hold_years=5.0,
                                   storage_per_year=0.01)
    assert 0.10 < hc["net_cagr"] < 0.14


def test_synthetic_control_recovers_sign():
    syn = data.synthetic_revival()
    cr = st.control_recovers(syn, planted_sign=1)
    assert cr["sign_ok"] == 1
    assert cr["cagr"] > 0.0
    assert np.isfinite(cr["sharpe"])
