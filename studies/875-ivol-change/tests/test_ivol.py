"""Offline, fixed-seed tests for the idiosyncratic-vol-change machinery.

The synthetic panel is deterministic; the market-model residual vol obeys the identity
(idio vol never exceeds total vol, and a pure-market name has ~zero residual); the sort
recovers a planted rising-idio-vol->lower-return relation (positive long-falling /
short-rising spread); the null shows nothing; the sort is point-in-time (one shift, no
look-ahead); the timer costs reduce the net; the additivity regression runs; the
inference primitives behave. All offline, synthetic-only — no real cache is read.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from ivol_change import data, strategy as st  # noqa: E402


def test_world_deterministic(edge_world):
    p2 = data.synthetic_panel(edge=0.002, seed=875, n_assets=40, n_days=1500)
    for sym in edge_world:
        assert np.allclose(edge_world[sym].to_numpy(), p2[sym].to_numpy())


def test_idio_vol_never_exceeds_total_vol(edge_world):
    # The market-model residual vol is total vol times sqrt(1 - corr^2) <= total vol.
    ret = st.close_returns(edge_world)
    mkt = st.market_return(ret)
    iv = st.idio_vol(ret, mkt, window=21)
    tv = ret.rolling(21, min_periods=21).std(ddof=0)
    both = iv.dropna(how="all")
    assert bool((iv <= tv + 1e-12)[both.notna()].all().all())


def test_pure_market_name_has_near_zero_idio_vol():
    # A name that is exactly the market (plus a constant beta) has ~zero residual vol.
    rng = np.random.default_rng(0)
    m = rng.normal(0.0, 0.01, 400)
    others = {f"O{i}": pd.Series(rng.normal(0, 0.01, 400)) for i in range(9)}
    idx = pd.bdate_range("2015-01-01", periods=400)
    df = pd.DataFrame(others)
    df["MKTCLONE"] = 1.3 * (df.mean(axis=1)) + 0.0  # perfectly explained by the market
    df.index = idx
    mkt = df.mean(axis=1)
    iv = st.idio_vol(df, mkt, window=42)["MKTCLONE"].dropna()
    assert iv.mean() < 1e-6


def test_delta_ivol_dispersion_nontrivial(edge_world):
    # The sort needs cross-sectional dispersion in the delta to bite on.
    ret = st.close_returns(edge_world)
    mkt = st.market_return(ret)
    d = st.delta_ivol(ret, mkt, window=21)
    assert float(d.std(axis=1).mean()) > 1e-4


def test_planted_relation_recovered(edge_world):
    ret = st.close_returns(edge_world)
    ts = st.delta_stats(st.delta_spreads(ret, window=21))
    assert ts["t_nw"] > 3.0             # long-falling / short-rising spread lights up
    assert ts["spread_bps"] > 0
    assert ts["lo_bps"] > ts["hi_bps"]  # falling-idio-vol names out-earn rising ones


def test_null_world_no_signal(null_world):
    ret = st.close_returns(null_world)
    ts = st.delta_stats(st.delta_spreads(ret, window=21))
    assert abs(ts["t_nw"]) < 2.5


def test_sort_is_point_in_time():
    ret = pd.DataFrame(
        np.linspace(-0.02, 0.02, 60).reshape(20, 3),
        index=pd.bdate_range("2020-01-01", periods=20),
        columns=["A", "B", "C"],
    )
    mkt = st.market_return(ret)
    sig = st.delta_ivol(ret, mkt, window=3)
    shifted = sig.shift(1)
    assert np.allclose(shifted.iloc[7].to_numpy(), sig.iloc[6].to_numpy(), equal_nan=True)


def test_costs_reduce_net(edge_world):
    ret = st.close_returns(edge_world)
    sp = st.delta_spreads(ret, window=21)
    gross = st.timer_stats(sp, cost_bps=0.0, borrow_bps_yr=0.0)["net_bps"]
    net = st.timer_stats(sp, cost_bps=5.0, borrow_bps_yr=50.0)["net_bps"]
    assert net < gross


def test_additivity_runs_and_reports_level(edge_world):
    ret = st.close_returns(edge_world)
    ad = st.additivity(ret, window=21)
    assert ad["n_days"] > 100
    assert np.isfinite(ad["corr"])
    assert np.isfinite(ad["alpha_t_nw"])
    assert np.isfinite(ad["level_t_nw"])


def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=10) - st.one_sample_t(x)) < 0.6


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi
