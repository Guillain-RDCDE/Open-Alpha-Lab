"""Fully-offline, deterministic tests for Study 852 — Movie-Sequel Fatigue.

No network on the default path: everything runs on the seeded synthetic worlds and on
pure functions. The one real-tape smoke test is skipped unless the yfinance cache is
present (absent on CI), so the whole suite is green synthetic-only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from sequel_fatigue import data as dt, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# The curated calendar
# --------------------------------------------------------------------------- #
def test_events_calendar_shape():
    assert len(dt.EVENTS) == 46
    seen_tickers = set()
    for franchise, title, seq, date, ticker in dt.EVENTS:
        assert isinstance(franchise, str) and franchise
        assert isinstance(title, str) and title
        assert isinstance(seq, int) and seq >= 1
        assert len(date) == 10 and date[4] == "-" and date[7] == "-"
        assert ticker in dt.STUDIO_TICKERS
        seen_tickers.add(ticker)
    assert seen_tickers == set(dt.STUDIO_TICKERS)
    years = sorted({int(d[:4]) for _, _, _, d, _ in dt.EVENTS})
    assert years[0] == 2003 and years[-1] >= 2024


def test_events_frame_sorted():
    ef = dt.events_frame()
    assert list(ef.columns) == ["franchise", "title", "seq", "opening", "ticker"]
    assert pd.api.types.is_datetime64_any_dtype(ef["opening"])


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_one_sample_t_matches_hand_calc():
    x = np.array([-0.01, -0.02, 0.03, 0.00, -0.015])
    r = st.one_sample_t(x)
    assert r["n"] == 5
    assert abs(r["mean"] - x.mean()) < 1e-12
    se = x.std(ddof=1) / np.sqrt(5)
    assert abs(r["t"] - x.mean() / se) < 1e-9


def test_ols_slope_recovers_known_line():
    # y = 2 + (-3) x + tiny noise -> slope ~ -3, strong negative t
    x = np.arange(1, 21, dtype=float)
    y = 2.0 - 3.0 * x + np.array([0.01 * ((-1) ** i) for i in range(20)])
    r = st.ols_slope(x, y)
    assert abs(r["slope"] + 3.0) < 0.01
    assert r["t"] < -50
    assert r["r"] < -0.99


def test_wilson_interval_brackets_point_estimate():
    lo, hi = st.wilson_interval(17, 43)
    assert lo < 17 / 43 < hi
    assert 0.0 <= lo <= hi <= 1.0


def test_newey_west_close_to_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.02, 3000)
    assert abs(st.newey_west_t(x, lags=4) - st.one_sample_t(x)["t"]) < 0.6


def test_welch_t_sign():
    a = np.array([-0.03, -0.02, -0.04, -0.01])
    b = np.array([0.01, 0.02, 0.00, 0.03])
    assert st.welch_t(a, b) < 0   # mean(a) < mean(b)


# --------------------------------------------------------------------------- #
# Synthetic world + the full pipeline on synthetic prices
# --------------------------------------------------------------------------- #
def test_world_deterministic(edge_world):
    a, b, ev = edge_world
    a2, b2, ev2 = dt.synthetic_world(edge=0.012, seed=852)
    assert np.allclose(a.to_numpy(), a2.to_numpy())
    assert np.allclose(b.to_numpy(), b2.to_numpy())
    assert ev.equals(ev2)


def test_build_event_cars_resolves_synthetic(edge_world):
    a, b, ev = edge_world
    prices = {"SYN": a, dt.BENCHMARK: b}
    e = ev.copy(); e["ticker"] = "SYN"; e["title"] = e["franchise"]
    cars = st.build_event_cars(prices, e)
    inc = cars[cars["included"]]
    assert len(inc) >= 30            # most synthetic entries resolve
    # anchor snapped forward, base session exists, car finite
    assert inc["car"].notna().all()
    assert (pd.to_datetime(inc["anchor"]) >= pd.to_datetime(inc["opening"])).all()


def test_synthetic_null_slope_is_quiet():
    # edge=0 -> the fatigue-slope detector should not systematically fire
    ts = np.array([st.synthetic_detect(edge=0.0, seed=852 + s)["t"] for s in range(20)])
    assert abs(ts.mean()) < 0.75
    # honest small-sample FPR bound: at ~41 events with a franchise fixed effect the
    # slope t is noisy; allow up to ~1/4 of seeds to breach |t|>=2 on the null.
    assert (np.abs(ts) >= 2).sum() <= 5


def test_synthetic_planted_edge_lowers_slope_monotonically():
    s0 = st.synthetic_detect(edge=0.004, seed=852)["slope"]
    s1 = st.synthetic_detect(edge=0.008, seed=852)["slope"]
    s2 = st.synthetic_detect(edge=0.012, seed=852)["slope"]
    assert s0 > s1 > s2          # more fatigue -> more negative slope
    assert s2 < -0.005           # a clear planted fatigue is detectable


def test_synthetic_planted_edge_recovers_negative_t(edge_world):
    a, b, ev = edge_world
    prices = {"SYN": a, dt.BENCHMARK: b}
    e = ev.copy(); e["ticker"] = "SYN"; e["title"] = e["franchise"]
    inc = st.build_event_cars(prices, e)
    inc = inc[inc["included"]]
    dem = st.fatigue_slope(inc, demean=True)
    assert dem["slope"] < 0 and dem["t"] < -1.5


def test_synthetic_persistence_recovered():
    # persist>0 with a fresh shock -> positive AR(1) on the reaction sequence
    t_null = np.mean([st.synthetic_detect(edge=0.0, seed=852 + s, persist=0.0,
                                          shock_sd=0.02)["ar1_t"] for s in range(8)])
    t_pers = np.mean([st.synthetic_detect(edge=0.0, seed=852 + s, persist=0.7,
                                          shock_sd=0.02)["ar1_t"] for s in range(8)])
    assert t_pers > t_null
    assert t_pers > 0.8


def test_permute_placebo_null_not_significant():
    # on a NULL synthetic world the observed slope sits inside the permutation cloud
    a, b, ev = dt.synthetic_world(edge=0.0, seed=852)
    prices = {"SYN": a, dt.BENCHMARK: b}
    e = ev.copy(); e["ticker"] = "SYN"; e["title"] = e["franchise"]
    inc = st.build_event_cars(prices, e); inc = inc[inc["included"]]
    pl = st.permute_slope_pvalue(inc, demean=True, n_perm=1000)
    assert pl["p_two"] > 0.05


def test_timer_costs_reduce_net():
    a, b, ev = dt.synthetic_world(edge=0.012, persist=0.7, shock_sd=0.02, seed=852)
    prices = {"SYN": a, dt.BENCHMARK: b}
    e = ev.copy(); e["ticker"] = "SYN"; e["title"] = e["franchise"]
    inc = st.build_event_cars(prices, e); inc = inc[inc["included"]]
    gross = st.fatigue_timer(inc, cost_bps=0.0, borrow_bps_yr=0.0)["net_mean"]
    net = st.fatigue_timer(inc, cost_bps=10.0, borrow_bps_yr=50.0)["net_mean"]
    assert net < gross


# --------------------------------------------------------------------------- #
# Real-tape smoke test — skipped when the yfinance cache is absent (CI)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not dt.have_real(), reason="real cache absent offline CI")
def test_real_tape_resolves_events():
    prices = dt.load_real()
    cars = st.build_event_cars(prices)
    inc = cars[cars["included"]]
    # 3 pre-2021 Transformers drop (PARA continuity); the rest resolve
    assert len(inc) == 43
    assert cars["included"].sum() + (~cars["included"]).sum() == 46
    # the fatigue slope is finite and the machinery runs end-to-end
    dem = st.fatigue_slope(inc, demean=True)
    assert np.isfinite(dem["slope"]) and np.isfinite(dem["t"])
