"""Offline, fixed-seed tests for the Dry-January / Veganuary machinery.

The synthetic panel is deterministic; a planted January abnormal-return seasonal is
recovered by the monthly-seasonality detector (dummy-regression Newey-West *t*); the null
world stays quiet across a ≥20-seed average; the monthly build and abnormal returns are
correct; the placebo ranks a planted month as extreme; the timer costs cut the net; the
inference primitives behave. All offline (synthetic only). The single real-cache smoke test
is skipped when the parquet is absent.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from dry_january import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic world — determinism & the positive control
# --------------------------------------------------------------------------- #
def test_world_deterministic(edge_world):
    p2 = data.synthetic_panel(edge=0.05, seed=849)
    for sym in edge_world:
        assert np.allclose(edge_world[sym].to_numpy(), p2[sym].to_numpy())


def test_planted_january_recovered(edge_world):
    out = st.synthetic_detect(edge_world, month=1)
    assert out["t_nw"] > 2.0          # the January dummy lights up
    assert out["beta_pct"] > 0        # planted edge is positive
    assert out["mean_pct"] > 0


def test_null_world_quiet_on_average():
    # House rule: no single lucky seed manufactures significance — average over 20.
    r = st.synthetic_mean_t(data, edge=0.0, n_seeds=20)
    assert abs(r["mean_t_nw"]) < 1.0
    assert r["fire_frac"] == 0.0


def test_synthetic_monotone_in_edge():
    r0 = st.synthetic_mean_t(data, edge=0.0, n_seeds=20)
    r1 = st.synthetic_mean_t(data, edge=0.05, n_seeds=20)
    assert r1["mean_t_nw"] > r0["mean_t_nw"] + 2.0
    assert r1["mean_beta_pct"] > r0["mean_beta_pct"]


# --------------------------------------------------------------------------- #
# Monthly build — returns, baskets, abnormal returns
# --------------------------------------------------------------------------- #
def test_monthly_return_is_month_end_pct_change():
    idx = pd.bdate_range("2020-01-01", periods=90)
    s = pd.Series(100.0 * np.cumprod(1.0 + np.full(90, 0.001)), index=idx, name="X")
    mr = st.monthly_return(s).dropna()
    # constant daily compounding -> positive, finite monthly returns
    assert (mr > 0).all()
    assert mr.index.is_monotonic_increasing


def test_abnormal_is_group_minus_bench():
    idx = pd.date_range("2000-01-31", periods=24, freq="ME")
    g = pd.Series(np.linspace(0.01, 0.03, 24), index=idx)
    b = pd.Series(np.linspace(0.005, 0.02, 24), index=idx)
    ar = st.abnormal_monthly(g, b)
    assert np.allclose(ar.to_numpy(), (g - b).to_numpy())


def test_basket_is_equal_weight_mean():
    idx = pd.bdate_range("2010-01-01", periods=300)
    rng = np.random.default_rng(0)
    a = pd.Series(100.0 * np.cumprod(1.0 + rng.normal(0, 0.01, 300)), index=idx, name="BUD")
    b = pd.Series(100.0 * np.cumprod(1.0 + rng.normal(0, 0.01, 300)), index=idx, name="STZ")
    closes = {"BUD": a, "STZ": b}
    basket = st.basket_monthly(closes, ["BUD", "STZ"])
    manual = pd.DataFrame(
        {"BUD": st.monthly_return(a), "STZ": st.monthly_return(b)}
    ).mean(axis=1)
    assert np.allclose(basket.dropna().to_numpy(), manual.dropna().to_numpy())


# --------------------------------------------------------------------------- #
# Seasonality inference — dummy regression, placebo, month table
# --------------------------------------------------------------------------- #
def _ar_with_bump(month, bump, n_years=25, seed=1):
    idx = pd.date_range("2000-01-31", periods=n_years * 12, freq="ME")
    rng = np.random.default_rng(seed)
    x = rng.normal(0.0, 0.02, len(idx))
    x = x + np.where(idx.month == month, bump, 0.0)
    return pd.Series(x, index=idx, name="ar")


def test_dummy_regression_recovers_bump():
    ar = _ar_with_bump(month=1, bump=0.05)
    reg = st.dummy_regression_nw(ar, month=1)
    assert reg["beta"] > 0.03            # ~0.05 planted premium
    assert reg["t_nw"] > 3.0


def test_dummy_regression_flat_when_no_bump():
    ar = _ar_with_bump(month=1, bump=0.0)
    reg = st.dummy_regression_nw(ar, month=1)
    assert abs(reg["t_nw"]) < 2.5


def test_month_placebo_ranks_planted_month_first():
    ar = _ar_with_bump(month=3, bump=0.06)
    pl = st.month_placebo(ar, target=3, tail="right")
    assert pl["rank"] == 1               # March is the most positive of 12
    assert pl["p_value"] <= 1.0 / 12 + 1e-9


def test_month_table_has_twelve_rows():
    ar = _ar_with_bump(month=1, bump=0.0)
    tab = st.month_table(ar)
    assert list(tab.index) == list(range(1, 13))
    assert (tab["n"] > 0).all()


# --------------------------------------------------------------------------- #
# The costed timer
# --------------------------------------------------------------------------- #
def _synthetic_closes(seed=7, n_years=8):
    idx = pd.bdate_range("2005-01-03", periods=n_years * 252)
    rng = np.random.default_rng(seed)
    out = {}
    for t in data.ALL_TICKERS:
        out[t] = pd.Series(
            100.0 * np.cumprod(1.0 + rng.normal(0.0, 0.012, len(idx))), index=idx, name=t
        )
    return out


def test_timer_costs_reduce_net():
    closes = _synthetic_closes()
    gross = st.timer_stats(closes, cost_bps=0.0, borrow_bps_yr=0.0)["net_pct"]
    net = st.timer_stats(closes, cost_bps=5.0, borrow_bps_yr=50.0)["net_pct"]
    assert net < gross
    assert st.timer_stats(closes)["n_years"] > 0


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=3) - st.one_sample_t(x)) < 0.6


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(15, 27)
    assert lo < 15 / 27 < hi


def test_one_sample_t_zero_mean_is_small():
    rng = np.random.default_rng(3)
    x = rng.normal(0.0, 1.0, 5000)
    assert abs(st.one_sample_t(x)) < 3.0


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped offline (no network, no fetch)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.exists(data.CACHE_PATH), reason="no cached real tape")
def test_real_cache_loads_and_has_januaries():
    closes = data.load_closes()
    assert data.BENCH in closes
    abn = st.build_abnormals(closes)
    jan = abn["alcohol"]
    jan = jan[jan.index.month == 1]
    assert jan.dropna().shape[0] >= 20      # decades of Januaries for the alcohol basket
    assert isinstance(data.fingerprint(data.load_panel()), str)
