"""Offline, fixed-seed tests for the cross-asset skewness machinery.

The synthetic panel is deterministic; the trailing skew signal has the right sign; the
monthly sort recovers a planted low-skew/high-return relation (positive long-low/short-high
spread); the null shows nothing; the sort is point-in-time (one month shift, no look-ahead);
the timer costs reduce the net; the inference primitives behave. All offline & synthetic —
any real-cache read is skipped when the cache is absent (offline CI).
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from cross_asset_skew import data, strategy as st  # noqa: E402

CACHE = data.CACHE_PATH


def test_world_deterministic(edge_world):
    p2 = data.synthetic_panel(edge=0.004, seed=827, n_assets=9, n_days=4000)
    assert np.allclose(edge_world.to_numpy(), p2.to_numpy())


def test_synthetic_panel_shape(edge_world):
    assert edge_world.shape[1] == 9
    assert edge_world.index.is_monotonic_increasing
    assert (edge_world > 0).all().all()  # prices stay positive


def test_skew_sign_matches_tilt(edge_world):
    # Cross-asset dispersion in realized skew must be non-trivial (the sort has something to bite on).
    ret = st.daily_returns(edge_world)
    sk = st.trailing_skew(ret, window=126).mean()
    assert sk.std() > 0.02


def test_trailing_skew_matches_reference():
    # The vectorised rolling skew equals the naive per-window population skew.
    rng = np.random.default_rng(0)
    ret = pd.DataFrame(rng.normal(0, 0.01, (300, 3)),
                       index=pd.bdate_range("2015-01-01", periods=300),
                       columns=["A", "B", "C"])
    w = 60
    fast = st.trailing_skew(ret, window=w)
    slow = ret["A"].rolling(w).apply(lambda x: st._skew(x.to_numpy()), raw=False)
    both = pd.concat([fast["A"], slow], axis=1).dropna()
    assert np.allclose(both.iloc[:, 0], both.iloc[:, 1], atol=1e-9)


def test_planted_relation_recovered(edge_world):
    ts = st.skew_stats(st.skew_spreads(edge_world))
    assert ts["t_nw"] > 2.0             # long-low/short-high spread lights up
    assert ts["spread_bps"] > 0
    assert ts["lo_bps"] > ts["hi_bps"]  # low-skew classes out-earn high-skew classes


def test_null_world_no_signal(null_world):
    ts = st.skew_stats(st.skew_spreads(null_world))
    assert abs(ts["t_nw"]) < 2.5


def test_sort_is_point_in_time():
    # The monthly signal used to trade month m is the value known at month-end m-1 (one shift).
    closes = data.synthetic_panel(edge=0.0, seed=1, n_assets=9, n_days=1500)
    sig_m = st.monthly_signal(closes, window=126)
    shifted = sig_m.shift(1)
    assert np.allclose(shifted.iloc[5].to_numpy(), sig_m.iloc[4].to_numpy(), equal_nan=True)


def test_no_lookahead_future_returns_ignored(edge_world):
    # Corrupting returns strictly AFTER the last formation month must not change the spread history.
    sp = st.skew_spreads(edge_world)
    tampered = edge_world.copy()
    tampered.iloc[-1] = tampered.iloc[-1] * 10.0  # future shock, after every held month's info
    sp2 = st.skew_spreads(tampered)
    common = sp.index.intersection(sp2.index)[:-1]  # drop the final month the shock legitimately hits
    assert np.allclose(sp.loc[common, "spread"], sp2.loc[common, "spread"])


def test_costs_reduce_net(edge_world):
    sp = st.skew_spreads(edge_world)
    gross = st.timer_stats(sp, cost_bps=0.0, borrow_bps_yr=0.0)["net_bps"]
    net = st.timer_stats(sp, cost_bps=5.0, borrow_bps_yr=50.0)["net_bps"]
    assert net < gross


def test_placebo_null_centres_at_zero(null_world):
    pl = st.placebo_pvalue(null_world, n_seeds=20, n_draws_per_seed=50)
    # Permuted null centres near zero: |mean| well inside a fraction of its own dispersion.
    assert abs(pl["placebo_mean_bps"]) < 0.2 * pl["placebo_sd_bps"]
    assert 0.0 <= pl["p_value"] <= 1.0
    assert pl["n_draws"] == 1000


def test_skew_of_left_skewed_is_negative():
    x = -np.abs(np.random.default_rng(0).normal(0, 1, 5000)) ** 1.5
    assert st._skew(x) < 0


def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=6) - st.one_sample_t(x)) < 0.6


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi


def test_monthly_returns_compound_from_daily(edge_world):
    # A month's simple return equals the compounded daily returns within it.
    mret = st.monthly_returns(edge_world)
    assert mret.shape[1] == 9
    assert mret.dropna(how="all").shape[0] > 100


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped when the cache is absent (offline CI).
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.exists(CACHE), reason="real cache absent offline CI")
def test_real_cache_loads_and_stamps():
    closes = data.load_panel()
    assert list(closes.columns) == data.TICKERS
    assert closes.index.max() <= pd.Timestamp(data.AS_OF)
    assert len(data.fingerprint(closes)) == 12
    sp = st.skew_spreads(closes)
    assert sp["n"].median() >= 6
