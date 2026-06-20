"""The trend signal, the one-lag backtest's invariants, the cost accounting, and the study's
two spines: (1) the overlay beats buy-and-hold on Sharpe *only* when persistence is real, and
(2) on a pure random walk the overlay still cuts the drawdown while adding no Sharpe — proof
that drawdown reduction is not the same as skill."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lithium_boom import strategy as st  # noqa: E402


# ---- signal ---------------------------------------------------------------
def test_trend_signal_is_binary_and_lagged_correctly(random_walk):
    bars, _ = random_walk
    sig = st.trend_signal(bars, sma_n=200)
    vals = sig.dropna().unique()
    assert set(vals) <= {0.0, 1.0}
    # the signal is undefined during the SMA warmup
    assert sig.iloc[:199].isna().all()


def test_single_execution_lag(random_walk):
    """The held position over day t equals the signal known at the close of t-1 — exactly
    one shift, no second silent lag."""
    bars, _ = random_walk
    sig = st.trend_signal(bars, sma_n=50)
    bt = st.backtest(bars, sig, cost_bps=0.0)
    expected = sig.shift(1).fillna(0.0)
    assert np.allclose(bt["pos"].to_numpy(), expected.to_numpy())


# ---- backtest invariants --------------------------------------------------
def test_backtest_columns(random_walk):
    bars, _ = random_walk
    sig = st.trend_signal(bars, sma_n=200)
    bt = st.backtest(bars, sig, cost_bps=10.0)
    assert list(bt.columns) == ["asset_ret", "pos", "strat_gross", "strat_net"]
    assert len(bt) == len(bars)


def test_flat_book_earns_cash_not_asset(random_walk):
    """When out of the market (pos=0) the gross book earns the daily risk-free rate."""
    bars, _ = random_walk
    sig = pd.Series(0.0, index=bars.index)  # never invested
    bt = st.backtest(bars, sig, cost_bps=0.0, rf_annual=0.0252)
    rf_d = 0.0252 / 252
    assert np.allclose(bt["strat_gross"].to_numpy(), rf_d)


def test_always_invested_matches_asset(random_walk):
    """Signal all-ones (after the lag) reproduces the asset's own return, gross."""
    bars, _ = random_walk
    sig = pd.Series(1.0, index=bars.index)
    bt = st.backtest(bars, sig, cost_bps=0.0)
    # after the first (lagged) day, gross == asset_ret
    assert np.allclose(bt["strat_gross"].to_numpy()[1:], bt["asset_ret"].to_numpy()[1:])


def test_cost_monotonically_lowers_return(random_walk):
    bars, _ = random_walk
    sig = st.trend_signal(bars, sma_n=200)
    means = [
        np.nanmean(st.backtest(bars, sig, cost_bps=c)["strat_net"].to_numpy())
        for c in (0.0, 5.0, 20.0)
    ]
    assert means[0] > means[1] > means[2]


def test_cost_only_charged_on_switches(random_walk):
    """A never-switching (all-flat) book pays no cost; a switching book does."""
    bars, _ = random_walk
    flat = pd.Series(0.0, index=bars.index)
    bt_flat = st.backtest(bars, flat, cost_bps=50.0)
    assert np.allclose(bt_flat["strat_net"].to_numpy(), bt_flat["strat_gross"].to_numpy())
    sig = st.trend_signal(bars, sma_n=200)
    bt = st.backtest(bars, sig, cost_bps=50.0)
    assert (bt["strat_net"] < bt["strat_gross"]).any()


# ---- statistics -----------------------------------------------------------
def test_sharpe_and_cagr_finite(trending):
    bars, _ = trending
    sig = st.trend_signal(bars, sma_n=200)
    bt = st.backtest(bars, sig, cost_bps=10.0)
    s = st.summarize(bt["strat_net"].to_numpy())
    assert np.isfinite(s["sharpe"])
    assert np.isfinite(s["cagr"])
    assert s["max_dd"] <= 0.0


def test_max_drawdown_sign_and_bounds(random_walk):
    bars, _ = random_walk
    r = bars["close"].pct_change().fillna(0.0).to_numpy()
    dd = st.max_drawdown(r)
    assert -1.0 <= dd <= 0.0


def test_bootstrap_ci_brackets_point_estimate(trending):
    bars, _ = trending
    sig = st.trend_signal(bars, sma_n=200)
    bt = st.backtest(bars, sig, cost_bps=10.0)
    r = bt["strat_net"].to_numpy()
    point = st.sharpe(r)
    lo, hi = st.block_bootstrap_sharpe_ci(r, n_boot=400, seed=1)
    assert lo <= hi
    assert lo - 0.6 <= point <= hi + 0.6  # the point estimate is near the bootstrap mass


# ---- the random-timing control --------------------------------------------
def test_random_timing_signal_reproducible_and_targets_exposure():
    a = st.random_timing_signal(5000, 0.4, seed=3)
    b = st.random_timing_signal(5000, 0.4, seed=3)
    assert np.array_equal(a, b)
    assert set(np.unique(a)) <= {0.0, 1.0}
    assert 0.36 < a.mean() < 0.44


# ---- the two spines --------------------------------------------------------
def test_trend_cuts_drawdown_even_on_random_walk(random_walk):
    """Spine #2 — the misattribution: on a pure boom-bust random walk the long/flat overlay
    cuts the drawdown but adds ~no Sharpe. Drawdown reduction is structural, not skill."""
    bars, _ = random_walk
    sig = st.trend_signal(bars, sma_n=200)
    bt = st.backtest(bars, sig, cost_bps=10.0)
    bh_dd = st.max_drawdown(bt["asset_ret"].to_numpy())
    tr_dd = st.max_drawdown(bt["strat_net"].to_numpy())
    bh_sh = st.sharpe(bt["asset_ret"].to_numpy())
    tr_sh = st.sharpe(bt["strat_net"].to_numpy())
    assert tr_dd > bh_dd  # less negative = smaller drawdown
    assert tr_sh - bh_sh < 0.05  # but essentially no Sharpe gain


def test_trend_beats_buyhold_only_with_persistence(random_walk, trending):
    """Spine #1 — the overlay's Sharpe edge over buy-and-hold appears with planted trend
    persistence and not on a random walk."""

    def sharpe_edge(bars):
        sig = st.trend_signal(bars, sma_n=200)
        bt = st.backtest(bars, sig, cost_bps=10.0)
        return st.sharpe(bt["strat_net"].to_numpy()) - st.sharpe(bt["asset_ret"].to_numpy())

    edge_rw = sharpe_edge(random_walk[0])
    edge_tr = sharpe_edge(trending[0])
    assert edge_tr > edge_rw
    assert edge_tr > 0.05  # a clear Sharpe improvement when trend is real


def test_trend_timing_beats_random_timing(trending):
    """The trend signal's timing beats a random-timing control of the same exposure budget —
    so the *when* carries information, not merely the lower average exposure."""
    bars, _ = trending
    sig = st.trend_signal(bars, sma_n=200)
    bt = st.backtest(bars, sig, cost_bps=10.0)
    expo = bt["pos"].mean()
    tr_sh = st.sharpe(bt["strat_net"].to_numpy())
    rand_sh = [
        st.sharpe(st.backtest(bars, st.random_timing_signal(len(bars), expo, seed=s),
                              cost_bps=10.0)["strat_net"].to_numpy())
        for s in range(40)
    ]
    assert tr_sh > np.nanmedian(rand_sh)
