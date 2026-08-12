"""Signal logic, backtest invariants, and the study's spine — all offline/synthetic.

The spine: on a planted dead-decade tape the 200-day trend filter cuts max drawdown vs
buy-and-hold AND beats an exposure-matched random control (genuine timing, not mere
exposure reduction); on a flat-vol null it produces no reliable excess-Sharpe advantage.
Costs monotonically reduce the strategy's return; the rebalance lag prevents look-ahead.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gold_trend import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Signal tests
# --------------------------------------------------------------------------- #
def test_signal_is_binary_and_lagged(dead_decade):
    prices, _ = dead_decade
    sig = st.trend_signal(prices["asset"], sma_n=200)
    vals = set(sig.dropna().unique().tolist())
    assert vals.issubset({0.0, 1.0})
    assert sig.iloc[0:1].isna().all()  # first bar NaN from the shift (no look-ahead)


def test_signal_all_in_when_monotone_rising():
    close = pd.Series(np.linspace(100, 300, 400),
                      index=pd.bdate_range("2020-01-02", periods=400))
    sig = st.trend_signal(close, sma_n=200)
    assert sig.dropna().iloc[200:].eq(1.0).all()


def test_signal_all_out_when_monotone_falling():
    close = pd.Series(np.linspace(300, 100, 400),
                      index=pd.bdate_range("2020-01-02", periods=400))
    sig = st.trend_signal(close, sma_n=200)
    assert sig.dropna().eq(0.0).all()


def test_no_lookahead_signal_uses_only_past():
    """Shifting the price tail must not change earlier signals (causality check)."""
    rng = np.random.default_rng(0)
    close = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 600))),
                      index=pd.bdate_range("2015-01-01", periods=600))
    sig_full = st.trend_signal(close, sma_n=200)
    close2 = close.copy()
    close2.iloc[500:] *= 3.0  # perturb only the future
    sig_pert = st.trend_signal(close2, sma_n=200)
    # Signals up to bar 500 are formed from data <= t and acted on at t+1, so bars
    # strictly before the perturbation are untouched.
    assert (sig_full.iloc[:500].fillna(-9) == sig_pert.iloc[:500].fillna(-9)).all()


def test_random_control_matches_in_market_fraction(dead_decade):
    prices, _ = dead_decade
    sig = st.trend_signal(prices["asset"], sma_n=200)
    rand = st.random_signal(sig, seed=0)
    assert abs(float(sig.dropna().mean()) - float(rand.dropna().mean())) < 0.05


def test_random_control_reproducible_and_seed_sensitive(dead_decade):
    prices, _ = dead_decade
    sig = st.trend_signal(prices["asset"])
    r1 = st.random_signal(sig, seed=5)
    r2 = st.random_signal(sig, seed=5)
    r3 = st.random_signal(sig, seed=6)
    assert (r1.dropna() == r2.dropna()).all()
    assert not (r1.dropna() == r3.dropna()).all()


# --------------------------------------------------------------------------- #
# Backtest engine invariants
# --------------------------------------------------------------------------- #
def test_backtest_columns_and_dropna(dead_decade):
    prices, _ = dead_decade
    sig = st.trend_signal(prices["asset"])
    bt = st.run_backtest(prices["asset"], prices["cash"], sig, cost_bps=5.0)
    assert {"r_gold", "r_cash", "signal", "r_strategy"}.issubset(bt.columns)
    assert not bt.isna().any().any()


def test_cash_days_earn_cash_not_asset():
    """When out (signal 0) the strategy must earn the cash return, not the asset return."""
    close = pd.Series(np.linspace(300, 150, 300),
                      index=pd.bdate_range("2020-01-02", periods=300))
    cash = pd.Series(100 * (1.0 + 0.03 / 252) ** np.arange(300),
                     index=close.index)
    sig = st.trend_signal(close, sma_n=200)
    bt = st.run_backtest(close, cash, sig, cost_bps=0.0)
    out_days = bt[bt["signal"] == 0]
    if len(out_days) > 10:
        resid = (out_days["r_strategy"] - out_days["r_cash"]).abs()
        assert (resid < 1e-9).mean() > 0.95  # only switch days differ (here cost=0 => none)


def test_costs_monotonically_reduce_return(dead_decade):
    prices, _ = dead_decade
    sig = st.trend_signal(prices["asset"])
    means = []
    for c in (0.0, 5.0, 25.0):
        bt = st.run_backtest(prices["asset"], prices["cash"], sig, cost_bps=c)
        means.append(bt["r_strategy"].mean())
    assert means[0] >= means[1] >= means[2]


# --------------------------------------------------------------------------- #
# Inference primitives sanity
# --------------------------------------------------------------------------- #
def test_newey_west_recovers_positive_mean():
    rng = np.random.default_rng(1)
    x = 0.001 + rng.normal(0, 0.01, 5000)
    assert st.newey_west_t(x) > 3
    assert abs(st.newey_west_t(rng.normal(0, 0.01, 5000))) < 3


def test_welch_and_one_sample_finite(dead_decade):
    prices, _ = dead_decade
    cmp = st.compare(prices["asset"], prices["cash"])
    assert np.isfinite(cmp["t_diff_vs_bh"])
    assert np.isfinite(st.one_sample_t(cmp["e_trend"].to_numpy()))
    assert np.isfinite(st.welch_t(cmp["e_trend"].to_numpy(), cmp["e_bh"].to_numpy()))


def test_wilson_interval_brackets_point():
    lo, hi = st.wilson_interval(60, 100)
    assert lo < 0.60 < hi


def test_bootstrap_ci_brackets_point(dead_decade):
    prices, _ = dead_decade
    cmp = st.compare(prices["asset"], prices["cash"])
    ci = st.bootstrap_sharpe_ci(cmp["e_bh"], n_boot=500, seed=912)
    assert ci["ci_low"] <= ci["sharpe"] <= ci["ci_high"]


# --------------------------------------------------------------------------- #
# The study's spine — machinery is unbiased
# --------------------------------------------------------------------------- #
def test_planted_regime_trend_cuts_drawdown(dead_decade):
    """On a genuine dead-decade tape the trend filter cuts max drawdown vs buy-and-hold."""
    prices, _ = dead_decade
    d = st.synthetic_detect(prices)
    # drawdowns are negative; a shallower (less negative) trend DD is a smaller magnitude.
    assert d["dd_trend_abs"] > d["dd_bh_abs"]  # trend DD less deep than BH DD


def test_planted_regime_beats_random_control(dead_decade):
    """The DD reduction is timing, not mere exposure: trend beats the random control."""
    prices, _ = dead_decade
    d = st.synthetic_detect(prices)
    assert d["dd_trend_abs"] > d["dd_rand_abs"]  # trend DD shallower than random's


def test_planted_regime_positive_excess_sharpe_advantage(dead_decade):
    prices, _ = dead_decade
    d = st.synthetic_detect(prices)
    assert d["excess_sharpe_adv"] > 0.0


def test_null_no_spurious_sharpe_alpha(flat_vol):
    """On the flat-vol null the trend filter must not manufacture a Sharpe advantage."""
    prices, _ = flat_vol
    d = st.synthetic_detect(prices)
    assert abs(d["excess_sharpe_adv"]) < 0.4


def test_null_across_seeds_is_centered(flat_vol):
    advs = np.array([
        st.synthetic_detect(data.synthetic_daily(signal_strength=0.0, seed=912 + s)[0])["excess_sharpe_adv"]
        for s in range(6)
    ])
    assert abs(advs.mean()) < 0.25
    assert (np.abs(advs) >= 0.4).sum() <= 1


def test_era_cut_returns_both_halves(dead_decade):
    prices, _ = dead_decade
    eras = st.era_cut(prices["asset"], prices["cash"], split="2009-01-01")
    assert eras["early"] is not None and eras["late"] is not None
    for e in eras.values():
        assert np.isfinite(e["excess_sharpe_adv"])


def test_calendar_year_table_shape(dead_decade):
    prices, _ = dead_decade
    tbl = st.calendar_year_table(prices["asset"], prices["cash"])
    assert {"bh_pct", "trend_pct", "in_gold_frac"}.issubset(tbl.columns)
    assert len(tbl) > 5
