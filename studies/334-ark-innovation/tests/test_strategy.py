"""The study's two spines, tested on the synthetic core:
(1) the behaviour gap (TWR − DWR) is ~0 under return-independent flows and clearly
    positive under performance-chasing flows; and
(2) the trend overlay beats a no-momentum null only when momentum is actually planted —
    plus the inference, IRR, cost and lag invariants the desk insists on."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ark_innovation import data, strategy as st  # noqa: E402

MPY = st.MONTHS_PER_YEAR
TPY = st.TRADING_DAYS_PER_YEAR


# ---- the money-weighted IRR machinery -------------------------------------
def test_irr_recovers_known_rate():
    """A single -100 in, +110 out one period later is a 10% per-period IRR; annualise it."""
    cf = np.array([-100.0, 110.0])
    irr = st.money_weighted_irr(cf, periods_per_year=1)
    assert abs(irr - 0.10) < 1e-6


def test_irr_nan_without_sign_change():
    assert np.isnan(st.money_weighted_irr(np.array([-1.0, -1.0, -1.0]), 12))


def test_time_weighted_return_cagr():
    """A price that doubles over exactly one year is a 100% CAGR."""
    price = pd.Series([100.0] + [100.0] * 250 + [200.0],
                      index=pd.date_range("2020-01-01", periods=252, freq="B"))
    twr = st.time_weighted_return(price, periods_per_year=251)
    assert abs(twr["total_return"] - 1.0) < 1e-9
    assert abs(twr["cagr"] - 1.0) < 1e-3


# ---- SPINE 1: the behaviour gap -------------------------------------------
def test_no_gap_for_steady_grower(steady_grower):
    """A fund that just goes up, with flat flows, has ~no behaviour gap — the average
    dollar earns roughly the buy-and-hold return (the null)."""
    frame, _ = steady_grower
    g = st.behaviour_gap(frame["price"], frame["flow"], MPY)
    assert np.isfinite(g["twr_cagr"]) and np.isfinite(g["dwr_irr"])
    assert abs(g["gap"]) < 0.05  # within 5 pts/yr of zero


def test_positive_gap_for_boom_bust(boom_bust, steady_grower):
    """A boom-bust fund with performance-chasing flows makes the average dollar earn
    materially less than buy-and-hold — a clearly positive gap, far bigger than the
    steady-grower null."""
    g_bb = st.behaviour_gap(boom_bust[0]["price"], boom_bust[0]["flow"], MPY)
    g_null = st.behaviour_gap(steady_grower[0]["price"], steady_grower[0]["flow"], MPY)
    assert g_bb["gap"] > 0.02              # average dollar lags by >2 pts/yr
    assert g_bb["gap"] > g_null["gap"] + 0.04  # and far more than the null


def test_cashflow_construction_signs(boom_bust):
    """Subscriptions are negative to the investor; the final cashflow includes the
    redemption of the terminal value (so the last entry is lifted by it)."""
    frame, _ = boom_bust
    cf = st.cashflows_from_flows(frame["price"], frame["flow"])
    assert cf[:-1].mean() < 0          # ongoing net subscriptions are negative
    assert cf[-1] > 0                  # terminal redemption dominates the last period


# ---- SPINE 2: momentum overlay only works with planted persistence --------
def test_momentum_overlay_beats_null_only_with_momentum(trending_daily, random_daily):
    """A time-series-momentum overlay clears the inference bar on a tape with planted
    return persistence, and stays below it on a no-momentum tape (the null)."""
    def t_of(close):
        sig = st.ts_momentum_signal(close, lookback=1)
        daily = st.overlay_returns(close, sig, cost_bps=0.0)
        return st.summarize_overlay(daily, "gross", TPY)["hac_t"]
    t_mom = t_of(trending_daily["close"])
    t_null = t_of(random_daily["close"])
    assert t_mom > 2.0          # clears the inference bar on planted momentum
    assert abs(t_null) < 2.0    # noise stays below the bar


# ---- inference + invariants -----------------------------------------------
def test_hac_tstat_on_pure_noise_is_small():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 5000)
    assert abs(st.hac_tstat(x)) < 3.0


def test_block_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(1)
    x = rng.normal(0.001, 0.02, 3000)
    lo, hi = st.block_bootstrap_ci(x, block=21, n_boot=500)
    assert lo < x.mean() < hi


def test_execution_lag_is_one_shift(trending_daily):
    """Position on day t must equal the signal of day t-1 (one documented lag)."""
    close = trending_daily["close"]
    sig = st.ma_crossover_signal(close, 20, 100)
    daily = st.overlay_returns(close, sig, cost_bps=0.0)
    assert np.allclose(daily["position"].to_numpy()[1:], sig.shift(1).fillna(0.0).to_numpy()[1:])


def test_costs_lower_net_return(trending_daily):
    close = trending_daily["close"]
    sig = st.ma_crossover_signal(close, 20, 100)
    g = st.summarize_overlay(st.overlay_returns(close, sig, cost_bps=0.0), "net", TPY)["mean_bps"]
    n = st.summarize_overlay(st.overlay_returns(close, sig, cost_bps=10.0), "net", TPY)["mean_bps"]
    assert g > n


def test_excess_sharpe_is_symmetric_subtraction():
    rng = np.random.default_rng(2)
    s = rng.normal(0.0005, 0.01, 2000)
    b = rng.normal(0.0004, 0.012, 2000)
    out = st.excess_sharpe(s, b, rf_daily=0.0001)
    assert np.isfinite(out["strat_excess_sharpe"])
    assert abs(out["diff"] - (out["strat_excess_sharpe"] - out["bench_excess_sharpe"])) < 1e-9


def test_mean_reversion_signal_is_long_flat(random_daily):
    sig = st.mean_reversion_signal(random_daily["close"], lookback=5, z=1.0)
    assert set(np.unique(sig.to_numpy())) <= {0.0, 1.0}
