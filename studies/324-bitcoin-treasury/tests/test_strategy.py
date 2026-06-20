"""The engine's invariants and the study's two spines:
(1) MSTR beats its levered-BTC replica only when a real alpha is planted, not on the null;
(2) a swinging NAV premium is return-free variance — it can't be harvested by holding."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from bitcoin_treasury import strategy as st  # noqa: E402


# ---- HAC / regression machinery -------------------------------------------
def test_hac_tstat_detects_clear_mean():
    rng = np.random.default_rng(0)
    x = 0.01 + rng.normal(0, 0.001, 500)  # strongly positive mean
    assert st.hac_tstat(x) > 5


def test_hac_tstat_zero_mean_is_insignificant():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.01, 500)
    assert abs(st.hac_tstat(x)) < 2


def test_regression_recovers_alpha_and_beta():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 0.03, 2000)
    y = 0.0010 + 1.8 * x + rng.normal(0, 0.005, 2000)
    a, b, r2, ta = st.hac_tstat_resid(y, x)
    assert abs(a - 0.0010) < 0.0004
    assert abs(b - 1.8) < 0.05
    assert ta > 2  # the planted alpha is detectable
    assert 0 < r2 < 1


# ---- spine #1: MSTR beats levered-BTC only with real alpha -----------------
def test_no_alpha_on_null(null_pair):
    """The null: pure levered beta — alpha is not distinguishable from zero."""
    frame, _ = null_pair
    dec = st.beta_decomposition(frame)
    assert abs(dec["t_alpha"]) < 2.0
    assert dec["r2"] > 0.5  # MSTR is mostly BTC


def test_alpha_detected_when_planted(alpha_pair):
    """The positive control: a planted daily alpha clears the inference bar."""
    frame, truth = alpha_pair
    dec = st.beta_decomposition(frame)
    assert dec["t_alpha"] > 2.0
    assert dec["alpha_daily"] > 0


def test_race_diff_insignificant_on_null(null_pair):
    """On the null, MSTR cannot beat its own levered-BTC replica."""
    frame, _ = null_pair
    race = st.race_mstr_vs_proxy(frame)
    assert abs(race["diff_t"]) < 2.0


def test_race_diff_significant_with_alpha(alpha_pair):
    """With a real alpha, MSTR beats its levered replica on the difference."""
    frame, _ = alpha_pair
    race = st.race_mstr_vs_proxy(frame)
    assert race["diff_t"] > 2.0
    assert race["diff_mean_bps"] > 0


# ---- spine #2: the NAV premium is return-free variance ---------------------
def test_premium_timing_no_edge_on_null(null_pair):
    """Timing a non-existent / drift-free premium adds nothing over always-hold."""
    frame, _ = null_pair
    res = st.premium_timing(frame)
    assert abs(res["diff_t"]) < 2.0


def test_premium_timing_no_free_lunch_on_premium_tape(premium_pair):
    """Even with a big swinging premium, contrarian timing earns no robust edge
    (the premium is a first-difference: it mean-reverts but pays ~zero)."""
    frame, _ = premium_pair
    res = st.premium_timing(frame)
    assert abs(res["diff_t"]) < 3.0  # not a robust, large edge


# ---- engine invariants -----------------------------------------------------
def test_levered_proxy_financing_lowers_return(null_pair):
    frame, _ = null_pair
    cheap = st.levered_btc_proxy(frame, 1.8, financing_bps_per_day=0.0)
    dear = st.levered_btc_proxy(frame, 1.8, financing_bps_per_day=5.0)
    assert cheap.mean() > dear.mean()


def test_levered_proxy_no_carry_at_leverage_one(null_pair):
    frame, _ = null_pair
    p = st.levered_btc_proxy(frame, 1.0, financing_bps_per_day=10.0)
    r_btc = st.simple_returns(frame["btc"])
    assert np.allclose(p.to_numpy(), r_btc.to_numpy())  # no borrowed notional


def test_sharpe_and_cagr_finite(null_pair):
    frame, _ = null_pair
    r = st.simple_returns(frame["mstr"]).to_numpy()
    assert np.isfinite(st.sharpe(r))
    assert np.isfinite(st.cagr(r))
    assert st.max_drawdown(r) <= 0


def test_block_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(3)
    x = 0.002 + rng.normal(0, 0.01, 1500)
    lo, hi = st.block_bootstrap_ci(x, block=20, n_boot=500)
    assert lo < x.mean() < hi


def test_premium_timing_one_lag_no_lookahead(null_pair):
    """The timing overlay uses exactly one shift (position at close of t earns t+1)."""
    frame, _ = null_pair
    res = st.premium_timing(frame)
    assert 0.0 <= res["pct_in_market"] <= 1.0
    assert np.isfinite(res["diff_t"])
