"""Offline, fixed-seed tests for the Treasury-noise machinery.

The synthetic tape is deterministic; the roughness measure is zero on a perfectly smooth
(quadratic) curve and grows with planted noise; the planted noise→forward-return relation
is recovered (negative slope, HAC t clears the bar) for both SPY and credit; the null
shows nothing; the forward-return build is point-in-time (no look-ahead); the OLS+HAC
regression recovers a known slope; the placebo is centred on the null; the timer runs; the
inference primitives behave. All offline, no network.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from treasury_noise import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic world
# --------------------------------------------------------------------------- #
def test_world_deterministic(planted_panel):
    p2 = data.synthetic_daily(edge=0.03, seed=863, n_days=3200)
    assert np.allclose(planted_panel.to_numpy(), p2.to_numpy())


def test_panel_schema(planted_panel):
    for col in data.YIELD_COLS + data.ASSET_TICKERS:
        assert col in planted_panel.columns


# --------------------------------------------------------------------------- #
# The noise (roughness) construction
# --------------------------------------------------------------------------- #
def test_perp_matrix_kills_quadratics():
    # P⊥ annihilates any quadratic-in-maturity curve: residuals are ~0 for a smooth curve.
    P = st._perp_matrix()
    m = data.MATURITIES
    for coeffs in [(1.0, 0.0, 0.0), (0.5, 0.2, 0.0), (3.0, -0.1, 0.02)]:
        a, b, c = coeffs
        y = a + b * m + c * m ** 2
        assert np.allclose(P @ y, 0.0, atol=1e-9)


def test_noise_zero_on_smooth_curve():
    # A curve that is exactly quadratic every day has zero roughness.
    m = data.MATURITIES
    y = 3.0 + 0.2 * m + 0.01 * m ** 2
    idx = pd.bdate_range("2015-01-01", periods=50)
    yields = pd.DataFrame(
        np.tile(y, (50, 1)), index=idx, columns=data.YIELD_COLS
    )
    noise = st.noise_series(yields)
    assert np.allclose(noise.to_numpy(), 0.0, atol=1e-9)


def test_noise_grows_with_planted_level():
    # A synthetic tape with a bigger noise_base yields larger average roughness.
    lo = st.build_daily(data.synthetic_daily(edge=0.0, seed=863, noise_base=0.03))
    hi = st.build_daily(data.synthetic_daily(edge=0.0, seed=863, noise_base=0.12))
    assert hi["noise"].mean() > lo["noise"].mean()


def test_noise_nonnegative(planted_panel):
    d = st.build_daily(planted_panel)
    assert (d["noise"] >= 0).all()


# --------------------------------------------------------------------------- #
# The planted relation is recovered; the null is silent
# --------------------------------------------------------------------------- #
def test_planted_spy_relation_recovered(planted_panel):
    d = st.build_daily(planted_panel)
    hd = st.headline(d, target="ret_spy", horizon=21)
    assert hd["slope_pct"] < 0          # high noise -> lower forward SPY return
    assert hd["t_nw"] < -2.5            # the predictive regression lights up
    assert hd["hi_minus_lo_pct"] < 0    # high-noise third under-earns the low-noise third


def test_planted_credit_relation_recovered(planted_panel):
    d = st.build_daily(planted_panel)
    hd = st.headline(d, target="ret_credit", horizon=21)
    assert hd["slope_pct"] < 0          # high noise -> wider credit (lower HYG-IEF)
    assert hd["t_nw"] < -2.0


def test_null_world_no_signal(null_panel):
    d = st.build_daily(null_panel)
    for tgt in ("ret_spy", "ret_credit"):
        hd = st.headline(d, target=tgt, horizon=21)
        assert abs(hd["t_nw"]) < 2.5    # the detector stays silent on the null


def test_null_placebo_centered(null_panel):
    d = st.build_daily(null_panel)
    pl = st.placebo_pvalue(d, target="ret_spy", horizon=21, n_perm=800)
    assert 0.05 < pl["p_value"] < 0.95           # observed slope inside the null cloud
    assert abs(pl["placebo_mean"]) < abs(pl["placebo_sd"])   # null centred near zero


def test_seed_robust_control():
    null = st.synthetic_mean_t(data, edge=0.0, n_seeds=8, target="ret_spy", horizon=21)
    planted = st.synthetic_mean_t(data, edge=0.03, n_seeds=8, target="ret_spy", horizon=21)
    assert abs(null["mean_t"]) < 1.5             # no false positive on average
    assert planted["mean_t"] < -2.5             # planted edge recovered across seeds
    assert planted["mean_slope_pct"] < null["mean_slope_pct"]


# --------------------------------------------------------------------------- #
# No look-ahead
# --------------------------------------------------------------------------- #
def test_forward_return_is_point_in_time(planted_panel):
    d = st.build_daily(planted_panel)
    fwd1 = st.forward_return(d, target="ret_spy", horizon=1)
    # fwd_1[t] must equal next day's return ret[t+1] (strictly future).
    assert np.allclose(fwd1.to_numpy(), d["ret_spy"].shift(-1).to_numpy(), equal_nan=True)


def test_forward_return_horizon_sum(planted_panel):
    d = st.build_daily(planted_panel)
    fwd3 = st.forward_return(d, target="ret_spy", horizon=3)
    manual = (d["ret_spy"].shift(-1) + d["ret_spy"].shift(-2) + d["ret_spy"].shift(-3))
    assert np.allclose(fwd3.to_numpy(), manual.to_numpy(), equal_nan=True)


# --------------------------------------------------------------------------- #
# The regression + inference primitives
# --------------------------------------------------------------------------- #
def test_predictive_regression_recovers_known_slope():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 3000)
    y = 2.0 - 3.0 * x + rng.normal(0, 1, 3000)
    reg = st.predictive_regression(x, y)
    assert abs(reg["slope"] + 3.0) < 0.1
    assert abs(reg["alpha"] - 2.0) < 0.1
    assert reg["t_nw"] < -10                     # a strong true slope is highly significant
    assert reg["r2"] > 0.8


def test_regression_null_slope_insignificant():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 2000)
    y = rng.normal(0, 1, 2000)                   # y independent of x
    reg = st.predictive_regression(x, y)
    assert abs(reg["t_nw"]) < 2.5


def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=10) - st.one_sample_t(x)) < 0.6


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi


def test_welch_detects_mean_gap():
    rng = np.random.default_rng(2)
    a = rng.normal(1.0, 1.0, 500)
    b = rng.normal(0.0, 1.0, 500)
    assert st.welch_t(a, b) > 5


# --------------------------------------------------------------------------- #
# The timer + robustness plumbing
# --------------------------------------------------------------------------- #
def test_timer_runs_and_costs_hurt(planted_panel):
    d = st.build_daily(planted_panel)
    cheap = st.timer_stats(d, cost_bps=0.0)
    dear = st.timer_stats(d, cost_bps=50.0)
    assert dear["spread_bps_day"] < cheap["spread_bps_day"]   # more cost -> worse net
    assert 0.0 <= dear["invested_frac"] <= 1.0


def test_horizon_sweep_shape(planted_panel):
    d = st.build_daily(planted_panel)
    tab = st.horizon_sweep(d, target="ret_spy", horizons=(5, 21, 63))
    assert list(tab.index) == ["5d", "21d", "63d"]
    assert tab["n"].min() > 100


def test_era_cut_shape(planted_panel):
    d = st.build_daily(planted_panel)
    tab = st.era_cut(d, target="ret_spy", horizon=21, split="2011-01-01")
    assert list(tab.index) == ["early", "late"]


# --------------------------------------------------------------------------- #
# Real cache — only if present (guarded; offline CI skips it)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.exists(data.CACHE_PATH),
                    reason="real cache absent offline CI")
def test_real_cache_loads_and_builds():
    df = data.load_panel()
    d = st.build_daily(df)
    assert len(d) > 1000
    assert (d["noise"] >= 0).all()
    hd = st.headline(d, target="ret_spy", horizon=21)
    assert np.isfinite(hd["t_nw"])
