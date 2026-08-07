"""Offline, fixed-seed tests for the Cochrane-Piazzesi machinery.

The synthetic tape is deterministic; the forward-rate identities are exact; the average
excess return is strictly forward-looking (no look-ahead); the regression recovers a
planted forward -> excess-return relation (high in-sample AND out-of-sample R^2) and
stays silent on the null (R^2 ~ 0, OOS R^2 ~ 0); the block placebo separates the two;
the timer costs bite; the inference primitives behave. All offline, no network.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from cp_factor import data, strategy as st  # noqa: E402

CACHE = data.CACHE_PATH


# --------------------------------------------------------------------------- #
# Determinism + schema
# --------------------------------------------------------------------------- #
def test_synthetic_deterministic(planted_world):
    p2 = data.synthetic_daily(edge=0.05, seed=824, n_days=2600)
    assert np.allclose(planted_world.to_numpy(), p2.to_numpy())


def test_synthetic_schema(null_world):
    for c in st.YIELD_COLS + st.ETF_COLS:
        assert c in null_world.columns
    assert (null_world[st.ETF_COLS] > 0).all().all()  # positive prices


# --------------------------------------------------------------------------- #
# Forward-rate identities (exact, hand-checked)
# --------------------------------------------------------------------------- #
def test_forward_rate_identity():
    # y5=4%, y10=5% -> forward 5->10 = 2*y10 - y5 = 6%
    df = pd.DataFrame(
        {"IRX": [1.0], "FVX": [4.0], "TNX": [5.0], "TYX": [5.0]},
        index=pd.bdate_range("2020-01-01", periods=1),
    )
    fwd = st.forward_rates(df)
    assert abs(fwd["f_2"].iloc[0] - 0.06) < 1e-12          # 2*0.05 - 0.04
    assert abs(fwd["y_short"].iloc[0] - 0.01) < 1e-12
    # f_1 = (5*y5 - 0.25*y_short)/4.75
    exp_f1 = (5 * 0.04 - 0.25 * 0.01) / 4.75
    assert abs(fwd["f_1"].iloc[0] - exp_f1) < 1e-12


def test_forwards_use_same_day_only():
    # forward_rates must not shift/leak across rows: row t depends only on yields at t
    df = data.synthetic_daily(edge=0.0, seed=1, n_days=400)
    fwd = st.forward_rates(df)
    df2 = df.copy()
    df2.iloc[100, :4] = df2.iloc[100, :4] * 1.5  # perturb yields on one day
    fwd2 = st.forward_rates(df2)
    changed = (fwd2 != fwd).any(axis=1).to_numpy()
    assert changed[100] and changed.sum() == 1  # only that row moved


# --------------------------------------------------------------------------- #
# No look-ahead in the forward excess return
# --------------------------------------------------------------------------- #
def test_avg_excess_is_forward_looking():
    df = data.synthetic_daily(edge=0.0, seed=2, n_days=800)
    h = 252
    rx = st.avg_excess_return(df, horizon=h)
    assert rx.iloc[-h:].isna().all()          # last horizon rows have no future price
    assert rx.iloc[: len(df) - h].notna().all()
    # hand-check one row against the raw price ratio minus the day-t risk-free
    t = 100
    P = df[st.ETF_COLS].to_numpy()
    rf = df["IRX"].iloc[t] / 100.0
    manual = np.mean(P[t + h] / P[t] - 1.0) - rf
    assert abs(rx.iloc[t] - manual) < 1e-9


def test_no_lookahead_perturb_future_only():
    # perturbing a price at time t must not change avg_rx rows AFTER t-... it can only
    # affect the row t-horizon (whose forward window ends at t). Rows > t are untouched.
    df = data.synthetic_daily(edge=0.0, seed=3, n_days=800)
    h = 252
    base = st.avg_excess_return(df, horizon=h)
    df2 = df.copy()
    df2.iloc[500, 4] *= 1.10  # bump one ETF price at t=500 (SHY col index 4)
    pert = st.avg_excess_return(df2, horizon=h)
    diff = (pert - base).abs() > 1e-12
    # the only affected signal rows are 500 (start) and 500-h (window end)
    affected = set(np.where(diff.to_numpy())[0])
    assert affected <= {500, 500 - h}


# --------------------------------------------------------------------------- #
# The planted relation is recovered; the null is silent
# --------------------------------------------------------------------------- #
def test_planted_relation_recovered(planted_world):
    reg = st.cp_regression(planted_world)
    assert reg["r2"] > 0.3                      # strong in-sample fit
    assert st.oos_r2(planted_world)["oos_r2"] > 0.2   # and it holds out of sample


def test_null_r2_is_silent(null_world):
    reg = st.cp_regression(null_world)
    assert reg["r2"] < 0.02                     # persistent forwards on white target ~ 0
    assert st.oos_r2(null_world)["oos_r2"] < 0.05


def test_planted_beats_null_r2(planted_world, null_world):
    assert st.cp_regression(planted_world)["r2"] > 10 * st.cp_regression(null_world)["r2"]


# --------------------------------------------------------------------------- #
# The block placebo separates planted from null
# --------------------------------------------------------------------------- #
def test_placebo_flags_planted(planted_world):
    pl = st.placebo_r2(planted_world, n_perm=200)
    assert pl["obs_r2"] > pl["placebo_mean_r2"]
    assert pl["p_value"] < 0.05


def test_placebo_null_not_significant(null_world):
    pl = st.placebo_r2(null_world, n_perm=200)
    assert pl["p_value"] > 0.10                 # a white target is not distinguishable


# --------------------------------------------------------------------------- #
# The timer costs bite
# --------------------------------------------------------------------------- #
def test_costs_reduce_net(planted_world):
    lo = st.timer_stats(planted_world, cost_bps=1.0, lookback=600)["net_mean_bps"]
    hi = st.timer_stats(planted_world, cost_bps=20.0, lookback=600)["net_mean_bps"]
    assert hi < lo


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_ols_hac_beta_matches_plain_ols():
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(500), rng.normal(size=(500, 2))])
    beta_true = np.array([0.5, -1.0, 2.0])
    y = X @ beta_true + rng.normal(0, 0.1, 500)
    out = st._ols_hac(X, y, lags=5)
    beta_np = np.linalg.lstsq(X, y, rcond=None)[0]
    assert np.allclose(out["beta"], beta_np, atol=1e-8)
    assert 0.99 < out["r2"] <= 1.0


def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=10) - st.one_sample_t(x)) < 0.6


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi


# --------------------------------------------------------------------------- #
# Real-cache smoke test (guarded — skipped when the cache is absent, e.g. offline CI)
# --------------------------------------------------------------------------- #
import pytest  # noqa: E402


@pytest.mark.skipif(not os.path.exists(CACHE), reason="real cache absent offline CI")
def test_real_cache_shapes():
    df = data.load_panel()
    assert set(st.YIELD_COLS + st.ETF_COLS) <= set(df.columns)
    assert df.index.max() <= pd.Timestamp(data.AS_OF)
    reg = st.cp_regression(df)
    assert reg["n"] > 3000 and np.isfinite(reg["r2"])
