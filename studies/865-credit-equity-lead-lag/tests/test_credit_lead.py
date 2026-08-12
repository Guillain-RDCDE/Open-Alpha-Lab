"""Offline, fixed-seed tests for the credit→equity lead-lag machinery.

The synthetic panel is deterministic; the credit trend has the right sign vs the latent
factor; the Granger-style predictive regression recovers a planted one-week lead and stays
silent on the null; the alignment is point-in-time (trend at week t vs the t+1 return, no
look-ahead); switch costs reduce the overlay's return; the NW-regression HAC t matches a
plain OLS t on iid data; the inference primitives behave. All offline, synthetic-only.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from credit_lead import data, strategy as st  # noqa: E402

REAL_CACHE = data.CACHE_PATH


# --------------------------------------------------------------------------- #
# Synthetic world
# --------------------------------------------------------------------------- #
def test_world_deterministic(edge_world):
    p2 = data.synthetic_panel(edge=0.008, seed=865, n_days=2600)
    for col in edge_world.columns:
        assert np.allclose(edge_world[col].to_numpy(), p2[col].to_numpy())


def test_panel_has_all_tickers(edge_world):
    assert list(edge_world.columns) == ["HYG", "IEF", "LQD", "SPY"]
    assert (edge_world > 0).all().all()   # positive prices


def test_credit_trend_varies(edge_world):
    tr = st.credit_trend(edge_world, lookback_wk=4).dropna()
    assert tr.notna().sum() > 100
    assert tr.std() > 0            # the trend actually varies (the timer has something to read)


# --------------------------------------------------------------------------- #
# The lead-lag detector: fires on the plant, silent on the null
# --------------------------------------------------------------------------- #
def test_planted_lead_recovered(edge_world):
    r = st.leadlag_regression(edge_world, lookback_wk=4)
    assert r["beta_t_nw"] > 4.0        # the predictive slope lights up
    assert r["beta"] > 0               # positive credit trend -> higher next-week SPY
    assert r["per_sd_bps"] > 0
    assert r["r2"] > 0.0


def test_planted_discrimination_and_overlay(edge_world):
    s = st.signal_stats(edge_world, lookback_wk=4)
    assert s["t_nw"] > 2.5
    assert s["on_bps"] > s["off_bps"]  # risk-on weeks earn a HIGHER next-week SPY
    t = st.overlay_stats(edge_world, lookback_wk=4, cost_bps=1.0)
    assert t["active_t_nw"] > 1.5      # the overlay adds value when the plant is real
    assert t["net_sharpe"] > t["bh_sharpe"]


def test_null_world_no_lead(null_world):
    r = st.leadlag_regression(null_world, lookback_wk=4)
    assert abs(r["beta_t_nw"]) < 2.5   # no predictability in the null
    s = st.signal_stats(null_world, lookback_wk=4)
    assert abs(s["t_nw"]) < 2.5


def test_null_control_calibrated():
    # Across seeds the null regression t is centred near zero and rarely fires.
    ts = np.array([
        st.leadlag_regression(data.synthetic_panel(edge=0.0, seed=865 + s, n_days=2000), 4)["beta_t_nw"]
        for s in range(12)
    ])
    assert abs(ts.mean()) < 1.0
    assert (np.abs(ts) >= 2).sum() <= 2


# --------------------------------------------------------------------------- #
# No look-ahead + costs
# --------------------------------------------------------------------------- #
def test_signal_is_point_in_time():
    # A weekly-close panel; the frame's next-week SPY on date t must equal the raw weekly
    # SPY return on the FOLLOWING weekly date (a shift(-1), no look-ahead into the trend).
    idx = pd.bdate_range("2019-01-04", periods=200, freq="C")
    panel = pd.DataFrame(
        {
            "HYG": np.linspace(100, 140, 200),
            "IEF": np.linspace(100, 108, 200),
            "LQD": np.linspace(100, 115, 200),
            "SPY": 100 * np.cumprod(1 + np.linspace(0.001, 0.002, 200)),
        },
        index=idx,
    )
    wr = st.weekly_returns(panel)
    df = st.leadlag_frame(panel, lookback_wk=2)
    t0 = df.index[3]
    loc = wr.index.get_loc(t0)
    nxt = wr.index[loc + 1]
    assert np.isclose(df.loc[t0, "r_spy_next"], wr["SPY"].loc[nxt])


def test_costs_reduce_overlay_return(edge_world):
    gross = st.overlay_stats(edge_world, lookback_wk=4, cost_bps=0.0)["net_cagr"]
    net = st.overlay_stats(edge_world, lookback_wk=4, cost_bps=20.0)["net_cagr"]
    assert net < gross


def test_regime_diff_series_mean_is_difference():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 500)
    flag = (rng.random(500) > 0.5).astype(float)
    g = st._regime_diff_series(flag, x)
    diff = x[flag == 1].mean() - x[flag == 0].mean()
    assert np.isclose(g.mean(), diff)


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_nw_regression_matches_ols_on_iid():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 1, 3000)
    y = 0.3 * x + rng.normal(0, 1, 3000)   # iid residual -> HAC ~ OLS
    beta, t_nw, r2 = st.nw_regression(x, y, lags=6)
    # plain OLS t on the slope
    xc = x - x.mean()
    b = (xc @ (y - y.mean())) / (xc @ xc)
    resid = y - (y.mean() - b * x.mean() + b * x)
    s2 = (resid @ resid) / (len(x) - 2)
    ols_t = b / np.sqrt(s2 / (xc @ xc))
    assert abs(beta - b) < 1e-9
    # HAC t ~ OLS t on iid residuals (within ~15% relative)
    assert abs(t_nw - ols_t) < 0.15 * abs(ols_t)
    assert 0.05 < r2 < 0.12


def test_nw_regression_zero_slope_insignificant():
    rng = np.random.default_rng(2)
    x = rng.normal(0, 1, 2000)
    y = rng.normal(0, 1, 2000)             # no relation
    _, t_nw, _ = st.nw_regression(x, y, lags=6)
    assert abs(t_nw) < 2.5


def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=6) - st.one_sample_t(x)) < 0.6


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi


def test_sharpe_and_cagr_positive_on_updrift():
    r = np.full(520, 0.002)     # steady up-drift
    assert st.sharpe(r) > 0
    assert st.cagr(r) > 0
    assert st.max_drawdown(r) == 0.0    # monotone up -> no drawdown


# --------------------------------------------------------------------------- #
# Real-cache smoke test — guarded so offline CI skips it
# --------------------------------------------------------------------------- #
import pytest  # noqa: E402


@pytest.mark.skipif(not os.path.exists(REAL_CACHE), reason="real cache absent offline CI")
def test_real_cache_loads_and_is_flat():
    panel = data.load_panel()
    assert list(panel.columns) == ["HYG", "IEF", "LQD", "SPY"]
    assert panel.index.max() <= pd.Timestamp(data.AS_OF)
    r = st.leadlag_regression(panel, lookback_wk=4)
    # the honest real-tape finding: no significant (correct-signed) lead of credit over equity
    assert abs(r["beta_t_nw"]) < 2.0
