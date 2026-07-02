"""Offline, deterministic tests for Study 587 (NFT-Floor-Beta)."""
from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from nft_floor_beta import data, strategy as st  # noqa: E402


def test_synthetic_deterministic():
    a, _ = data.synthetic_series(seed=587)
    b, _ = data.synthetic_series(seed=587)
    assert data.fingerprint(a[["eth_ret", "floor_ret"]]) == data.fingerprint(b[["eth_ret", "floor_ret"]])


def test_floors_follow_eth_high_beta():
    df, _ = data.synthetic_series(lead_alpha=0.0, seed=587)
    beta = st.contemporaneous_beta(df, lag=1)
    # NFT floor is a high-beta (>1) lagged echo of ETH
    assert beta["beta"] > 1.0
    assert beta["corr"] > 0.5
    # lead-lag corr peaks at a NEGATIVE lag (floors follow ETH)
    llc = st.lead_lag_corr(df, max_lag=5)
    assert llc.idxmax() < 0


def test_null_has_no_predictive_lead():
    df, _ = data.synthetic_series(lead_alpha=0.0, seed=587)
    mom = st.floor_momentum(df, 14)
    fwd = st.forward_return(df, 5)
    reg = st.predictive_regression(mom, fwd)
    # null world: floor momentum does not forecast forward ETH
    assert abs(reg["hac_t"]) < 2.0


def test_positive_control_seed_robust():
    null_t = st.synthetic_mean_hac_t(data, lead_alpha=0.0, n_seeds=20)
    lead_t = st.synthetic_mean_hac_t(data, lead_alpha=0.10, n_seeds=20)
    assert abs(null_t) < 1.0          # flat at the null
    assert lead_t > 2.0               # a planted lead clears the bar


def test_no_lookahead_forward_window():
    df, _ = data.synthetic_series(seed=587)
    fwd = st.forward_return(df, 5)
    # the last `horizon` rows must be NaN (no full forward window)
    assert fwd.tail(5).isna().all()


def test_overlay_net_below_gross():
    df, _ = data.synthetic_series(lead_alpha=0.0, seed=587)
    ov = st.overlay_backtest(df, lookback=14, cost_bps=10.0)
    assert ov["net_ann"] <= ov["gross_ann"]
    assert 0.0 <= ov["frac_long"] <= 1.0


def test_placebo_pvalue_range():
    df, _ = data.synthetic_series(lead_alpha=0.0, seed=587)
    mom = st.floor_momentum(df, 14)
    fwd = st.forward_return(df, 5)
    p = st.placebo_pvalue(mom, fwd, n_perm=100, seed=587)
    assert 0.0 < p <= 1.0
