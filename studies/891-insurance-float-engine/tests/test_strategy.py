"""Offline, fixed-seed tests for the insurance-float machinery.

The synthetic world is deterministic; a planted risk-adjusted edge is recovered (positive
excess-vs-excess Sharpe advantage, positive CAPM alpha with a real *t*); the null shows
nothing; the two-factor decomposition always finds the planted financial-sector (bank)
loading; the rotation signal is point-in-time (one shift, no look-ahead); costs reduce the
net isolation spread; and the inference primitives behave. All offline — no real cache needed.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from insurance_float import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Determinism
# --------------------------------------------------------------------------- #
def test_world_deterministic(edge_world):
    w2 = data.synthetic_world(edge_ann=0.04, seed=891, n_months=240)
    for c in edge_world.columns:
        assert np.allclose(edge_world[c].to_numpy(), w2[c].to_numpy())


def test_world_index_no_overflow(edge_world):
    # PeriodRange->timestamp must stay well below the pandas ns horizon (~year 2262).
    assert edge_world.index.max().year < 2100
    assert edge_world.index.is_monotonic_increasing


# --------------------------------------------------------------------------- #
# Planted edge recovered; null silent
# --------------------------------------------------------------------------- #
def test_planted_edge_recovered(edge_world):
    d = st.synthetic_detect(edge_world)
    assert d["advantage"] > 0.15          # planted edge lifts the excess Sharpe over market
    assert d["capm_alpha_ann_pct"] > 1.5  # a real positive CAPM alpha
    assert d["capm_t_alpha"] > 2.0        # ...and it's significant
    # even controlling for the bank factor, a genuine float edge survives here
    assert d["two_alpha_ann_pct"] > 1.0


def test_null_world_no_signal(null_world):
    d = st.synthetic_detect(null_world)
    assert abs(d["advantage"]) < 0.15     # no risk-adjusted edge over market
    assert abs(d["capm_t_alpha"]) < 2.0   # CAPM alpha not distinguishable from zero
    assert abs(d["two_t_alpha"]) < 2.0    # and the two-factor alpha is dead too


def test_bank_loading_always_present(edge_world, null_world):
    # The financial-sector confound is planted in BOTH worlds and must always be detected.
    for w in (edge_world, null_world):
        two = st.decompose_financials(w, "KIE")
        assert two["load_bank"] > 0.1
        assert two["t_load_bank"] > 3.0


# --------------------------------------------------------------------------- #
# Race / Sharpe sanity
# --------------------------------------------------------------------------- #
def test_race_advantage_is_sharpe_difference(edge_world):
    r = st.sharpe_race(edge_world, "KIE")
    assert abs(r["advantage"] - (r["sharpe_ins"] - r["sharpe_mkt"])) < 1e-12
    assert r["n"] == len(edge_world)


def test_bootstrap_ci_brackets_point(edge_world):
    b = st.bootstrap_sharpe(edge_world, "KIE", n_boot=500)
    assert b["ci_low"] <= b["sharpe"] <= b["ci_high"]


# --------------------------------------------------------------------------- #
# No-lookahead rebalance lag
# --------------------------------------------------------------------------- #
def test_rotation_signal_is_point_in_time():
    ret = pd.DataFrame(
        {"KIE": np.linspace(-0.02, 0.03, 30), "SPY": np.linspace(0.01, -0.01, 30)},
        index=pd.period_range("2010-01", periods=30, freq="M").to_timestamp(how="end"),
    )
    sig = st.rotation_signal(ret, "KIE", "SPY", lookback=6)
    # Reconstruct the un-shifted comparison and confirm the signal at t equals it at t-1.
    trail_ins = (1 + ret["KIE"]).rolling(6).apply(np.prod, raw=True)
    trail_mkt = (1 + ret["SPY"]).rolling(6).apply(np.prod, raw=True)
    raw_cmp = trail_ins > trail_mkt
    assert bool(sig.iloc[10]) == bool(raw_cmp.iloc[9])


def test_costs_reduce_isolation_net(edge_world):
    free = st.isolation_trade(edge_world, "KIE", cost_bps_oneway=0.0, borrow_annual_bps=0.0)
    costed = st.isolation_trade(edge_world, "KIE", cost_bps_oneway=5.0, borrow_annual_bps=40.0)
    assert costed["net_ann_pct"] < free["net_ann_pct"]
    assert costed["charge_ann_pct"] > 0.0


def test_rotation_costs_charged(edge_world):
    r = st.rotation_strategy(edge_world, "KIE", cost_bps=5.0)
    assert 0.0 <= r["share_ins"] <= 1.0
    assert r["switches"] >= 0


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_nw_mean_t_matches_plain_t_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.004, 0.04, 3000)
    _, t_nw = st.nw_mean_t(x, lags=6)
    plain = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    assert abs(t_nw - plain) < 0.5


def test_nw_ols_recovers_known_beta():
    rng = np.random.default_rng(1)
    x = rng.normal(0, 0.04, 2000)
    y = 0.001 + 1.3 * x + rng.normal(0, 0.01, 2000)
    beta, se, r2 = st._nw_ols(y, np.column_stack([np.ones(len(y)), x]), lags=6)
    assert abs(beta[1] - 1.3) < 0.05
    assert 0.0 < r2 < 1.0


def test_ann_stats_maxdd_nonpositive(edge_world):
    s = st.ann_stats(edge_world, "KIE")
    assert s["maxdd_pct"] <= 0.0
    assert s["vol_pct"] > 0.0


# --------------------------------------------------------------------------- #
# Real-cache test — only runs if a real _cache/ parquet is present (absent on CI)
# --------------------------------------------------------------------------- #
import pytest  # noqa: E402


@pytest.mark.skipif(not os.path.exists(data.PRICES_CACHE),
                    reason="no real _cache/ parquet (offline / CI)")
def test_real_cache_shape():
    prices = data.load_prices()
    ret = data.monthly_returns(prices)
    for t in data.TICKERS:
        assert t in ret.columns
    assert ret.index.max() <= pd.Timestamp(data.AS_OF)
    assert len(ret) > 150  # ~19 years of months
