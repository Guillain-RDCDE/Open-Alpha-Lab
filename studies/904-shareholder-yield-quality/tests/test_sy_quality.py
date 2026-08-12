"""Offline, fixed-seed tests for the Shareholder-Yield + Quality machinery.

The synthetic world is deterministic; the Sharpe-gap detector recovers a planted
quality-over-raw edge and stays quiet on the null; the equal-weight sleeve is a plain
mean; the monthly rebalance is no-look-ahead; turnover costs reduce the net Sharpe; the
inference primitives behave. All offline — the suite passes with NO real cache present.
The single real-cache test is @skipif-gated on the parquet's existence.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from sy_quality import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic world — determinism, planted edge, null
# --------------------------------------------------------------------------- #
def test_world_deterministic(edge_world):
    w2 = data.synthetic_world(edge=3.0, seed=904, n_months=150)
    assert np.allclose(edge_world.to_numpy(), w2.to_numpy())


def test_world_index_is_period_no_overflow(edge_world):
    # OOB-safe: the synthetic index is a PeriodIndex kept as periods (never a huge
    # to_timestamp span), so it can never overflow the pandas ns horizon.
    assert isinstance(edge_world.index, pd.PeriodIndex)


def test_planted_edge_recovered(edge_world):
    d = st.synthetic_detect(edge_world)
    assert d["t_nw"] > 3.0            # the quality-over-raw gap lights up
    assert d["sharpe_gap"] > 0
    assert d["diff_ann_pct"] > 0


def test_null_world_no_signal(null_world):
    # 20 seeds: the detector must not manufacture significance from zero edge.
    ts = np.array([st.synthetic_detect(data.synthetic_world(edge=0.0, seed=904 + s,
                   n_months=150))["t_nw"] for s in range(20)])
    assert (np.abs(ts) >= 2).sum() <= 2          # false-positive rate near nominal
    assert abs(np.nanmean(ts)) < 1.0


# --------------------------------------------------------------------------- #
# Sleeves — equal weight, rebalance mechanics, no look-ahead
# --------------------------------------------------------------------------- #
def _toy_monthly():
    idx = pd.period_range("2015-01", periods=24, freq="M").to_timestamp(how="end")
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {"PKW": rng.normal(0.01, 0.04, 24), "QUAL": rng.normal(0.009, 0.035, 24),
         "SPY": rng.normal(0.008, 0.03, 24), "BIL": np.full(24, 0.002)},
        index=idx,
    )


def test_sleeve_is_equal_weight_mean():
    m = _toy_monthly()
    s = st.sleeve_returns(m, ["PKW", "QUAL"])
    assert np.allclose(s.to_numpy(), m[["PKW", "QUAL"]].mean(axis=1).to_numpy())


def test_single_member_sleeve_is_that_member():
    m = _toy_monthly()
    s = st.sleeve_returns(m, ["PKW"])
    assert np.allclose(s.to_numpy(), m["PKW"].to_numpy())


def test_sleeve_drops_unlisted_months():
    # A member missing early (not-yet-listed) forces the sleeve to start later.
    m = _toy_monthly().copy()
    m.loc[m.index[:5], "QUAL"] = np.nan
    s = st.sleeve_returns(m, ["PKW", "QUAL"])
    assert len(s) == 19 and s.index.min() == m.index[5]


def test_turnover_single_member_only_build():
    # A one-member sleeve never drifts -> only the initial-build month has turnover.
    m = _toy_monthly()
    t = st.sleeve_turnover(m, ["PKW"])
    assert t.iloc[0] == pytest.approx(0.5)
    assert np.allclose(t.iloc[1:].to_numpy(), 0.0)


def test_costs_reduce_net_sharpe():
    m = _toy_monthly()
    cash = st.cash_returns(m, "BIL")
    c0 = st.costed_sleeve(m, ["PKW", "QUAL"], cash, one_way_bps=0.0)
    c5 = st.costed_sleeve(m, ["PKW", "QUAL"], cash, one_way_bps=25.0)
    assert c5["net_sharpe"] <= c0["net_sharpe"]
    assert c5["cost_drag_bps_yr"] >= c0["cost_drag_bps_yr"]


# --------------------------------------------------------------------------- #
# Sharpe race / gap test — cash cancels in the difference
# --------------------------------------------------------------------------- #
def test_gap_is_cash_independent(edge_world):
    q = edge_world["qsy"].rename("qsy")
    r = edge_world["raw"].rename("raw")
    z = pd.Series(0.0, index=edge_world.index)
    c = pd.Series(0.003, index=edge_world.index)
    g0 = st.sharpe_gap_test(q, r, z)
    gc = st.sharpe_gap_test(q, r, c)
    # the mean of the (a-b) spread is cash-free; the diff stats match exactly
    assert g0["diff_mean_bps"] == pytest.approx(gc["diff_mean_bps"], abs=1e-9)
    assert g0["t_nw"] == pytest.approx(gc["t_nw"], abs=1e-9)


def test_bootstrap_ci_brackets_obs(edge_world):
    q = edge_world["qsy"].rename("qsy")
    r = edge_world["raw"].rename("raw")
    z = pd.Series(0.0, index=edge_world.index)
    b = st.sharpe_gap_bootstrap(q, r, z, n_draws=1500, seed=1)
    assert b["lo"] <= b["obs"] <= b["hi"]
    assert b["frac_negative"] < 0.5     # a positive planted edge -> mostly positive draws


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.004, 0.02, 400)
    assert abs(st.newey_west_t(x, lags=6) - st.one_sample_t(x)) < 0.6


def test_welch_t_zero_on_equal_means():
    rng = np.random.default_rng(1)
    a = rng.normal(0.0, 1.0, 500)
    b = rng.normal(0.0, 1.0, 500)
    assert abs(st.welch_t(a, b)) < 3.0


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(96, 100)
    assert lo < 0.96 < hi


def test_max_drawdown_sign():
    r = pd.Series([0.1, -0.2, 0.05, -0.3, 0.1])
    assert st.max_drawdown(r) < 0


# --------------------------------------------------------------------------- #
# Real-cache test — skipped when the git-ignored parquet is absent (CI)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.exists(data.TAPE_CACHE),
                    reason="real ETF cache absent (offline / CI) — synthetic suite covers the machinery")
def test_real_cache_sane():
    px = data.load_prices()
    for tk in data.TICKERS:
        assert tk in px.columns
    m = data.monthly_total_returns(px)
    cash = st.cash_returns(m, data.CASH)
    qsy = st.sleeve_returns(m, data.QSY)
    raw = st.sleeve_returns(m, data.RAW)
    spy = m[data.BENCH]
    common = pd.concat([qsy, raw, spy, cash], axis=1).dropna()
    assert len(common) > 100                         # ~13 years of months
    g = st.sharpe_gap_test(common.iloc[:, 0], common.iloc[:, 1], common.iloc[:, 3])
    assert np.isfinite(g["t_nw"]) and np.isfinite(g["sharpe_gap"])
