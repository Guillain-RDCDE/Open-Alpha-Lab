"""Offline, fixed-seed tests for the agency-MBS-carry machinery.

The synthetic world is deterministic; the duration-neutral estimator recovers a planted
carry with the right beta; the null world shows nothing; costs reduce the net; the era
cut and Sharpe race behave; and the inference primitives are sane. All offline — no real
cache required. A single real-cache smoke test is skipped when _cache/ is absent (CI).
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import pytest  # noqa: E402

from mbs_carry import data, strategy as st  # noqa: E402


def test_world_deterministic():
    a = data.synthetic_world(carry_annual=0.02, seed=886)
    b = data.synthetic_world(carry_annual=0.02, seed=886)
    assert np.allclose(a["mbs"].to_numpy(), b["mbs"].to_numpy())
    assert np.allclose(a["ief"].to_numpy(), b["ief"].to_numpy())


def test_synthetic_index_no_overflow(edge_world):
    # PeriodIndex->timestamp must stay well under the pandas ns horizon (~year 2262).
    assert edge_world.index.max() < pd.Timestamp("2100-01-01")
    assert edge_world.index.is_monotonic_increasing


def test_planted_carry_recovered(edge_world):
    s = st.synthetic_detect(edge_world)
    assert s["t_hac"] > 3.0                      # the planted carry lights up
    assert 1.4 < s["carry_ann_pct"] < 2.6        # ~ the planted +2%/yr
    assert 0.4 < s["beta"] < 0.7                 # recovers beta_true ~ 0.55


def test_null_world_no_carry(null_world):
    s = st.synthetic_detect(null_world)
    assert abs(s["t_hac"]) < 2.5
    assert abs(s["carry_ann_pct"]) < 1.0


def test_bootstrap_ci_clears_zero_on_edge_and_straddles_on_null(edge_world, null_world):
    c_edge, _ = st.carry_series(edge_world)
    c_null, _ = st.carry_series(null_world)
    ci_e = st.mean_ci_bootstrap(c_edge.values)
    ci_n = st.mean_ci_bootstrap(c_null.values)
    assert ci_e["ci_low"] > 0.0                  # planted edge: CI clears zero
    assert ci_n["ci_low"] < 0.0 < ci_n["ci_high"]  # null: CI straddles zero


def test_beta_hedge_neutralises_rate_factor(edge_world):
    # The carry residual must be far less correlated with the Treasury leg than the raw MBS leg.
    carry, _ = st.carry_series(edge_world)
    raw_corr = abs(np.corrcoef(edge_world["mbs"], edge_world["ief"])[0, 1])
    res_corr = abs(np.corrcoef(carry, edge_world["ief"])[0, 1])
    assert res_corr < 0.2
    assert res_corr < raw_corr


def test_costs_reduce_net(edge_world):
    gross = st.carry_stats(edge_world)["carry_ann_pct"]
    net = st.costed_carry(edge_world)["net_ann_pct"]
    assert net < gross


def test_static_beta_differs_from_empirical(edge_world):
    _, b_emp = st.carry_series(edge_world)
    _, b_static = st.carry_series(edge_world, beta=0.80)
    assert b_static == 0.80
    assert abs(b_emp - 0.80) > 0.05


def test_era_cut_shape(edge_world):
    eras = st.era_cut(edge_world, ["2014-01-01", "2020-01-01"])
    assert len(eras) == 3
    assert all(e["n"] >= 8 for e in eras)


def test_sharpe_race_keys(edge_world):
    r = st.sharpe_race(edge_world)
    assert "mbs_sharpe" in r and "ief_sharpe" in r
    assert r["mbs_vol_pct"] > 0 and r["ief_vol_pct"] > 0


def test_calendar_year_and_drawdown(edge_world):
    carry, _ = st.carry_series(edge_world)
    cy = st.calendar_year_table(carry)
    assert len(cy) >= 15
    assert st.max_drawdown(carry) <= 0.0


def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=6) - st.one_sample_t(x)) < 0.6


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi


def test_annualized_sharpe_sign():
    rng = np.random.default_rng(1)
    assert st.annualized_sharpe(rng.normal(0.01, 0.02, 500)) > 0


@pytest.mark.skipif(not os.path.exists(data.PRICES_CACHE),
                    reason="real cache absent (CI / offline)")
def test_real_cache_smoke():
    prices = data.load_prices()
    panel = data.monthly_panel(prices, asof=data.AS_OF)
    ef = data.excess_frame(panel, "MBB")
    s = st.carry_stats(ef)
    assert s["n"] > 100
    assert 0.3 < s["beta"] < 0.9                 # realized MBS duration beta
    assert np.isfinite(s["t_hac"])
