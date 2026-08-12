"""Offline, fixed-seed tests for the CLO-AAA-carry machinery.

The synthetic world is deterministic; a planted excess-of-cash carry is recovered (positive
excess Sharpe, HAC t clears 2, bootstrap CI clear of zero); the null shows nothing; a high-vol
"duration" decoy that earns no excess ranks BELOW the steady carry on Sharpe; excess-of-cash
subtracts the cash leg; a young leg is graded on its own valid window (no zero-padding);
costs reduce the net and more churn reduces it further; the inference primitives behave. All
offline. A real-cache smoke test is gated on the parquet/csv existing.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from clo_aaa import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic world + detector
# --------------------------------------------------------------------------- #
def test_world_deterministic():
    a = data.synthetic_world(carry_annual=0.012, seed=888)
    b = data.synthetic_world(carry_annual=0.012, seed=888)
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_planted_carry_recovered(carry_world):
    s = st.carry_stats(carry_world, "CARRY", cash="BIL", n_boot=800)
    assert s["excess_ann_pct"] > 0.5          # the planted ~1.2%/yr carry shows up
    assert s["sharpe"] > 0.8                    # steady low-vol carry -> high excess Sharpe
    assert s["t_hac"] > 2.0                     # HAC t clears the bar
    assert s["sharpe_lo"] > 0.0                 # bootstrap CI keeps clear of zero


def test_null_world_flat(null_world):
    # A single finite sample of zero-mean noise can wander to a middling raw Sharpe by luck,
    # so the honest null property is that the HAC t does NOT clear significance for THIS seed
    # and, across seeds, the detector essentially never fires (matches results.md: 0/12).
    s = st.carry_stats(null_world, "CARRY", cash="BIL", n_boot=800)
    assert abs(s["t_hac"]) < 2.0                # no carry -> HAC t must not fire
    fires = 0
    for sd in range(12):
        w = data.synthetic_world(carry_annual=0.0, seed=888 + sd).rename(
            columns={"cash": "BIL", "carry": "CARRY", "dur": "DUR"})
        if abs(st.carry_stats(w, "CARRY", cash="BIL", n_boot=200)["t_hac"]) >= 2.0:
            fires += 1
    assert fires <= 1                           # at most a lone false positive in 12 seeds


def test_duration_decoy_ranks_below_carry(carry_world):
    # The high-vol "dur" leg earns ~no excess but is far more volatile: the excess-Sharpe
    # race must rank the steady carry ABOVE it.
    tbl = st.race(carry_world, ["CARRY", "DUR"], cash="BIL", n_boot=400)
    assert tbl.index[0] == "CARRY"
    assert tbl.loc["CARRY", "sharpe"] > tbl.loc["DUR", "sharpe"]
    assert tbl.loc["CARRY", "vol_ann_pct"] < tbl.loc["DUR", "vol_ann_pct"]


# --------------------------------------------------------------------------- #
# Excess-of-cash mechanics
# --------------------------------------------------------------------------- #
def test_excess_subtracts_and_drops_cash():
    ret = pd.DataFrame(
        {"A": [0.01, 0.02, 0.03], "BIL": [0.001, 0.001, 0.001]},
        index=pd.bdate_range("2021-01-04", periods=3),
    )
    ex = st.excess_returns(ret, cash="BIL")
    assert "BIL" not in ex.columns
    assert np.allclose(ex["A"].to_numpy(), [0.009, 0.019, 0.029])


def test_young_leg_graded_on_own_window():
    # A leg with leading NaNs (a young ETF) must be graded only on its valid rows, not
    # padded with zeros that would deflate its mean and Sharpe.
    idx = pd.bdate_range("2021-01-04", periods=100)
    young = np.full(100, np.nan)
    young[60:] = 0.0005                          # 40 valid days of a steady carry
    ret = pd.DataFrame({"YOUNG": young, "BIL": np.zeros(100)}, index=idx)
    s = st.carry_stats(ret, "YOUNG", cash="BIL", n_boot=200)
    assert s["n"] == 40
    assert s["excess_ann_pct"] > 0


# --------------------------------------------------------------------------- #
# Tradability — costs bite
# --------------------------------------------------------------------------- #
def test_costs_reduce_net(carry_world):
    gross = st.costed_carry(carry_world, "CARRY", cash="BIL", spread_bps_oneway=0.0)["net_ann_pct"]
    net = st.costed_carry(carry_world, "CARRY", cash="BIL",
                          spread_bps_oneway=3.0, rebalances_per_year=12.0)["net_ann_pct"]
    assert net < gross


def test_more_churn_costs_more(carry_world):
    lo = st.costed_carry(carry_world, "CARRY", cash="BIL", rebalances_per_year=1.0)["net_ann_pct"]
    hi = st.costed_carry(carry_world, "CARRY", cash="BIL", rebalances_per_year=24.0)["net_ann_pct"]
    assert hi < lo


def test_relative_trade_pays_for_its_short(carry_world):
    r = st.relative_trade(carry_world, leg="CARRY", short="DUR",
                          borrow_annual_bps=40.0, spread_bps_oneway=3.0)
    assert r["charge_ann_pct"] > 0
    assert r["net_ann_pct"] < r["gross_ann_pct"]


# --------------------------------------------------------------------------- #
# Head-to-head + primitives
# --------------------------------------------------------------------------- #
def test_head_to_head_equals_leg_minus_bench(carry_world):
    h = st.head_to_head(carry_world, "CARRY", "DUR", cash="BIL")
    sub = carry_world[["CARRY", "DUR"]].dropna()
    expect = (sub["CARRY"] - sub["DUR"]).mean() * st.TRADING_DAYS * 100
    assert abs(h["diff_ann_pct"] - expect) < 1e-9


def test_drawdown_is_sane(carry_world):
    dd = st.carry_stats(carry_world, "CARRY", cash="BIL", n_boot=200)["maxdd_pct"]
    assert -100.0 < dd <= 0.0


def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0002, 0.005, 3000)
    assert abs(st.newey_west_t(x, lags=10) - st.one_sample_t(x)) < 0.6


# --------------------------------------------------------------------------- #
# Real-cache smoke test (skipped when the cache is absent, e.g. on CI)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.exists(data.PRICES_CACHE),
                    reason="real _cache/clo_prices.csv absent (offline CI)")
def test_real_cache_shape_and_sign():
    prices = data.load_prices()
    for t in data.TICKERS:
        assert t in prices.columns
    assert prices.index.max() <= pd.Timestamp(data.AS_OF)
    ret = data.daily_returns(prices)
    j = st.carry_stats(ret, "JAAA", cash="BIL")
    assert j["n"] > 1000
    assert j["excess_ann_pct"] > 0             # AAA-CLO carry is positive over the sample
    assert j["vol_ann_pct"] < 5.0              # floating-rate senior tranche: tiny vol
