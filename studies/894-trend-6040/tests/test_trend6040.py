"""Offline, fixed-seed tests for the trend-overlay machinery.

The synthetic panel is deterministic; the 200-day signal is point-in-time (one shift, no
look-ahead); the overlay recovers a planted drawdown/Sharpe benefit and stays flat on the
null; switching costs and tax drag reduce the net; the inference primitives behave. All
offline — the whole suite passes with NO real cache present. One real-cache test is
skipped when the cache is absent.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

import pytest  # noqa: E402

from trend6040 import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic world determinism + shape
# --------------------------------------------------------------------------- #
def test_world_deterministic(planted_world):
    p2 = data.synthetic_prices(edge=1.0, seed=894, n_days=6000)
    assert np.allclose(planted_world.to_numpy(), p2.to_numpy())
    assert list(planted_world.columns) == ["SPY", "IEF", "BIL"]


def test_index_no_overflow(planted_world):
    # PeriodIndex/RangeIndex trap: a business-day index must stay well below year 2262.
    assert planted_world.index.max().year < 2100


# --------------------------------------------------------------------------- #
# No look-ahead — the signal is lagged by exactly one day
# --------------------------------------------------------------------------- #
def test_signal_is_point_in_time():
    close = pd.Series(
        np.linspace(100.0, 130.0, 260) + np.sin(np.arange(260)),
        index=pd.bdate_range("2020-01-01", periods=260), name="X",
    )
    ma = st.moving_average(close, 200)
    raw = (close >= ma).astype(float).where(ma.notna())
    sig = st.trend_signal(close, 200)
    # sig at t equals the un-lagged rule at t-1 (one-day lag, no look-ahead)
    assert np.allclose(sig.iloc[205], raw.iloc[204], equal_nan=True)
    assert np.isnan(sig.iloc[0])  # first day cannot be known


def test_no_lookahead_warmup_is_nan(planted_world):
    # The signal is undefined (NaN) until the 200-day MA is defined: warmup rows carry
    # NaN, not a silent in/out default, so the overlay window starts only once the MA is
    # real. Row 200 (0-indexed) is the first with a defined lagged signal.
    sig = st.trend_signal(planted_world["SPY"], 200)
    assert sig.iloc[:200].isna().all()
    assert not np.isnan(sig.iloc[200])


# --------------------------------------------------------------------------- #
# Planted world — the overlay recovers a real drawdown + Sharpe benefit
# --------------------------------------------------------------------------- #
def test_planted_cuts_drawdown(planted_world):
    d = st.synthetic_detect(planted_world)
    # the mechanical benefit: the filter ducks the deep bears -> shallower max DD
    assert d["dd_cut"] > 0.10
    assert d["maxdd_strat"] > d["maxdd_bench"]  # less negative = shallower


def test_planted_beats_null_sharpe(planted_world, null_world):
    dp = st.synthetic_detect(planted_world)
    dn = st.synthetic_detect(null_world)
    # planted must show a bigger DD cut AND a higher Sharpe advantage than the null
    assert dp["dd_cut"] > dn["dd_cut"]
    assert dp["sharpe_adv"] > dn["sharpe_adv"]


def test_planted_multiseed_positive_advantage():
    advs = np.array([
        st.synthetic_detect(data.synthetic_prices(edge=1.0, seed=894 + s, n_days=5000))["sharpe_adv"]
        for s in range(10)
    ])
    assert advs.mean() > 0.15          # a real average Sharpe pickup when bears exist
    assert (advs > 0).sum() >= 8       # robust across seeds


def test_null_no_systematic_advantage():
    advs = np.array([
        st.synthetic_detect(data.synthetic_prices(edge=0.0, seed=894 + s, n_days=5000))["sharpe_adv"]
        for s in range(10)
    ])
    # with no bear to duck, the overlay should not systematically out-Sharpe the static book
    assert abs(advs.mean()) < 0.15


# --------------------------------------------------------------------------- #
# Costs & tax reduce the net
# --------------------------------------------------------------------------- #
def test_switching_costs_reduce_net(planted_world):
    a = st.excess_race(st.trend_overlay(planted_world, cost_bps=0.0), "r_net")
    b = st.excess_race(st.trend_overlay(planted_world, cost_bps=5.0), "r_net")
    assert b["cagr_strat"] < a["cagr_strat"]
    assert b["sharpe_adv"] < a["sharpe_adv"]


def test_tax_drag_reduces_net(planted_world):
    a = st.excess_race(st.trend_overlay(planted_world, cost_bps=3.0, tax_rate=0.0), "r_net")
    b = st.excess_race(st.trend_overlay(planted_world, cost_bps=3.0, tax_rate=0.30), "r_net")
    assert b["cagr_strat"] < a["cagr_strat"]
    assert b["sharpe_adv"] < a["sharpe_adv"]


def test_tax_only_on_gains():
    # a leg that only ever loses while invested pays no exit tax
    sig = np.array([1, 1, 1, 0, 0], dtype=float)
    r_down = np.array([-0.01, -0.02, -0.01, 0.0, 0.0])
    assert st._exit_tax(sig, r_down, 0.30).sum() == 0.0
    r_up = np.array([0.02, 0.03, 0.01, 0.0, 0.0])
    assert st._exit_tax(sig, r_up, 0.30).sum() > 0.0


# --------------------------------------------------------------------------- #
# Inference primitives behave
# --------------------------------------------------------------------------- #
def test_newey_west_matches_one_sample_on_iid():
    x = np.random.default_rng(0).normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=10) - st.one_sample_t(x)) < 0.6


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi


def test_welch_sign():
    a = np.random.default_rng(1).normal(1.0, 1.0, 500)
    b = np.random.default_rng(2).normal(0.0, 1.0, 500)
    assert st.welch_t(a, b) > 2.0


def test_bootstrap_ci_brackets_point(planted_world):
    bt = st.trend_overlay(planted_world, cost_bps=3.0)
    bs = st.sharpe_adv_bootstrap(bt, which="r_net", n_boot=500, seed=1)
    assert bs["lo"] <= bs["adv"] <= bs["hi"]
    assert 0.0 <= bs["p_pos"] <= 1.0


def test_era_cut_partitions(planted_world):
    bt = st.trend_overlay(planted_world, cost_bps=3.0)
    ec = st.era_cut(bt, split="2013-01-01", which="r_net")
    assert ec["early"] is not None and ec["late"] is not None
    assert ec["early"]["n_days"] + ec["late"]["n_days"] == len(bt)


def test_calendar_year_table(planted_world):
    bt = st.trend_overlay(planted_world, cost_bps=3.0)
    cy = st.calendar_year_table(bt, "r_net")
    assert {"overlay_%", "static_%", "diff_pp"}.issubset(cy.columns)
    assert len(cy) > 5


# --------------------------------------------------------------------------- #
# Real-cache test — skipped when the cache is absent (offline CI)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not data.have_real(), reason="real ETF cache absent (offline CI)")
def test_real_tape_drawdown_cut():
    px = data.load_prices()
    bt = st.trend_overlay(px, cost_bps=3.0, tax_rate=0.0)
    rc = st.excess_race(bt, "r_net")
    # the documented real-tape fact: the overlay materially cuts max drawdown
    assert rc["dd_cut"] > 0.10
    assert rc["maxdd_strat"] > rc["maxdd_bench"]
    assert rc["n_days"] > 4000
