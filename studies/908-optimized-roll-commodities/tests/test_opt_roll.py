"""Offline, fixed-seed tests for the optimized-roll machinery.

The synthetic world is deterministic; the excess-of-cash frame nets the cash leg; the
Sharpe race recovers a planted roll edge (positive Sharpe advantage, HAC t on the return
difference, bootstrap CI clear of zero) and finds NOTHING at the null; costs reduce the
net advantage; the era cut and calendar-year table are sane; the inference primitives
behave. All offline — no real cache required.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from opt_roll import data, strategy as st  # noqa: E402


def test_world_deterministic(edge_world):
    w2 = data.synthetic_world(roll_edge_annual=0.03, seed=908)
    assert np.allclose(edge_world.to_numpy(), w2.to_numpy())


def test_synthetic_index_is_periodindex(edge_world):
    # OOB-safe: kept as a PeriodIndex, never .to_timestamp with a large periods count.
    assert isinstance(edge_world.index, pd.PeriodIndex)
    assert list(edge_world.columns) == ["optimized", "front", "cash"]


def test_excess_frame_nets_cash():
    df = pd.DataFrame({"a": [0.02, 0.03], "b": [0.01, 0.00], "cash": [0.005, 0.005]})
    ex = st.excess_frame(df, cash="cash")
    assert "cash" not in ex.columns
    assert np.allclose(ex["a"].to_numpy(), [0.015, 0.025])
    assert np.allclose(ex["b"].to_numpy(), [0.005, -0.005])


def test_planted_edge_recovered(edge_world):
    r = st.synthetic_detect(edge_world)
    assert r["sharpe_adv"] > 0.05          # optimized beats front
    assert r["t_diff"] > 2.0               # the return difference is significant
    assert r["diff_ann_pct"] > 1.0         # ~ the planted 3 %/yr, up to noise
    assert r["adv_ci_lo"] > 0.0            # bootstrap advantage CI clear of zero


def test_null_world_no_edge(null_world):
    r = st.synthetic_detect(null_world)
    assert abs(r["t_diff"]) < 2.0          # no significant difference
    assert r["adv_ci_lo"] < 0.0 < r["adv_ci_hi"]  # advantage CI straddles zero


def test_costs_reduce_net_advantage(edge_world):
    ex = st.excess_frame(edge_world, cash="cash")
    gross = st.sharpe_race(ex, "optimized", "front")["diff_ann_pct"]
    # charge the optimized leg the wider spread; the net difference must shrink
    net = st.costed_race(ex, "optimized", "front",
                         spread_bps={"optimized": 20.0, "front": 2.0},
                         turnover_per_year=2.0)["diff_ann_pct_net"]
    assert net < gross


def test_era_race_runs_on_periodindex(edge_world):
    ex = st.excess_frame(edge_world, cash="cash")
    eras = [("first half", "2010-08", "2018-06"), ("second half", "2018-07", "2026-06")]
    out = st.era_race(ex, "optimized", "front", eras)
    assert len(out) == 2
    assert all("diff_ann_pct" in e and e["n"] > 8 for e in out)


def test_calendar_year_table(edge_world):
    ex = st.excess_frame(edge_world, cash="cash")
    cyt = st.calendar_year_table(ex, ["optimized", "front"])
    assert "optimized" in cyt.columns and "front" in cyt.columns
    assert len(cyt) >= 10                  # ~16 calendar years in the synthetic span


def test_max_drawdown_sign_and_bounds():
    up = np.full(24, 0.01)                 # monotone up -> no drawdown
    assert st.max_drawdown(up) == 0.0 or st.max_drawdown(up) > -1e-9
    mixed = np.array([0.1, -0.5, 0.1, 0.1])
    dd = st.max_drawdown(mixed)
    assert -100.0 < dd < 0.0


def test_nw_matches_plain_t_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 3000)
    mu, t_nw = st.nw_mean_t(x, lags=6)
    se = x.std(ddof=1) / np.sqrt(len(x))
    t_plain = x.mean() / se
    assert abs(t_nw - t_plain) < 0.6


def test_annualized_sharpe_scaling():
    rng = np.random.default_rng(1)
    x = rng.normal(0.01, 0.04, 5000)
    sr = st.annualized_sharpe(x, ppy=12)
    manual = x.mean() / x.std(ddof=1) * np.sqrt(12)
    assert abs(sr - manual) < 1e-9
