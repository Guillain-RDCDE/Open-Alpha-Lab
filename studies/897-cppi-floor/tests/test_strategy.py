"""Offline, fixed-seed tests for the CPPI machinery.

The engine is point-in-time (the weight on day t uses only the state at t−1); the floor accretes
at the cash rate and bounds the drawdown in a continuous fall; gap risk breaches it exactly when a
one-day drop exceeds 1/m; the cushion cash-locks at zero; a constant fraction of the risky asset
has the SAME excess-of-cash Sharpe as the risky asset; costs reduce the net; the bear world's floor
holds while the calm null protects nothing. All offline and deterministic.
"""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from cppi import strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# The CPPI engine — no look-ahead, floor accretion, cash-lock
# --------------------------------------------------------------------------- #
def test_first_weight_is_state_at_inception_not_todays_return():
    # Day-0 weight = clip(m·(V−F)/V, 0, cap) from the initial state (1, floor) — independent of r[0].
    r_up = np.array([0.05, 0.0, 0.0]); r_dn = np.array([-0.05, 0.0, 0.0])
    cash = np.zeros(3)
    w_up = st.cppi_book(r_up, cash, multiplier=5.0, floor_frac=0.80)["weight"][0]
    w_dn = st.cppi_book(r_dn, cash, multiplier=5.0, floor_frac=0.80)["weight"][0]
    assert abs(w_up - w_dn) < 1e-12                       # today's return can't move today's weight
    assert abs(w_up - min(5.0 * 0.20, 1.0)) < 1e-12       # = m·cushion/V = 5·0.20 = 1.0 (capped)


def test_no_lookahead_future_return_does_not_change_past_weights():
    rng = np.random.default_rng(0)
    r = rng.normal(0.0003, 0.012, 200)
    cash = np.full(200, 0.02 / 252)
    base = st.cppi_book(r, cash, multiplier=5.0, floor_frac=0.80)["weight"]
    r2 = r.copy(); r2[-1] += 0.5                          # perturb only the LAST day
    pert = st.cppi_book(r2, cash, multiplier=5.0, floor_frac=0.80)["weight"]
    assert np.allclose(base[:-1], pert[:-1])              # earlier weights untouched


def test_floor_accretes_at_cash_rate():
    r = np.zeros(50)
    cash = np.full(50, 0.03 / 252)
    book = st.cppi_book(r, cash, multiplier=5.0, floor_frac=0.80)
    expected = 0.80 * np.cumprod(1.0 + cash)
    assert np.allclose(book["floor"], expected)


def test_cushion_cash_locks_when_exhausted():
    # A hard sustained fall drives the cushion to zero -> weight collapses to 0 (cash-locked).
    r = np.full(300, -0.02)
    cash = np.full(300, 0.02 / 252)
    book = st.cppi_book(r, cash, multiplier=5.0, floor_frac=0.80)
    assert book["weight"][-1] < 1e-6
    assert book["min_cushion"] <= 1e-6


def test_floor_holds_in_continuous_fall_no_breach(bear_world):
    # Absent an overnight gap past 1/m, daily-rebalanced CPPI never breaches the floor.
    frame, _ = bear_world
    ret = st.to_returns(frame)
    r = st.race(ret, multiplier=5.0, floor_frac=0.80)
    assert r["n_breach"] == 0
    assert r["dd_cppi"] > r["dd_bh"] + 0.10              # CPPI drawdown clearly shallower
    # The mechanical guarantee is equity ≥ floor on every day (the floor is never breached).
    assert bool((r["equity"] >= r["floor"] - 1e-9).all())


# --------------------------------------------------------------------------- #
# Gap risk — a one-day drop past 1/m breaks the floor
# --------------------------------------------------------------------------- #
def test_gap_breaches_floor_when_drop_exceeds_one_over_m(gap_world, nogap_world):
    fg, _ = gap_world
    fn, _ = nogap_world
    rg = st.race(st.to_returns(fg), multiplier=5.0, floor_frac=0.80)
    rn = st.race(st.to_returns(fn), multiplier=5.0, floor_frac=0.80)
    assert rg["n_breach"] > 0                             # 25% gap > 20% -> breaches
    assert rn["n_breach"] == 0                            # 15% gap < 20% -> holds


def test_gap_stress_threshold_is_one_over_m(bear_world):
    # Splice a single overnight crash: it breaks the floor iff the drop exceeds 1/m = 20%.
    frame, _ = bear_world
    ret = st.to_returns(frame)
    gs = st.gap_stress(ret, gap_sizes=(0.10, 0.25), multiplier=5.0, floor_frac=0.80)
    assert bool(gs.loc[0.10, "breaches_floor"]) is False
    assert bool(gs.loc[0.25, "breaches_floor"]) is True


# --------------------------------------------------------------------------- #
# The fair-race identity and costs
# --------------------------------------------------------------------------- #
def test_constant_mix_excess_sharpe_equals_risky(calm_world):
    # A constant fraction w of SPY funded from cash has the SAME excess-of-cash Sharpe as SPY.
    frame, _ = calm_world
    ret = st.to_returns(frame)
    r = st.race(ret, multiplier=5.0, floor_frac=0.80)
    assert abs(r["static"]["sharpe"] - r["buy_hold"]["sharpe"]) < 1e-9
    assert abs(r["sharpe_vs_bh"] - r["sharpe_vs_static"]) < 1e-9


def test_costs_reduce_net(bear_world):
    frame, _ = bear_world
    ret = st.to_returns(frame)
    rr = ret["SPY"].to_numpy(); cr = ret["BIL"].to_numpy()
    gross = st.cppi_book(rr, cr, 5.0, 0.80, cost_bps=0.0)["net"].sum()
    net = st.cppi_book(rr, cr, 5.0, 0.80, cost_bps=10.0)["net"].sum()
    assert net < gross


def test_cost_sweep_cppi_sharpe_monotone_decreasing(bear_world):
    frame, _ = bear_world
    ret = st.to_returns(frame)
    sweep = st.cost_sweep(ret, one_way_bps=(0.0, 2.0, 10.0), multiplier=5.0, floor_frac=0.80)
    s = sweep["cppi_sharpe"].to_numpy()
    assert s[0] >= s[1] >= s[2]                           # more cost -> weakly lower Sharpe


def test_band_reduces_turnover(bear_world):
    frame, _ = bear_world
    ret = st.to_returns(frame)
    rr = ret["SPY"].to_numpy(); cr = ret["BIL"].to_numpy()
    daily = st.cppi_book(rr, cr, 5.0, 0.80, band=0.0)["turnover"].sum()
    banded = st.cppi_book(rr, cr, 5.0, 0.80, band=0.05)["turnover"].sum()
    assert banded < daily                                 # a rebalance tolerance trades less


# --------------------------------------------------------------------------- #
# The bear world's floor bites; the calm null protects nothing
# --------------------------------------------------------------------------- #
def test_bear_world_floor_bites(bear_world):
    frame, _ = bear_world
    d = st.synthetic_detect(frame)
    assert d["dd_protection"] > 0.10                      # CPPI drawdown much shallower than BH
    assert d["n_breach"] == 0


def test_calm_null_nothing_to_protect(calm_world):
    frame, _ = calm_world
    d = st.synthetic_detect(frame)
    assert abs(d["dd_protection"]) < 0.05                 # the two drawdowns nearly coincide


def test_multiplier_sweep_breach_threshold_is_one_over_m(bear_world):
    frame, _ = bear_world
    ret = st.to_returns(frame)
    ms = st.multiplier_sweep(ret, multipliers=(4.0, 5.0), floor_frac=0.80)
    assert abs(ms.loc[4.0, "breach_thresh"] - 0.25) < 1e-12
    assert abs(ms.loc[5.0, "breach_thresh"] - 0.20) < 1e-12
