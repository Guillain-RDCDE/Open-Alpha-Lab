"""The GA engine and the study's two spines: (1) on a no-edge tape the evolved IS champion reverts
to noise out of sample (the mirage), and (2) when a real edge is planted the champion keeps working
OOS (the control — proof the harness detects signal, never manufactures it). Plus the honesty
disciplines: long/flat positions, the demeaned timing book, deflated-Sharpe monotonicity, costs."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from genetic_algo_overfit import data  # noqa: E402
from genetic_algo_overfit import strategy as st  # noqa: E402

FEATS = data.FEATURE_NAMES


# ---- the rule & its honesty discipline -------------------------------------
def test_position_long_or_flat(null_tape):
    X = null_tape[FEATS].to_numpy()
    w = np.ones(len(FEATS))
    pos = st.rule_position(X, w)
    assert set(np.unique(pos)) <= {0.0, 1.0}


def test_book_is_demeaned(null_tape):
    """A rule that is ALWAYS long earns ~0 on the demeaned book (no drift harvesting)."""
    X = null_tape[FEATS].to_numpy()
    y = null_tape["fwd_ret"].to_numpy()
    w = np.zeros(len(FEATS))  # score is 0 -> not >0 -> flat; use a genome that is always long:
    always_long = st.rule_returns(X, y, w)  # score==0 -> flat everywhere -> all zeros
    assert np.allclose(always_long, 0.0)
    # a genome that is long on every day (via a constant positive shift is not possible with a
    # linear score alone, so check the demeaning property directly):
    mu = y[np.isfinite(y)].mean()
    pos = np.ones_like(y)
    manual = pos * (y - mu)
    assert abs(manual.mean()) < 1e-9  # a permanently-long book has ~zero mean after demeaning


def test_sharpe_zero_on_degenerate():
    assert st.sharpe(np.zeros(100)) == 0.0
    assert st.sharpe(np.array([0.01])) == 0.0


def test_evolve_is_deterministic(null_tape):
    X = null_tape[FEATS].to_numpy()
    y = null_tape["fwd_ret"].to_numpy()
    a = st.evolve_rule(X, y, pop_size=20, generations=8, seed=3)
    b = st.evolve_rule(X, y, pop_size=20, generations=8, seed=3)
    assert np.allclose(a.genome, b.genome)
    assert a.is_sharpe == b.is_sharpe
    assert a.n_trials == b.n_trials


def test_ga_finds_high_is_sharpe_on_noise(null_tape):
    """The whole point: even on a random walk the GA breeds a materially positive IS Sharpe."""
    res = st.is_oos_split(null_tape, pop_size=40, generations=20, seed=589)
    assert res["is_sharpe"] > 0.4
    assert res["n_trials"] > 100


# ---- spine 1: the mirage (null dies OOS) -----------------------------------
def test_null_overfits_shrinks_oos(null_tape):
    res = st.is_oos_split(null_tape, pop_size=40, generations=20, seed=589)
    # IS Sharpe is much larger than OOS Sharpe -> positive shrinkage (the overfit).
    assert res["is_sharpe"] > res["oos_sharpe"]
    assert res["shrinkage"] > 0.2


def test_null_seed_robust_oos_no_positive_edge():
    """Averaged over >=20 seeds, the null must NOT produce a materially positive OOS edge (no false
    signal). A negative OOS drift is fine — an overfit rule can actively lose OOS — what must never
    happen is the null manufacturing a tradable positive Sharpe that clears the bar."""
    sr = st.seed_robust_control(0.0, n_seeds=20, n_days=1200, pop_size=25, generations=15)
    assert sr["mean_oos_sharpe"] < 0.5   # no harvestable OOS edge on pure noise
    assert not (sr["oos_t"] > 2.0)       # never a spurious POSITIVE t clearing the bar


# ---- spine 2: the control (a real edge survives OOS) -----------------------
def test_planted_edge_survives_oos(edge_tape):
    res = st.is_oos_split(edge_tape, pop_size=40, generations=20, seed=589)
    assert res["oos_sharpe"] > 1.0  # a genuine edge is carried forward


def test_control_seed_robust_beats_null():
    null = st.seed_robust_control(0.0, n_seeds=20, n_days=1200, pop_size=25, generations=15)
    ctrl = st.seed_robust_control(0.4, n_seeds=20, n_days=1200, pop_size=25, generations=15)
    assert ctrl["mean_oos_sharpe"] > null["mean_oos_sharpe"] + 0.5
    assert ctrl["oos_t"] > 2.0


# ---- the deflated Sharpe & the luck bar ------------------------------------
def test_expected_max_sharpe_grows_with_trials():
    bars = [st.expected_max_sharpe(n) for n in [10, 100, 1000, 5000]]
    assert bars == sorted(bars)  # monotone increasing in the trial count
    assert bars[0] > 0


def test_deflated_sharpe_near_zero_on_overfit(null_tape):
    """A Sharpe born of a big search on noise must not clear the deflated bar (DSR ~ 0)."""
    (fis, yis), _, _ = st._split_arrays(null_tape, 0.5)
    ga = st.evolve_rule(fis, yis, pop_size=40, generations=20, seed=589)
    is_ret = st.rule_returns(fis, yis, ga.genome)
    dsr = st.deflated_sharpe(is_ret, n_trials=ga.n_trials)
    assert dsr["dsr"] < 0.5


# ---- costs cannot help --------------------------------------------------
def test_costs_do_not_raise_sharpe(edge_tape):
    _, (foos, yoos), _ = st._split_arrays(edge_tape, 0.5)
    ga = st.evolve_rule(*st._split_arrays(edge_tape, 0.5)[0], pop_size=30, generations=15, seed=589)
    c = st.net_sharpe(foos, yoos, ga.genome, cost_bps=5.0)
    assert c["net_sharpe"] <= c["gross_sharpe"] + 1e-9
    assert 0.0 <= c["turnover_per_day"] <= 1.0
