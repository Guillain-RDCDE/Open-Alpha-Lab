"""The engine and the study's spine: (1) the paper (0-cost) book earns a genuine gross edge
on the planted world and ~nothing on the null; (2) the same edge is eaten as cost rises —
net Sharpe falls monotonically down the cost ladder and the strategy dies (net t < 2) at the
realistic rung; (3) it dies as a FUNCTION of turnover — the gross Sharpe is ~flat across the
persistence sweep while the net Sharpe collapses as turnover rises; (4) the book is
dollar-neutral and point-in-time (one shift, no look-ahead); (5) the inference primitives
behave. All offline, fixed-seed."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from cost_gap import data, strategy as st  # noqa: E402


# ---- the book: dollar-neutral, point-in-time -------------------------------
def test_book_is_dollar_neutral(edge_world):
    _, sig, _ = edge_world
    W = st.book_weights(sig, frac=0.2).to_numpy()
    # after the warm-up row, each row sums to ~0 (long leg +1, short leg -1)
    row_sums = np.abs(W[1:].sum(axis=1))
    assert row_sums.max() < 1e-9
    gross_exposure = np.abs(W[1:]).sum(axis=1)
    assert np.allclose(gross_exposure, 2.0)      # 2x NAV gross, dollar-neutral


def test_book_is_point_in_time(edge_world):
    """Weights on day t come from the signal known at t-1 (a single shift), never day t."""
    _, sig, _ = edge_world
    W = st.book_weights(sig, frac=0.2)
    # rebuild the weights from an explicitly pre-shifted signal with NO internal shift path:
    shifted = sig.shift(1)
    # the long set on row t must be the top-frac names of row t-1 of the ORIGINAL signal
    t = 100
    k = int(0.2 * sig.shape[1])
    top_prev = set(sig.iloc[t - 1].nlargest(k).index)
    longs = set(W.columns[W.iloc[t].to_numpy() > 0])
    assert longs == top_prev
    assert shifted.iloc[t].equals(sig.iloc[t - 1])


# ---- the paper edge fires on the plant, silent on the null -----------------
def test_paper_edge_present_on_plant(edge_world):
    rets, sig, _ = edge_world
    book = st.book_returns(rets, sig, frac=0.2)
    b = st.book_stats(book, cost_bps=0.0, impact_coef_bps=0.0)
    assert b["gross_bps"] > 0
    assert b["gross_t"] > 4.0            # a dazzling paper t
    assert b["gross_sharpe"] > 1.5


def test_paper_edge_absent_on_null(null_world):
    rets, sig, _ = null_world
    book = st.book_returns(rets, sig, frac=0.2)
    b = st.book_stats(book, cost_bps=0.0, impact_coef_bps=0.0)
    assert abs(b["gross_t"]) < 2.5       # nothing to find


# ---- costs eat the edge: monotone decline, death at the realistic rung -----
def test_costs_reduce_net_monotonically(edge_world):
    rets, sig, _ = edge_world
    book = st.book_returns(rets, sig, frac=0.2)
    ladder = st.cost_ladder(book)
    net = ladder["net_sharpe"].to_numpy()
    assert (np.diff(net) < 0).all()      # every rung of cost lowers the net Sharpe
    assert ladder.loc["paper (0 cost)", "net_sharpe"] > 2.0
    assert ladder.loc["realistic", "net_t"] < 2.0     # the edge dies (no longer significant)
    assert ladder.loc["stressed", "net_sharpe"] < 0   # loses money when stressed


def test_apply_costs_monotone_in_cost(edge_world):
    rets, sig, _ = edge_world
    book = st.book_returns(rets, sig, frac=0.2)
    means = [st.apply_costs(book, cost_bps=c, impact_coef_bps=0).mean() for c in (0, 5, 10, 20)]
    assert all(m2 < m1 for m1, m2 in zip(means, means[1:]))


def test_impact_is_superlinear_in_turnover(edge_world):
    """Doubling the impact coefficient must cost strictly more than the linear term would,
    confirming the turnover^2 (participation) shape."""
    rets, sig, _ = edge_world
    book = st.book_returns(rets, sig, frac=0.2)
    base = st.apply_costs(book, cost_bps=0, impact_coef_bps=0).mean()
    imp = st.apply_costs(book, cost_bps=0, impact_coef_bps=50).mean()
    turn2 = (book["turnover"].to_numpy() ** 2).mean()
    assert np.isclose(base - imp, 50 * 1e-4 * turn2, rtol=1e-6)


# ---- the money chart: alpha dies as a FUNCTION of turnover -----------------
def test_turnover_curve_gross_flat_net_collapses():
    tc = st.turnover_curve(data, edge=0.0005,
                           persistences=(0.995, 0.98, 0.96, 0.9, 0.7, 0.3),
                           n_days=2520, seed=842)
    # turnover rises as persistence falls (rows ordered from high to low persistence)
    turn = tc["mean_turnover"].to_numpy()
    assert (np.diff(turn) > 0).all()                 # turnover strictly rises down the sweep
    # gross Sharpe is ~flat (the edge is held fixed)
    gross = tc["gross_sharpe"].to_numpy()
    assert gross.max() - gross.min() < 0.7
    # net Sharpe collapses monotonically and goes negative at high turnover
    net = tc["net_sharpe"].to_numpy()
    assert (np.diff(net) < 0).all()
    assert net[0] > 1.0 and net[-1] < 0
    # break-even cost falls as turnover rises
    be = tc["breakeven_bps"].to_numpy()
    assert (np.diff(be) < 0).all()


def test_breakeven_matches_definition(edge_world):
    rets, sig, _ = edge_world
    book = st.book_returns(rets, sig, frac=0.2)
    be = st.breakeven_cost_bps(book)
    # at exactly the break-even one-way linear cost, the net mean is ~0
    net_at_be = st.apply_costs(book, cost_bps=be, impact_coef_bps=0).mean()
    assert abs(net_at_be) < 1e-9


# ---- synthetic control: fires on a plant, silent on the null --------------
def test_seed_robust_control_unbiased():
    null = st.seed_robust_control(data, edge=0.0, n_seeds=20, n_days=1200)
    plant = st.seed_robust_control(data, edge=0.0005, n_seeds=20, n_days=1200)
    assert null["fire_count"] <= 2          # ~false-positive rate, not a systematic signal
    assert abs(null["mean_t"]) < 0.5
    assert plant["fire_count"] >= 18        # the planted edge lights up almost always
    assert plant["mean_t"] > 3.0


# ---- inference primitives --------------------------------------------------
def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=10) - st.one_sample_t(x)) < 0.6


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi


def test_sharpe_and_welch_sane():
    rng = np.random.default_rng(1)
    a = 0.001 + 0.01 * rng.standard_normal(3000)
    b = 0.01 * rng.standard_normal(3000)
    assert st.sharpe(a) > st.sharpe(b)
    assert st.welch_t(a, b) > 0
