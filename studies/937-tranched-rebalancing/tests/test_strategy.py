"""Schedule mechanics, backtest invariants, and the study's spine — all offline.

The spine: on a synthetic tape the same rule run on 21 different rebalance offsets prints
materially different Sharpes (the timing-luck cone), the cone is present **whether or not
the rule has edge**, and tranching shrinks it monotonically to exactly zero at N = 21
without inflating turnover. The sleeve itself must beat an exposure-matched random control
on the planted regime and must not on the null.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tranching import data, strategy as st  # noqa: E402

N_LIST = (1, 4, 21)


def _cone(prices, cost_bps=5.0, signal="trend"):
    state = (st.trend_state(prices["risky"]) if signal == "trend"
             else st.random_state(st.trend_state(prices["risky"]), seed=7))
    R, TV = st.offset_return_matrix(prices["risky"], prices["safe"], state, cost_bps=cost_bps)
    r_cash = st.cash_returns_from_index(prices["cash"])
    return st.cone(R, TV, r_cash, n_list=N_LIST)


# --------------------------------------------------------------------------- #
# Schedule mechanics
# --------------------------------------------------------------------------- #
def test_offset_positions_change_only_after_a_rebalance_day():
    close = pd.Series(np.arange(200, 500, dtype=float),
                      index=pd.bdate_range("2020-01-02", periods=300))
    state = pd.Series(np.tile([0.0, 1.0], 150), index=close.index)
    pos = st.offset_positions(state, offset=3, period=21)
    changed = pos.diff().abs().fillna(0.0) > 0
    # a change at bar i means the decision was taken at bar i-1, a rebalance day
    idx = np.flatnonzero(changed.to_numpy())
    assert set((idx - 1) % 21) <= {3}


def test_offset_positions_are_lagged_one_day():
    state = pd.Series(np.zeros(60), index=pd.bdate_range("2021-01-04", periods=60))
    state.iloc[21] = 1.0                      # decision taken on rebalance day 21
    pos = st.offset_positions(state, offset=0, period=21)
    assert pos.iloc[21] == 0.0                # not yet traded
    assert pos.iloc[22] == 1.0                # traded the next day — exactly one lag


def test_no_lookahead_future_perturbation_leaves_past_untouched(regime):
    prices, _ = regime
    risky = prices["risky"]
    s1 = st.trend_state(risky)
    risky2 = risky.copy()
    risky2.iloc[4000:] *= 3.0                 # perturb only the future
    s2 = st.trend_state(risky2)
    p1 = st.offset_positions(s1, offset=5)
    p2 = st.offset_positions(s2, offset=5)
    assert (p1.iloc[:4000].fillna(-9) == p2.iloc[:4000].fillna(-9)).all()


def test_tranche_offsets_are_distinct_and_spread():
    for n in N_LIST:
        offs = st.tranche_offsets(n, base=0, period=21)
        assert len(offs) == n
        assert len(set(offs)) == n
        assert all(0 <= o < 21 for o in offs)
    assert st.tranche_offsets(21, base=4) == list(range(21))   # full set, any anchor
    assert st.tranche_offsets(1, base=7) == [7]


def test_combine_of_one_tranche_is_that_tranche(regime):
    prices, _ = regime
    state = st.trend_state(prices["risky"])
    R, TV = st.offset_return_matrix(prices["risky"], prices["safe"], state)
    r, t = st.combine(R, TV, [3])
    assert np.allclose(r.to_numpy(), R[3].to_numpy())
    assert np.allclose(t.to_numpy(), TV[3].to_numpy())


def test_offset_matrix_is_aligned_and_finite(regime):
    prices, _ = regime
    state = st.trend_state(prices["risky"])
    R, TV = st.offset_return_matrix(prices["risky"], prices["safe"], state)
    assert list(R.columns) == list(range(21))
    assert not R.isna().any().any()
    assert not TV.isna().any().any()
    assert (TV >= 0).all().all()


def test_costs_monotonically_reduce_returns(regime):
    prices, _ = regime
    state = st.trend_state(prices["risky"])
    means = []
    for c in (0.0, 5.0, 25.0):
        R, _ = st.offset_return_matrix(prices["risky"], prices["safe"], state, cost_bps=c)
        means.append(float(R.mean().mean()))
    assert means[0] >= means[1] >= means[2]


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_recovers_positive_mean():
    rng = np.random.default_rng(1)
    x = 0.001 + rng.normal(0, 0.01, 5000)
    assert st.newey_west_t(x) > 3
    assert abs(st.newey_west_t(rng.normal(0, 0.01, 5000))) < 3


def test_spearman_matches_known_cases():
    assert st.spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert st.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)


def test_one_sample_and_sharpe_finite(regime):
    prices, _ = regime
    state = st.trend_state(prices["risky"])
    R, _ = st.offset_return_matrix(prices["risky"], prices["safe"], state)
    assert np.isfinite(st.one_sample_t(R[0].to_numpy()))
    assert np.isfinite(st.sharpe(R[0]))
    assert np.isfinite(st.sharpe_diff_tstat(R[0], R[10]))


# --------------------------------------------------------------------------- #
# The spine — the cone, and what tranching does to it
# --------------------------------------------------------------------------- #
def test_cone_exists_at_one_tranche(regime):
    """The identical rule on 21 different rebalance dates prints different Sharpes."""
    cn = _cone(regime[0])
    assert cn[1]["sharpe_sd"] > 0.01
    assert cn[1]["sharpe_range"] > 0.03


def test_cone_exists_on_the_null_too(flat):
    """Timing luck is an artefact of the date, not a symptom of edge."""
    cn = _cone(flat[0])
    assert cn[1]["sharpe_sd"] > 0.01


def test_cone_exists_for_an_edge_free_rule(regime):
    """Even a random-timing rule has a cone — so the cone cannot be evidence of skill."""
    cn = _cone(regime[0], signal="random")
    assert cn[1]["sharpe_sd"] > 0.01


def test_random_cone_sweep_is_a_distribution_not_a_lucky_seed(regime):
    """The edge-free control must be quoted across seeds: one schedule is one draw."""
    prices, _ = regime
    state = st.trend_state(prices["risky"])
    r_cash = st.cash_returns_from_index(prices["cash"])
    sds = st.random_cone_sd_sweep(prices["risky"], prices["safe"], state, r_cash,
                                  seeds=range(3))
    assert len(sds) == 3
    assert np.all(sds > 0.0)
    assert sds.std() > 0.0                                  # seeds really do differ
    again = st.random_cone_sd_sweep(prices["risky"], prices["safe"], state, r_cash,
                                    seeds=range(3))
    assert np.allclose(sds, again)                          # and are deterministic


def test_dispersion_shrinks_monotonically_with_tranches(regime):
    cn = _cone(regime[0])
    sds = [cn[n]["sharpe_sd"] for n in N_LIST]
    assert all(a >= b - 1e-12 for a, b in zip(sds, sds[1:]))
    assert sds[0] > sds[-1]


def test_full_tranching_collapses_the_cone_exactly(regime):
    """At N = period every rotation is the same set of offsets: dispersion is exactly 0."""
    cn = _cone(regime[0])
    assert cn[21]["sharpe_sd"] == pytest.approx(0.0, abs=1e-12)
    assert cn[21]["cagr_range"] == pytest.approx(0.0, abs=1e-12)
    assert cn[21]["tw_spread_pct"] == pytest.approx(0.0, abs=1e-9)


def test_tranching_does_not_inflate_turnover(regime):
    cn = _cone(regime[0])
    assert cn[21]["turnover_mean"] <= cn[1]["turnover_mean"] * 1.05


def test_tranched_book_sits_inside_the_cone(regime):
    """Tranching is not a return trick: the book lands between the luckiest and unluckiest."""
    cn = _cone(regime[0])
    assert cn[1]["sharpe_min"] <= cn[21]["sharpe_mean"] <= cn[1]["sharpe_max"]


def test_tickets_grow_with_tranches(regime):
    prices, _ = regime
    state = st.trend_state(prices["risky"])
    R, TV = st.offset_return_matrix(prices["risky"], prices["safe"], state)
    t1 = st.ticket_stats(TV, 1, n_days=len(R))["tickets_per_year"]
    t21 = st.ticket_stats(TV, 21, n_days=len(R))["tickets_per_year"]
    assert t21 > 5 * t1     # proportional cost is flat, but broker tickets are not


def test_bootstrap_brackets_the_point_estimates(regime):
    prices, _ = regime
    state = st.trend_state(prices["risky"])
    R, TV = st.offset_return_matrix(prices["risky"], prices["safe"], state)
    r_cash = st.cash_returns_from_index(prices["cash"])
    bs = st.dispersion_bootstrap(R, TV, r_cash, n_boot=120, seed=937)
    assert bs["sd_ci"][0] <= bs["sd_point"] <= bs["sd_ci"][1]
    assert bs["gain_ci"][0] <= bs["gain_point"] <= bs["gain_ci"][1]


def test_persistence_and_best_worst_shapes(regime):
    prices, _ = regime
    state = st.trend_state(prices["risky"])
    R, _ = st.offset_return_matrix(prices["risky"], prices["safe"], state)
    r_cash = st.cash_returns_from_index(prices["cash"])
    ps = st.persistence(R, r_cash)
    assert 1 <= ps["late_rank_of_early_winner"] <= 21
    assert -1.0 <= ps["rho"] <= 1.0
    # the chase edge must be compared with the tranching gain on the SAME rows
    assert np.isfinite(ps["gain_late"]) and np.isfinite(ps["gain_early"])
    assert ps["chase_edge_late"] == pytest.approx(
        ps["late_sharpe_of_early_winner"] - ps["late_sharpe_mean"], abs=1e-12)
    bw = st.best_worst_test(R, r_cash)
    assert bw["sharpe_gap"] >= 0.0
    assert bw["best_offset"] != bw["worst_offset"]


def test_era_cut_and_cost_sweep_run(regime):
    prices, _ = regime
    state = st.trend_state(prices["risky"])
    r_cash = st.cash_returns_from_index(prices["cash"])
    eras = st.era_cut(prices["risky"], prices["safe"], state, r_cash,
                      split=str(prices.index[len(prices) // 2].date()))
    assert eras["early"] is not None and eras["late"] is not None
    for e in eras.values():
        assert np.isfinite(e["sharpe_sd_n1"])
    rows = st.cost_sweep(prices["risky"], prices["safe"], state, r_cash,
                         cost_grid=(0.0, 25.0))
    assert rows[0]["sharpe_mean_n1"] >= rows[1]["sharpe_mean_n1"]


# --------------------------------------------------------------------------- #
# Positive & negative controls — the machinery is unbiased
# --------------------------------------------------------------------------- #
def test_planted_regime_sleeve_beats_random_control(regime):
    d = st.synthetic_detect(regime[0])
    assert d["sleeve_minus_random"] > 0.15


def test_null_sleeve_does_not_beat_random_control(flat):
    d = st.synthetic_detect(flat[0])
    assert abs(d["sleeve_minus_random"]) < 0.25


def test_null_across_seeds_is_centered():
    advs = np.array([
        st.synthetic_detect(data.synthetic_daily(n_years=32, signal_strength=0.0,
                                                 seed=937 + s)[0])["sleeve_minus_random"]
        for s in range(3)
    ])
    assert abs(advs.mean()) < 0.15


def test_tranching_works_in_both_worlds(regime, flat):
    """The fix does not depend on the rule having edge — it is pure schedule arithmetic."""
    for prices in (regime[0], flat[0]):
        d = st.synthetic_detect(prices)
        assert d["sharpe_sd_n1"] > 0.01
        assert d["sharpe_sd_full"] == pytest.approx(0.0, abs=1e-12)
        assert d["turnover_full"] <= d["turnover_n1"] * 1.05
