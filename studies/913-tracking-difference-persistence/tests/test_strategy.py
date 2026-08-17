"""Measurement, rule and inference invariants — all offline/synthetic.

The spine: on a panel with a planted fee ladder the tracking-difference rank persists and
the ``cheapest`` rule beats the dearest ``leader``; on a flat-fee panel neither happens.
Partial calendar years never enter the annual work, the tracking difference is invariant to
the choice of index proxy, and the rules carry exactly one execution lag.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from td_persist import data, strategy as st  # noqa: E402


def _funds(prices):
    return [c for c in prices.columns if c != "cash"]


# --------------------------------------------------------------------------- #
# The measurement
# --------------------------------------------------------------------------- #
def test_complete_years_drops_partial_years():
    idx = pd.bdate_range("2019-11-01", "2022-03-15")
    r = pd.DataFrame({"a": np.zeros(len(idx))}, index=idx)
    yrs = st.complete_years(r)
    assert yrs == [2020, 2021]          # 2019 and 2022 are stubs and must be dropped


def test_annual_returns_compound_within_the_year(fee_ladder):
    prices, _ = fee_ladder
    p = prices[_funds(prices)]
    ann = st.annual_returns(p)
    r = st.daily_returns(p)
    y = int(ann.index[3])
    manual = float((1.0 + r[r.index.year == y]["fund_0"]).prod() - 1.0)
    assert abs(manual - float(ann.loc[y, "fund_0"])) < 1e-12


def test_relative_td_sums_to_zero_each_year(fee_ladder):
    prices, _ = fee_ladder
    td = st.tracking_difference(st.annual_returns(prices[_funds(prices)]))
    assert np.allclose(td.sum(axis=1).to_numpy(), 0.0, atol=1e-8)


def test_td_rank_is_invariant_to_the_index_proxy(fee_ladder):
    """A common per-year constant shifts every fund equally, so ranks cannot move."""
    prices, _ = fee_ladder
    ann = st.annual_returns(prices[_funds(prices)])
    a = st.tracking_difference(ann, proxy="mean").rank(axis=1)
    b = st.tracking_difference(ann, proxy="leader").rank(axis=1)
    c = st.tracking_difference(ann, proxy="fund_2").rank(axis=1)
    assert (a == b).all().all() and (a == c).all().all()


def test_spearman_matches_known_cases():
    assert abs(st.spearman([1, 2, 3, 4], [1, 2, 3, 4]) - 1.0) < 1e-12
    assert abs(st.spearman([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-12
    assert np.isnan(st.spearman([1, 1, 1], [1, 2, 3]))


def test_pairwise_rank_corr_matrix_is_symmetric_with_unit_diagonal(fee_ladder):
    prices, _ = fee_ladder
    td = st.tracking_difference(st.annual_returns(prices[_funds(prices)]))
    m = st._pairwise_rank_corr_matrix(td)
    assert m.shape == (len(td), len(td))
    assert np.allclose(m, m.T)
    assert np.allclose(np.diag(m), 1.0)


# --------------------------------------------------------------------------- #
# The rules: lag, costs, weights
# --------------------------------------------------------------------------- #
def test_weights_sum_to_one_every_live_year(fee_ladder):
    prices, truth = fee_ladder
    td = st.tracking_difference(st.annual_returns(prices[_funds(prices)]))
    for rule in st.RULES:
        w = st.rule_weights(td, truth["fees_bps"], rule)
        assert np.allclose(w.sum(axis=1).to_numpy(), 1.0)
        assert (w.to_numpy() >= -1e-12).all()


def test_first_complete_year_is_consumed_by_the_ranking(fee_ladder):
    prices, truth = fee_ladder
    td = st.tracking_difference(st.annual_returns(prices[_funds(prices)]))
    w = st.rule_weights(td, truth["fees_bps"], "winner")
    assert int(w.index.min()) == int(td.index.min()) + 1
    assert len(w) == len(td) - 1


def test_cheapest_and_leader_never_trade(fee_ladder):
    prices, truth = fee_ladder
    td = st.tracking_difference(st.annual_returns(prices[_funds(prices)]))
    assert st.switch_count(td, truth["fees_bps"], "cheapest") == 0
    assert st.switch_count(td, truth["fees_bps"], "leader") == 0


def test_one_execution_lag_no_lookahead(fee_ladder):
    """Perturbing the tape's tail must not change any earlier rule return."""
    prices, truth = fee_ladder
    p = prices[_funds(prices)]
    base = st.run_rules(p, truth["fees_bps"], cost_bps=1.0)
    cut = base.index[-300]
    p2 = p.copy()
    p2.loc[p2.index > cut] *= 1.5           # perturb only the future
    pert = st.run_rules(p2, truth["fees_bps"], cost_bps=1.0)
    common = base.index[base.index <= cut]
    assert np.allclose(base.loc[common].to_numpy(), pert.loc[common].to_numpy())


def test_costs_monotonically_reduce_the_traded_rule(fee_ladder):
    prices, truth = fee_ladder
    p = prices[_funds(prices)]
    means = [st.run_rules(p, truth["fees_bps"], cost_bps=c)["winner"].mean()
             for c in (0.0, 5.0, 25.0)]
    assert means[0] >= means[1] >= means[2]


def test_untraded_rules_are_cost_invariant(fee_ladder):
    prices, truth = fee_ladder
    p = prices[_funds(prices)]
    a = st.run_rules(p, truth["fees_bps"], cost_bps=0.0)["cheapest"]
    b = st.run_rules(p, truth["fees_bps"], cost_bps=25.0)["cheapest"]
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_leader_rule_is_exactly_the_first_column(fee_ladder):
    prices, truth = fee_ladder
    p = prices[_funds(prices)]
    daily = st.run_rules(p, truth["fees_bps"], cost_bps=0.0)
    r = st.daily_returns(p).reindex(daily.index)
    assert np.allclose(daily["leader"].to_numpy(), r["fund_0"].to_numpy())


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_recovers_positive_mean():
    rng = np.random.default_rng(1)
    x = 0.001 + rng.normal(0, 0.01, 5000)
    assert st.newey_west_t(x) > 3
    assert abs(st.newey_west_t(rng.normal(0, 0.01, 5000))) < 3


def test_one_sample_t_handles_degenerate_input():
    assert np.isnan(st.one_sample_t([1.0]))
    assert np.isnan(st.one_sample_t([2.0, 2.0, 2.0]))


def test_block_bootstrap_ci_brackets_the_mean():
    rng = np.random.default_rng(3)
    x = 5.0 + rng.normal(0, 2.0, 40)
    ci = st.block_bootstrap_ci(x, n_boot=1000, seed=913)
    assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]
    assert ci["frac_negative"] < 0.05


def test_tax_breakeven_is_monotone_and_free_when_sheltered():
    tbl = st.tax_breakeven_years(6.45)
    assert (tbl["tax_000"] == 0.0).all()
    assert tbl["tax_200"].is_monotonic_increasing
    assert tbl.loc[50.0, "tax_200"] > 100.0        # a taxable switch never pays back
    assert np.isinf(st.tax_breakeven_years(-1.0).loc[50.0, "tax_200"])


# --------------------------------------------------------------------------- #
# The study's spine — the machinery is unbiased
# --------------------------------------------------------------------------- #
def test_planted_ladder_persists(fee_ladder):
    prices, truth = fee_ladder
    d = st.synthetic_detect(prices, truth)
    assert d["mean_spearman"] > 0.3
    assert d["t_spearman"] > 2.0


def test_planted_ladder_cheapest_beats_leader(fee_ladder):
    prices, truth = fee_ladder
    d = st.synthetic_detect(prices, truth)
    assert d["cheapest_minus_leader_bp"] > 0.5 * truth["fee_spread_bps_eff"]
    assert d["t_cheapest_minus_leader"] > 2.0


def test_planted_ladder_winner_does_not_beat_cheapest(fee_ladder):
    """Even where persistence is real, chasing last year's winner is not the way to use it."""
    prices, truth = fee_ladder
    d = st.synthetic_detect(prices, truth)
    assert d["winner_minus_cheapest_bp"] < d["cheapest_minus_leader_bp"]


def test_null_no_persistence_across_seeds():
    """On flat fees the rank must not persist — checked across seeds, not one lucky draw."""
    vals = []
    for s in range(8):
        prices, truth = data.synthetic_panel(signal_strength=0.0, seed=913 + s)
        td = st.tracking_difference(st.annual_returns(prices[_funds(prices)]))
        vals.append(float(st.yoy_rank_correlations(td).mean()))
    vals = np.array(vals)
    assert abs(vals.mean()) < 0.25
    assert (np.abs(vals) >= 0.45).sum() <= 1


def test_null_no_spurious_fee_gap():
    """On flat fees no rule may out-earn another by a meaningful margin."""
    gaps = []
    for s in range(6):
        prices, truth = data.synthetic_panel(signal_strength=0.0, seed=913 + s)
        d = st.synthetic_detect(prices, truth)
        gaps.append(d["cheapest_minus_leader_bp"])
    gaps = np.array(gaps)
    assert abs(gaps.mean()) < 4.0
    assert abs(gaps.mean()) < 0.2 * 30.0      # far below the planted-world spread


def test_era_cut_splits_and_reports_both_halves(fee_ladder):
    prices, truth = fee_ladder
    p = prices[_funds(prices)][::-1].sort_index()
    e = st.era_cut(p, truth["fees_bps"], "cheapest", "leader")
    assert set(("early", "late")).issubset(e)
    for tag in ("early", "late"):
        assert e[tag]["n_years"] > 0
        assert np.isfinite(e[tag]["annual_gap_bp_yr"])


def test_cost_sweep_is_monotone_for_a_traded_rule(fee_ladder):
    prices, truth = fee_ladder
    sweep = st.cost_sweep(prices[_funds(prices)], truth["fees_bps"], "winner", "cheapest")
    assert sweep["annual_gap_bp_yr"].is_monotonic_decreasing


def test_excess_of_cash_race_covers_every_rule(fee_ladder):
    prices, truth = fee_ladder
    race = st.excess_of_cash_race(prices[_funds(prices)], prices["cash"], truth["fees_bps"])
    assert set(race.index) == set(st.RULES)
    assert race["excess_sharpe"].notna().all()
    # Same index underneath every arm: the Sharpe spread must be tiny.
    assert race["excess_sharpe"].max() - race["excess_sharpe"].min() < 0.10


def test_gap_stats_reports_both_units(fee_ladder):
    prices, truth = fee_ladder
    # Leader-first ordering, mirroring the real families where the flagship is the dearest.
    dearest_first = sorted(_funds(prices), key=lambda f: -truth["fees_bps"][f])
    g = st.gap_stats(prices[dearest_first], truth["fees_bps"], "cheapest", "leader")
    for k in ("annual_gap_bp_yr", "t_annual", "t_annual_hac", "daily_gap_bp_yr",
              "t_daily_hac", "ci_low", "ci_high"):
        assert np.isfinite(g[k])
    assert g["n_years"] >= 5 and g["n_days"] > 1000


# --------------------------------------------------------------------------- #
# The hindsight-free control: the money question with NO fee sheet at all
# --------------------------------------------------------------------------- #
def test_hindsight_free_control_recovers_the_planted_ladder(fee_ladder):
    """Every fund cheaper than the flagship must beat it, without consulting any fee."""
    prices, truth = fee_ladder
    dearest_first = sorted(_funds(prices), key=lambda f: -truth["fees_bps"][f])
    tbl = st.per_fund_gap_vs_leader(prices[dearest_first])
    # one row per non-flagship fund, plus the equal-weight blend
    assert len(tbl) == len(dearest_first)
    assert "EQW_non_flagship-fund_3" in tbl.index[-1] or tbl.index[-1].startswith("EQW")
    assert (tbl["annual_gap_bp_yr"] > 0).all()
    assert tbl.loc[tbl.index[-1], "t_annual"] > 2.0


def test_hindsight_free_control_is_silent_on_the_flat_fee_null():
    """No fee ladder, no gap — the control must not manufacture one."""
    gaps = []
    for s in range(6):
        prices, _ = data.synthetic_panel(signal_strength=0.0, seed=913 + s)
        tbl = st.per_fund_gap_vs_leader(prices[_funds(prices)])
        gaps.append(float(tbl.loc[tbl.index[-1], "annual_gap_bp_yr"]))
    assert abs(np.mean(gaps)) < 4.0


def test_hindsight_free_control_needs_no_expense_ratios():
    """Signature guard: the control must not accept (and so cannot peek at) a fee sheet."""
    import inspect
    params = set(inspect.signature(st.per_fund_gap_vs_leader).parameters)
    assert "expense_ratio_bps" not in params


def test_persistence_permutation_p_is_a_probability(fee_ladder):
    prices, _ = fee_ladder
    td = st.tracking_difference(st.annual_returns(prices[_funds(prices)]))
    p = st.persistence(td, n_perm=300, seed=913)
    assert 0.0 <= p["p_permutation"] <= 1.0
    assert p["n_pairs"] == len(td) - 1


def test_two_fund_family_is_flagged_degenerate(fee_ladder):
    prices, _ = fee_ladder
    td = st.tracking_difference(st.annual_returns(prices[["fund_0", "fund_3"]]))
    p = st.persistence(td, n_perm=200, seed=913)
    assert p["degenerate_two_fund"] is True
    assert set(np.unique(p["yoy"].to_numpy())).issubset({-1.0, 1.0})


def test_unknown_rule_raises(fee_ladder):
    prices, truth = fee_ladder
    td = st.tracking_difference(st.annual_returns(prices[_funds(prices)]))
    with pytest.raises(ValueError):
        st.rule_weights(td, truth["fees_bps"], "hindsight")
