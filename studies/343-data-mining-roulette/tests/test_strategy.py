"""The rule engine, the honest stats, and the study's two spines:
(1) the harness rediscovers a PLANTED edge and ranks it at the top (positive control), and
(2) on a pure NULL the best of N random rules still posts a 'significant' naive t-stat,
    yet Bonferroni passes ~none and the Reality Check is not significant — luck mimicking
    skill, caught only by correcting for the search."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_mining_roulette import strategy as st  # noqa: E402


# ---- rule generation & representation --------------------------------------
def test_random_rules_are_well_formed(null_tape):
    feats, _, _ = null_tape
    rules = st.random_rules(list(feats.columns), n_rules=50, seed=1)
    assert len(rules) == 50
    for r in rules:
        assert r["feature"] in feats.columns
        assert r["op"] in (">", "<")
        assert 0.2 <= r["q"] <= 0.8


def test_random_rules_deterministic(null_tape):
    feats, _, _ = null_tape
    a = st.random_rules(list(feats.columns), n_rules=30, seed=5)
    b = st.random_rules(list(feats.columns), n_rules=30, seed=5)
    assert a == b
    c = st.random_rules(list(feats.columns), n_rules=30, seed=6)
    assert a != c


def test_rule_position_is_binary_long_flat(null_tape):
    feats, _, _ = null_tape
    pos = st.rule_position(feats, {"feature": "mom_5", "op": ">", "q": 0.5, "long_short": False})
    vals = set(np.unique(pos.dropna().to_numpy()))
    assert vals <= {0.0, 1.0}


def test_rule_position_long_short(null_tape):
    feats, _, _ = null_tape
    pos = st.rule_position(feats, {"feature": "mom_5", "op": ">", "q": 0.5, "long_short": True})
    vals = set(np.unique(pos.dropna().to_numpy()))
    assert vals <= {-1.0, 1.0}


# ---- backtest invariants ---------------------------------------------------
def test_costs_lower_returns(control_tape):
    feats, fwd, _ = control_tape
    rule = {"feature": "f0_signal", "op": ">", "q": 0.5, "long_short": False}
    m0 = st.backtest_rule(feats, fwd, rule, cost_bps=0.0).mean()
    m1 = st.backtest_rule(feats, fwd, rule, cost_bps=20.0).mean()
    assert m0 > m1


def test_backtest_excess_of_cash(control_tape):
    """With rf>0, a flat (cash) day earns exactly 0 excess, not -rf."""
    feats, fwd, _ = control_tape
    # A rule that is flat on some days: when flat, excess return contribution is 0.
    rule = {"feature": "noise_a", "op": ">", "q": 0.5, "long_short": False}
    net = st.backtest_rule(feats, fwd, rule, rf_daily=0.0002, cost_bps=0.0)
    pos = st.rule_position(feats, rule)
    flat_days = net[(pos == 0)]
    assert np.allclose(flat_days.dropna().to_numpy(), 0.0)


def test_hac_tstat_finite(null_tape):
    feats, fwd, _ = null_tape
    net = st.backtest_rule(feats, fwd, {"feature": "mom_5", "op": ">", "q": 0.5, "long_short": False})
    assert np.isfinite(st.hac_tstat(net))


# ---- spine #1: the positive control ----------------------------------------
def test_harness_rediscovers_planted_edge(control_tape):
    """A faithful miner ranks a rule on the SIGNAL feature at (or near) the very top."""
    feats, fwd, truth = control_tape
    res = st.run_experiment(feats, fwd, n_rules=1000, rc_boot=300, seed=343)
    top5_features = list(res["table"].head(5)["feature"])
    assert truth["signal_feature"] in top5_features
    # The champion clears Bonferroni and the Reality Check is significant on real edge.
    assert res["pass_counts"]["n_pass_bonferroni"] >= 1
    assert res["rc"]["reality_check_p"] < 0.05


# ---- spine #2: the p-hacking verdict (the CONFIRMED axis) -------------------
def test_null_best_rule_looks_significant_naively(null_tape):
    """On a pure null, the luckiest of 1000 rules posts a naive |HAC t| >= 2 — the trap."""
    feats, fwd, _ = null_tape
    res = st.run_experiment(feats, fwd, n_rules=1000, rc_boot=300, seed=343)
    assert abs(res["best"]["hac_t"]) >= 2.0
    # And a whole bunch of rules clear the naive bar by chance.
    assert res["pass_counts"]["n_pass_naive"] >= 5


def test_null_correction_kills_the_mirage(null_tape):
    """The correction for the search: Bonferroni passes none and the Reality Check is
    NOT significant on the null — the champion is exactly what luck produces."""
    feats, fwd, _ = null_tape
    res = st.run_experiment(feats, fwd, n_rules=1000, rc_boot=500, seed=343)
    assert res["pass_counts"]["n_pass_bonferroni"] == 0
    assert res["rc"]["reality_check_p"] > 0.05


def test_expected_false_positive_count_is_positive(null_tape):
    feats, fwd, _ = null_tape
    res = st.run_experiment(feats, fwd, n_rules=1000, rc_boot=200, seed=343)
    assert res["pass_counts"]["expected_false_positives"] > 0
    assert res["pass_counts"]["single_test_alpha"] == pytest.approx(0.0455, abs=0.01)


def test_bonferroni_threshold_exceeds_two(null_tape):
    """For N=1000 the family-wise threshold is far above the naive |t|=2 bar."""
    feats, fwd, _ = null_tape
    res = st.run_experiment(feats, fwd, n_rules=1000, rc_boot=100, seed=343)
    assert res["pass_counts"]["bonferroni_t"] > 3.5


def test_reality_check_p_in_unit_interval(control_tape):
    feats, fwd, _ = control_tape
    res = st.run_experiment(feats, fwd, n_rules=200, rc_boot=200, seed=343)
    assert 0.0 <= res["rc"]["reality_check_p"] <= 1.0
