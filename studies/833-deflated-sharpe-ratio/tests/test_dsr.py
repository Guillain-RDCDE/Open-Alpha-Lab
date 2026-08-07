"""Offline, fixed-seed tests for the Deflated-Sharpe-Ratio machinery.

The pitfall appears at the planted magnitude (the best of N empty strategies inflates with N,
tracking the expected-maximum-Sharpe formula) and the correction removes it (the DSR of the
null winner shrinks to a coin flip while an honest single strategy keeps a high DSR). All
deterministic, offline, synthetic-only.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from deflated_sharpe import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Determinism & the data layer
# --------------------------------------------------------------------------- #
def test_null_panel_deterministic(null_pool):
    p2 = data.null_panel(n_strategies=1000, n_days=1260, ann_vol=0.15, seed=833)
    assert np.allclose(null_pool, p2)


def test_null_true_edge_is_zero(null_pool):
    # Every column has population Sharpe 0: the cross-column MEAN Sharpe is ~0.
    srs = st.panel_sr_per_period(null_pool)
    assert abs(np.nanmean(srs)) < 0.02          # per-period, ~0
    assert abs(float(np.nanmean(srs)) * np.sqrt(252)) < 0.4   # annualised, ~0


def test_honest_has_real_edge(honest):
    # The positive control's realised annualised Sharpe is genuinely near its planted 1.0.
    assert 0.6 < st.sharpe_ann(honest) < 1.4


# --------------------------------------------------------------------------- #
# The pitfall — the best Sharpe inflates with the trial count
# --------------------------------------------------------------------------- #
def test_expected_max_sharpe_monotone_and_zero_at_one():
    assert st.expected_max_sharpe(1) == 0.0            # a single hypothesis: no inflation
    vals = [st.expected_max_sharpe(n, var_sharpe=1.0) for n in (2, 10, 100, 1000)]
    assert all(b > a for a, b in zip(vals, vals[1:]))  # strictly increasing in N


def test_best_of_N_inflates_with_N():
    v = 1.0 / (1260 - 1)
    small = st.best_sharpe_experiment(data.null_panel(10, 1260, 0.15, 1))
    big = st.best_sharpe_experiment(data.null_panel(1000, 1260, 0.15, 1))
    assert big["obs_max_sharpe_ann"] > small["obs_max_sharpe_ann"]   # more trials, luckier max
    assert big["exp_max_sharpe_ann"] > small["exp_max_sharpe_ann"]


def test_observed_max_tracks_formula():
    # Averaged over seeds, the observed best Sharpe matches E[max] within a modest tolerance.
    ic = st.inflation_curve(n_grid=(10, 100, 1000), n_days=1260, ann_vol=0.15,
                            n_seeds=20, base_seed=833)
    for o, p in zip(ic["obs_best_ann"], ic["pred_ann"]):
        assert abs(o - p) < 0.15                      # observed ≈ formula


def test_cross_trial_sr_variance_matches_theory(null_pool):
    # Under H0 each per-period SR estimate has sampling var ~ 1/(T-1).
    srs = st.panel_sr_per_period(null_pool)
    v = float(np.nanvar(srs, ddof=1))
    assert abs(v - 1.0 / (1260 - 1)) < 3e-4


# --------------------------------------------------------------------------- #
# The correction — the DSR removes the inflation
# --------------------------------------------------------------------------- #
def test_dsr_shrinks_null_winner(null_pool):
    srs = st.panel_sr_per_period(null_pool)
    winner = null_pool[:, int(np.nanargmax(srs))]
    d = st.deflated_sharpe_ratio(winner, n_trials=1000)
    assert d["sharpe_ann"] > 1.0                 # the raw Sharpe looks great
    assert d["dsr"] < 0.95                        # the DSR sees through it (consistent with luck)
    assert abs(d["deflated_excess_ann"]) < 0.5   # winner shrunk back toward zero


def test_naive_screen_fools_but_dsr_does_not(null_pool):
    srs = st.panel_sr_per_period(null_pool)
    winner = null_pool[:, int(np.nanargmax(srs))]
    assert abs(st.one_sample_t(winner)) >= 2.0   # naive |t| screen FIRES on an empty winner
    assert st.deflated_sharpe_ratio(winner, 1000)["dsr"] < 0.95   # DSR does not


def test_honest_single_strategy_keeps_high_dsr(honest):
    d = st.deflated_sharpe_ratio(honest, n_trials=1)   # a single hypothesis, no search
    assert d["sr0"] == 0.0                             # N=1 => no expected-max bar
    assert d["dsr"] > 0.90                             # the correction spares genuine skill


def test_null_calibration_naive_vs_dsr():
    cal = st.null_dsr_calibration(n_trials=1000, n_days=1260, ann_vol=0.15,
                                  n_seeds=20, base_seed=833)
    assert cal["naive_fire_rate"] > 0.8              # naive screen manufactures winners
    assert cal["dsr_fire_rate"] <= 0.15              # DSR near the nominal false-positive rate
    assert 0.4 < cal["mean_dsr"] < 0.6               # winner is a coin flip
    assert abs(cal["mean_deflated_excess_ann"]) < 0.2


def test_honest_control_beats_null_control():
    hc = st.honest_control(true_ann_sharpe=1.0, n_days=1260, ann_vol=0.15,
                           n_seeds=20, base_seed=833)
    assert hc["mean_dsr"] > 0.9                      # honest single strategy: high DSR


# --------------------------------------------------------------------------- #
# The Mirage — in-sample champion collapses out-of-sample, dies on costs
# --------------------------------------------------------------------------- #
def test_is_champion_collapses_oos(null_pool):
    champ = st.in_sample_champion(null_pool, frac=0.5)
    assert champ["is_sharpe_ann"] > 1.0                        # gorgeous in-sample
    assert champ["oos_sharpe_ann"] < champ["is_sharpe_ann"]    # evaporates live
    assert abs(champ["oos_t_nw"]) < 2.0                        # OOS indistinguishable from 0


def test_costs_reduce_net(null_pool):
    champ = st.in_sample_champion(null_pool, frac=0.5)
    oos = null_pool[champ["is_n"]:, champ["champion"]]
    gross = st.timer_stats(oos, cost_bps=0.0)["net_bps"]
    net = st.timer_stats(oos, cost_bps=5.0)["net_bps"]
    assert net < gross


# --------------------------------------------------------------------------- #
# Inference primitives behave
# --------------------------------------------------------------------------- #
def test_psr_in_unit_interval(honest):
    p = st.probabilistic_sharpe_ratio(honest)
    assert 0.0 <= p <= 1.0


def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=10) - st.one_sample_t(x)) < 0.6


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi


def test_fingerprint_stable():
    a = data.config_fingerprint(1000, 1260, 0.15, 833)
    b = data.config_fingerprint(1000, 1260, 0.15, 833)
    assert a == b and len(a) == 12
