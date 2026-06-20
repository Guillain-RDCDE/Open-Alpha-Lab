"""The per-effect inference, the four correction procedures, and the study's two spines:
(1) on the global null every correction returns ~0 while naive over-rejects, and
(2) on a positive control the corrections recover real edges with controlled error,
with BH out-powering the FWER procedures when edges are faint."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from multiple_testing import strategy as st  # noqa: E402


# ---- per-effect inference --------------------------------------------------
def test_hac_tstat_zero_on_mean_zero_noise():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 0.01, 4000)
    assert abs(st.hac_tstat(x)) < 3.0          # essentially never far from 0


def test_hac_tstat_large_on_strong_drift():
    rng = np.random.default_rng(0)
    x = rng.normal(0.002, 0.01, 4000)          # clear positive mean
    assert st.hac_tstat(x) > 5.0


def test_two_sided_p_monotone():
    assert st.two_sided_p(0.0) > st.two_sided_p(2.0) > st.two_sided_p(4.0)
    assert 0.0 <= st.two_sided_p(3.0) <= 1.0


def test_alpha_to_t_roundtrip():
    # |t|=1.96 <-> two-sided alpha 0.05
    assert abs(st.alpha_to_t(0.05) - 1.96) < 0.02
    assert abs(st.two_sided_p(st.alpha_to_t(0.01)) - 0.01) < 1e-3


def test_block_bootstrap_ci_brackets_mean():
    rng = np.random.default_rng(1)
    x = rng.normal(0.001, 0.01, 3000)
    lo, hi = st.block_bootstrap_ci(x, n_boot=1000, seed=1)
    assert lo < x.mean() < hi


# ---- the four corrections --------------------------------------------------
def test_holm_at_least_as_powerful_as_bonferroni():
    """Holm rejects a superset of Bonferroni at the same alpha (uniformly more powerful)."""
    rng = np.random.default_rng(2)
    p = np.sort(rng.uniform(0, 0.1, 50))
    bonf = st.bonferroni_reject(p, alpha=0.05)
    holm = st.holm_reject(p, alpha=0.05)
    assert np.all(holm[bonf])                  # every Bonferroni rejection is a Holm rejection


def test_bh_at_least_as_powerful_as_holm():
    """BH (FDR) rejects a superset of Holm (FWER) at the same alpha."""
    rng = np.random.default_rng(3)
    p = np.sort(rng.uniform(0, 0.05, 50))
    holm = st.holm_reject(p, alpha=0.05)
    bh = st.bh_reject(p, alpha=0.05)
    assert bh.sum() >= holm.sum()
    assert np.all(bh[holm])


def test_bonferroni_threshold():
    p = np.array([0.001, 0.01, 0.04, 0.5])
    rej = st.bonferroni_reject(p, alpha=0.05)   # threshold 0.05/4 = 0.0125
    assert list(rej) == [True, True, False, False]


def test_bh_all_null_rejects_nothing():
    rng = np.random.default_rng(4)
    p = rng.uniform(0, 1, 200)                  # uniform p-values = global null
    assert st.bh_reject(p, alpha=0.05).sum() <= 5  # ~ FDR-controlled, near 0


# ---- spine #1: the global null ---------------------------------------------
def test_null_corrections_find_nothing(null_battery):
    """On a battery with nothing real, every correction returns ~0; naive over-rejects."""
    rets, truth, _ = null_battery
    res = st.run_experiment(rets, truth, alpha=0.05, t_thresh=2.0)
    c = res["counts"]
    assert c["bonferroni"] == 0
    assert c["holm"] == 0
    assert c["bh"] <= 1
    # the naive count is positive (false positives by construction) often enough to matter
    assert c["naive"] >= 1


def test_fwer_monte_carlo_calibration():
    """Naive FWER ~1 under the global null; FWER procedures hold near/below alpha."""
    mc = st.fwer_monte_carlo(m_effects=100, n_days=2520, n_reps=60, alpha=0.05, seed=346)
    assert mc["fwer"]["naive"] > 0.8           # naive almost always fires a false positive
    assert mc["fwer"]["bonferroni"] <= 0.20    # FWER held near alpha (small-rep slack)
    assert mc["fwer"]["holm"] <= 0.20
    assert mc["avg_false_positives"]["bonferroni"] < mc["avg_false_positives"]["naive"]


# ---- spine #2: the positive control ----------------------------------------
def test_strong_control_recovers_all_with_zero_false(strong_battery):
    """With a strong planted edge the corrections find all real effects and add no false ones."""
    rets, truth, _ = strong_battery
    sc = st.run_experiment(rets, truth, alpha=0.05, t_thresh=2.0)["score"]
    for proc in ("bonferroni", "holm", "bh"):
        assert sc[proc]["true_pos"] == 10
        assert sc[proc]["false_pos"] == 0
    # the naive count drags in false positives (its failing)
    assert sc["naive"]["false_pos"] >= 1


def test_weak_control_bh_out_powers_fwer(weak_battery):
    """The FWER-vs-FDR trade-off: with faint edges, BH finds more real effects than Holm."""
    rets, truth, _ = weak_battery
    sc = st.run_experiment(rets, truth, alpha=0.05, t_thresh=2.0)["score"]
    assert sc["bh"]["power"] > sc["holm"]["power"]
    assert sc["bh"]["power"] > sc["bonferroni"]["power"]
    # BH's realised FDR stays modest (the point of FDR control)
    assert sc["bh"]["fdr"] <= 0.20


# ---- costs & invariants ----------------------------------------------------
def test_costs_lower_means(strong_battery):
    """Charging turnover costs reduces each effect's net mean (monotone in cost)."""
    rets, _, _ = strong_battery
    m0 = st.screen_battery(rets, cost_bps=0.0)["mean_excess_ann"].mean()
    m1 = st.screen_battery(rets, cost_bps=20.0)["mean_excess_ann"].mean()
    assert m1 < m0


def test_screen_table_sorted_by_abs_t(null_battery):
    rets, _, _ = null_battery
    tab = st.screen_battery(rets)
    t = tab["hac_t"].abs().to_numpy()
    assert np.all(np.diff(t) <= 1e-9)          # non-increasing


def test_discovery_counts_keys(null_battery):
    rets, _, _ = null_battery
    res = st.run_experiment(rets)
    assert set(res["counts"]) == {"m_effects", "naive", "bonferroni", "holm", "bh"}
