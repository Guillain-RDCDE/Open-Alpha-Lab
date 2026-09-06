"""Strategy tests for Study 968 — resamplers, intervals and coverage, all offline."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from boot_choice import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# The resamplers themselves
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("method", st.METHODS)
def test_resamplers_return_a_full_sample_of_valid_indices(method):
    rng = np.random.default_rng(0)
    idx = st._resample_indices(500, method, 10, rng)
    assert idx.size == 500
    assert idx.min() >= 0 and idx.max() < 500


def test_moving_block_never_wraps_and_circular_always_can():
    rng = np.random.default_rng(1)
    for _ in range(50):
        idx = st._resample_indices(100, "moving", 10, rng)
        d = np.diff(idx)
        # inside a block the index always increases by 1; a wrap would show -99
        assert (d[d < 0] > -99).all() or (d >= 0).all()
    wraps = 0
    for s in range(200):
        idx = st._resample_indices(20, "circular", 5, np.random.default_rng(s))
        wraps += int(np.any(np.diff(idx) < -10))
    assert wraps > 0


def test_stationary_block_lengths_average_to_the_parameter():
    rng = np.random.default_rng(2)
    idx = st._resample_indices(20000, "stationary", 10, rng)
    breaks = int(np.sum(np.diff(idx) != 1)) + 1
    assert 20000 / breaks == pytest.approx(10, rel=0.25)


def test_block_of_one_is_the_iid_bootstrap():
    a = st._resample_indices(300, "circular", 1, np.random.default_rng(3))
    b = st._resample_indices(300, "iid", 1, np.random.default_rng(3))
    assert np.array_equal(a, b)


def test_default_block_follows_the_cube_root_rule():
    assert st.default_block(1000) == 10
    assert st.default_block(8000) == 20
    assert st.default_block(10) >= 2


# --------------------------------------------------------------------------- #
# Intervals
# --------------------------------------------------------------------------- #
def test_statistic_matches_hand_computation():
    x = np.array([0.01, -0.005, 0.02, 0.0])
    assert st.statistic(x, "mean") == pytest.approx(x.mean())
    assert st.statistic(x, "sharpe") == pytest.approx(
        x.mean() / x.std(ddof=1) * np.sqrt(252))


def test_bootstrap_ci_brackets_the_point_estimate(planted):
    r, _ = planted
    for m in st.METHODS:
        ci = st.bootstrap_ci(r, "sharpe", m, n_boot=400)
        assert ci["ci_low"] < ci["point"] < ci["ci_high"]
        assert ci["width"] > 0


def test_bootstrap_is_deterministic_given_a_seed(planted):
    r, _ = planted
    a = st.bootstrap_ci(r, "sharpe", "circular", n_boot=300, seed=5)
    b = st.bootstrap_ci(r, "sharpe", "circular", n_boot=300, seed=5)
    c = st.bootstrap_ci(r, "sharpe", "circular", n_boot=300, seed=6)
    assert a["ci_low"] == b["ci_low"] and a["ci_high"] == b["ci_high"]
    assert a["ci_low"] != c["ci_low"]


def test_block_bootstrap_is_wider_than_iid_on_autocorrelated_data():
    """The whole point: dependence inflates the true standard error, and only blocks see it."""
    r, _ = data.synthetic_returns(n_years=10, ar1=0.30, signal_strength=1.0, seed=968)
    iid = st.bootstrap_ci(r, "mean", "iid", n_boot=800)
    blk = st.bootstrap_ci(r, "mean", "circular", n_boot=800, block=21)
    # Theory says the inflation factor for AR(1)=0.3 is sqrt((1+phi)/(1-phi)) = 1.36; a
    # finite block recovers part of it, so the test asks for a clear direction, not the limit.
    assert blk["width"] > iid["width"] * 1.08


def test_methods_agree_on_iid_data(iid_returns):
    r, _ = iid_returns
    widths = [st.bootstrap_ci(r, "mean", m, n_boot=800)["width"] for m in st.METHODS]
    assert max(widths) / min(widths) < 1.25


def test_analytic_ci_matches_the_engine(planted):
    r, _ = planted
    from quantlab.analytics import sharpe_with_se
    a = st.analytic_ci(r, "sharpe", method="lo")
    engine = sharpe_with_se(pd.Series(r), method="lo")
    assert a["point"] == pytest.approx(engine["sharpe_ann"])
    assert a["width"] == pytest.approx(2 * 1.959963984540054 * engine["se_ann"])


# --------------------------------------------------------------------------- #
# Coverage
# --------------------------------------------------------------------------- #
def test_coverage_is_near_nominal_on_iid_data():
    def sampler(seed):
        return data.synthetic_returns(n_years=5, ar1=0.0, signal_strength=0.0, seed=seed)
    cov = st.coverage_experiment(sampler, "mu_daily", "mean", n_reps=120, n_boot=250,
                                 methods=("iid", "circular"))
    for m in ("iid", "circular"):
        assert 0.88 <= cov.loc[m, "coverage"] <= 1.0


def test_iid_bootstrap_undercovers_when_returns_are_autocorrelated():
    def sampler(seed):
        return data.synthetic_returns(n_years=5, ar1=0.30, signal_strength=1.0, seed=seed)
    cov = st.coverage_experiment(sampler, "mu_daily", "mean", n_reps=150, n_boot=250,
                                 methods=("iid", "circular"), block=21)
    assert cov.loc["iid", "coverage"] < cov.loc["circular", "coverage"]
    assert cov.loc["iid", "coverage"] < 0.93


def test_coverage_table_shape_and_tails():
    def sampler(seed):
        return data.synthetic_returns(n_years=4, seed=seed)
    cov = st.coverage_experiment(sampler, "sharpe_ann", "sharpe", n_reps=60, n_boot=200)
    assert set(cov.index) == set(st.METHODS) | {"analytic"}
    assert np.allclose(cov["coverage"] + cov["miss_low"] + cov["miss_high"], 1.0, atol=1e-9)
    assert (cov["mean_width"] > 0).all()


def test_block_sweep_is_monotone_in_width_where_dependence_exists():
    def sampler(seed):
        return data.synthetic_returns(n_years=4, ar1=0.25, signal_strength=1.0, seed=seed)
    sw = st.block_sweep(sampler, "mu_daily", "mean", blocks=(1, 10, 40), n_reps=60, n_boot=200)
    assert sw.loc[40, "mean_width"] > sw.loc[1, "mean_width"]
    assert sw.loc[40, "coverage"] >= sw.loc[1, "coverage"]


def test_dependence_profile_reads_the_planted_structure():
    r, _ = data.synthetic_returns(n_years=10, ar1=0.25, signal_strength=1.0, seed=968)
    p = st.dependence_profile(r)
    assert p["ar1"] > 0.15
    assert p["abs_ar1"] > 0.05
    assert p["kurtosis"] > 1.0


def test_real_tape_intervals_include_every_method(planted):
    r, _ = planted
    tbl = st.real_tape_intervals(pd.Series(r), "sharpe", n_boot=400)
    assert set(st.METHODS) <= set(tbl.index)
    assert (tbl["ci_high"] > tbl["ci_low"]).all()


# --------------------------------------------------------------------------- #
# The verdict rule
# --------------------------------------------------------------------------- #
def _headline(**over):
    h = {"max_coverage_spread": 0.08, "worst_coverage_sharpe": 0.84,
         "best_coverage_sharpe": 0.93, "iid_coverage_ar1": 0.86,
         "best_method": "Stationary (Politis-Romano 1994)", "best_worst_case_gap": 0.018,
         "max_real_width_ratio": 0.3, "spy_ci_low": 0.2, "spy_ci_high": 0.9}
    h.update(over)
    return h


def test_verdict_signal_ladder():
    assert st.verdict(_headline())["signal"] == "Real"
    assert st.verdict(_headline(max_coverage_spread=0.03))["signal"] == "Weak"
    assert st.verdict(_headline(max_coverage_spread=0.005))["signal"] == "None"


def test_verdict_usefulness_ladder():
    assert st.verdict(_headline())["trad"] == "Useful"
    assert st.verdict(_headline(best_worst_case_gap=0.04))["trad"] == "Fragile"
    assert st.verdict(_headline(best_worst_case_gap=0.09))["trad"] == "Mirage"


def test_verdict_prose_quotes_its_inputs():
    v = st.verdict(_headline(worst_coverage_sharpe=0.77))
    assert "77%" in v["signal_why"] and "77%" in v["one_sentence"]
    assert set(v) == {"signal", "signal_why", "trad", "trad_why", "one_sentence"}
