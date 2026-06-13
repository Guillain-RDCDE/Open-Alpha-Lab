"""Strategy-engine tests for Study 94 (Level-Pegging), including the spine tests."""

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from level_pegging import strategy  # noqa: E402


def test_stats_on_known_series():
    # A flat 0.1%/day series: CAGR and Sharpe must be finite and positive.
    ret = pd.Series(np.full(1000, 0.001))
    d = strategy._stats(ret)
    assert d["cagr"] > 0
    assert d["max_dd"] == 0.0  # monotone up -> no drawdown
    assert d["final"] > 1.0


def test_fee_reduces_return(broadening_panel):
    frame = broadening_panel[0]
    free = strategy.arm_returns(frame, "ew", fee_bps_yr=0.0)
    paid = strategy.arm_returns(frame, "ew", fee_bps_yr=50.0)
    assert paid["final"] < free["final"]
    assert paid["cagr"] < free["cagr"]


def test_race_diff_is_ew_minus_cw(broadening_panel):
    frame = broadening_panel[0]
    r = strategy.race(frame)
    expected = r["ew"]["net"] - r["cw"]["net"]
    assert np.allclose(r["diff"].to_numpy(), expected.to_numpy())


def test_hac_tstat_zero_mean_is_small():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, 5000)
    assert abs(strategy.hac_tstat(x)) < 3.0


def test_alpha_beta_recovers_planted_sign_broadening(broadening_panel):
    ab = strategy.alpha_beta(broadening_panel[0])
    assert ab["alpha_ann"] > 0  # small-cap premium -> positive EW alpha
    assert 0.0 < ab["beta"] < 2.0
    assert 0.0 < ab["r2"] < 1.0


def test_split_contrast_returns_pvalue(broadening_panel):
    r = strategy.race(broadening_panel[0])
    sc = strategy.split_contrast(r["diff"], "2010-01-01", n_boot=500)
    assert 0.0 <= sc["p_boot"] <= 1.0
    assert "early_ann" in sc and "late_ann" in sc


# --- spine tests: the harness must read a planted answer correctly ---
def test_spine_broadening_equal_weight_wins(broadening_panel):
    """When small names carry the premium, equal-weight must out-earn cap-weight with +alpha."""
    frame = broadening_panel[0]
    r = strategy.race(frame)
    ab = strategy.alpha_beta(frame)
    assert r["ew"]["final"] > r["cw"]["final"]
    assert r["ew"]["sharpe"] > r["cw"]["sharpe"]
    assert ab["alpha_ann"] > 0 and ab["t_alpha"] > 2.0


def test_spine_megacap_equal_weight_loses(megacap_panel):
    """When a few mega names dominate, cap-weight must win and equal-weight's alpha is negative."""
    frame = megacap_panel[0]
    r = strategy.race(frame)
    ab = strategy.alpha_beta(frame)
    assert r["cw"]["final"] > r["ew"]["final"]
    assert r["cw"]["sharpe"] > r["ew"]["sharpe"]
    assert ab["alpha_ann"] < 0 and ab["t_alpha"] < -2.0
