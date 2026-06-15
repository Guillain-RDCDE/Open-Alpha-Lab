"""Weight schedules, the lifecycle engine's invariants, and the study's spine:
the HMA glidepath trades return for lower variance — measurably and honestly."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from hundred_minus_age import strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# Weight schedule sanity
# ---------------------------------------------------------------------------

def test_hma_weight_is_100_minus_age():
    assert abs(st.hma_weight(30) - 0.70) < 1e-9
    assert abs(st.hma_weight(60) - 0.40) < 1e-9
    assert abs(st.hma_weight(65) - 0.35) < 1e-9


def test_hma_weight_floor():
    """Very old investors should not go below the floor."""
    assert st.hma_weight(95) >= 0.10
    assert st.hma_weight(100) >= 0.10


def test_hma_weight_decreases_with_age():
    ages = [25, 35, 45, 55, 65]
    weights = [st.hma_weight(a) for a in ages]
    assert all(weights[i] > weights[i + 1] for i in range(len(weights) - 1))


def test_hma_rising_weight_increases_with_age():
    ages = [25, 35, 45, 55, 65]
    weights = [st.hma_rising_weight(a) for a in ages]
    assert all(weights[i] < weights[i + 1] for i in range(len(weights) - 1))


def test_hma_rising_weight_bounds():
    assert abs(st.hma_rising_weight(25) - 0.30) < 1e-9
    assert abs(st.hma_rising_weight(65) - 0.70) < 1e-9


def test_fixed_weight_is_constant():
    assert st.fixed_weight(0.60) == 0.60
    assert st.fixed_weight(0.80) == 0.80


# ---------------------------------------------------------------------------
# Lifecycle engine invariants
# ---------------------------------------------------------------------------

def test_terminal_wealth_is_positive(synth_with_premium):
    returns, _ = synth_with_premium
    for strat in ("hma", "sixty_forty", "all_equity", "hma_rising"):
        curve = st.simulate_lifecycle(returns, strategy=strat, cost_bps=0)
        assert curve.iloc[-1] > 0, f"{strat} produced non-positive terminal wealth"


def test_curve_is_monotone_no_withdrawals(synth_no_premium):
    """With no withdrawals and no catastrophic crashes the curve should generally
    grow.  We check that terminal wealth exceeds the sum of contributions (=1 unit)."""
    returns, _ = synth_no_premium
    curve = st.simulate_lifecycle(returns, strategy="sixty_forty", cost_bps=0)
    # Sum of contributions = 1 by construction (1/(40*12) * 40*12)
    assert curve.iloc[-1] > 1.0


def test_costs_reduce_terminal_wealth(synth_with_premium):
    returns, _ = synth_with_premium
    curve_no_cost = st.simulate_lifecycle(returns, strategy="sixty_forty", cost_bps=0)
    curve_with_cost = st.simulate_lifecycle(returns, strategy="sixty_forty", cost_bps=20)
    assert curve_no_cost.iloc[-1] > curve_with_cost.iloc[-1]


def test_all_equity_earns_more_with_premium(synth_with_premium):
    """When equities earn a genuine premium, 100% equities should beat bonds-heavy."""
    returns, _ = synth_with_premium
    curve_eq = st.simulate_lifecycle(returns, strategy="all_equity", cost_bps=0)
    curve_hma = st.simulate_lifecycle(returns, strategy="hma", cost_bps=0)
    # HMA starts at 75% equity and slides down; all_equity has persistent premium
    # On this single 40-year tape, all_equity should be clearly ahead
    assert curve_eq.iloc[-1] > curve_hma.iloc[-1]


def test_insufficient_data_raises(synth_with_premium):
    returns, _ = synth_with_premium
    short = returns.iloc[:100]  # only ~8 years
    with pytest.raises(ValueError):
        st.simulate_lifecycle(short, strategy="hma")


# ---------------------------------------------------------------------------
# The study's spine — cohort statistics
# ---------------------------------------------------------------------------

def test_rolling_cohorts_returns_dataframe(synth_with_premium):
    returns, _ = synth_with_premium
    cohorts = st.rolling_cohorts(returns, n_years=5)
    assert "hma" in cohorts.columns
    assert "sixty_forty" in cohorts.columns
    assert "all_equity" in cohorts.columns
    assert len(cohorts) > 0


def test_cohort_stats_keys(synth_with_premium):
    returns, _ = synth_with_premium
    cohorts = st.rolling_cohorts(returns, n_years=5)
    stats = st.cohort_stats(cohorts)
    for key in ("mean", "median", "p05", "p95", "vol"):
        assert key in stats.columns, f"Missing stat: {key}"


def test_hma_lower_vol_than_all_equity(synth_with_premium):
    """The defining promise of the glidepath: lower variance of terminal outcomes."""
    returns, _ = synth_with_premium
    cohorts = st.rolling_cohorts(returns, n_years=10)
    stats = st.cohort_stats(cohorts)
    assert stats.loc["hma", "vol"] < stats.loc["all_equity", "vol"], (
        "HMA should produce lower variance of terminal wealth than 100% equities"
    )


def test_hac_tstat_finite(synth_with_premium):
    returns, _ = synth_with_premium
    # Use a short n_years so many cohorts fit in the 42-year tape
    cohorts = st.rolling_cohorts(returns, n_years=3)
    t = st.hac_tstat(
        cohorts["hma"].dropna().values,
        cohorts["sixty_forty"].dropna().values,
    )
    assert np.isfinite(t)


def test_path_stats_keys(synth_with_premium):
    returns, _ = synth_with_premium
    curve = st.simulate_lifecycle(returns, strategy="hma")
    stats = st.path_stats(curve)
    assert "terminal_wealth" in stats
    assert "max_drawdown" in stats
    assert stats["max_drawdown"] <= 0.0
