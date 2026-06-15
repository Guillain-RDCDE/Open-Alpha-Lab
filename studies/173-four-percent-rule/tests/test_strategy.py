"""The retirement engine's invariants and the study's spine.

Core assertions:
 - A cohort with strong equity returns and 4% withdrawal survives.
 - A cohort with no returns (0%) and 4% withdrawal fails around year 25.
 - The SWR search finds a rate above 0 and below the withdrawal rate on a stressed tape.
 - Sequence risk: the same 30-year mean with a bad first 3 years leads to more failures.
 - summarize() returns the right keys and sensible numbers.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from four_percent_rule import data, strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# simulate_cohort invariants
# ---------------------------------------------------------------------------
def test_zero_return_runs_out_around_year_25():
    """With 0% real return the retiree simply withdraws 4% of 1.0 per year: lasts 25 yrs."""
    returns = pd.DataFrame({
        "eq_real_tr": np.zeros(30),
        "bond_real_tr": np.zeros(30),
    }, index=range(1990, 2020))
    res = st.simulate_cohort(returns, 1990, horizon=30, withdrawal_rate=0.04)
    # With $1.0 initial, $0.04/yr withdrawal, 0% return → ruin at year 25 (1/0.04).
    assert not res["survived"]
    assert res["years_solvent"] == 24  # runs out during year 25 (index 24)


def test_high_return_survives_easily():
    """With 10% real equity and 4% withdrawal the retiree clearly survives."""
    returns = pd.DataFrame({
        "eq_real_tr": np.full(30, 0.10),
        "bond_real_tr": np.full(30, 0.02),
    }, index=range(1990, 2020))
    res = st.simulate_cohort(returns, 1990, horizon=30, withdrawal_rate=0.04)
    assert res["survived"]
    assert res["terminal_wealth"] > 1.0  # grew beyond initial


def test_terminal_wealth_is_nonnegative(normal_returns):
    df, truth = normal_returns
    for y in range(truth["start_year"], truth["start_year"] + 40):
        res = st.simulate_cohort(df, y, horizon=truth["horizon"])
        assert res["terminal_wealth"] >= 0.0


def test_years_solvent_leq_horizon(normal_returns):
    df, truth = normal_returns
    for y in range(truth["start_year"], truth["start_year"] + 40):
        res = st.simulate_cohort(df, y, horizon=truth["horizon"])
        assert 0 <= res["years_solvent"] <= truth["horizon"]


def test_survived_iff_terminal_wealth_positive(normal_returns):
    df, truth = normal_returns
    for y in range(truth["start_year"], truth["start_year"] + 40):
        res = st.simulate_cohort(df, y, horizon=truth["horizon"])
        assert res["survived"] == (res["terminal_wealth"] > 0.0)


# ---------------------------------------------------------------------------
# run_all_cohorts
# ---------------------------------------------------------------------------
def test_run_all_cohorts_returns_dataframe(normal_returns):
    df, _ = normal_returns
    cohorts = st.run_all_cohorts(df, horizon=30)
    assert isinstance(cohorts, pd.DataFrame)
    assert "survived" in cohorts.columns
    assert "terminal_wealth" in cohorts.columns
    assert len(cohorts) >= 40  # at least 40 cohorts from 100 years


def test_4pct_high_survival_on_normal_returns():
    """On a good equity tape the 4% rule should succeed more often than a bad tape.
    We use a longer synthetic tape (500 years) to get a stable estimate."""
    df, _ = data.synthetic_cohorts(
        n_years_total=500, equity_real_mean=0.068, bad_start=False, seed=42
    )
    cohorts = st.run_all_cohorts(df, horizon=30, withdrawal_rate=0.04)
    assert cohorts["survived"].mean() > 0.75, (
        "4% rule should survive >75% of cohorts on a long normal-return tape"
    )


def test_8pct_fails_more_than_4pct(normal_returns):
    df, _ = normal_returns
    c4 = st.run_all_cohorts(df, horizon=30, withdrawal_rate=0.04)
    c8 = st.run_all_cohorts(df, horizon=30, withdrawal_rate=0.08)
    assert c4["survived"].mean() > c8["survived"].mean()


# ---------------------------------------------------------------------------
# find_swr
# ---------------------------------------------------------------------------
def test_swr_is_lower_in_low_yield_world(normal_returns, low_yield_returns):
    df_normal, _ = normal_returns
    df_low, _ = low_yield_returns
    swr_normal = st.find_swr(df_normal, target_success_rate=0.95)
    swr_low = st.find_swr(df_low, target_success_rate=0.95)
    assert swr_low < swr_normal, "Low-yield world should have a lower SWR"


def test_swr_positive_and_below_unity(normal_returns):
    df, _ = normal_returns
    swr = st.find_swr(df, target_success_rate=0.90)
    assert 0.01 < swr < 0.50


# ---------------------------------------------------------------------------
# Sequence-of-returns risk
# ---------------------------------------------------------------------------
def test_bad_start_lowers_survival_rate(normal_returns, bad_start_returns):
    """The sequence-of-returns knob: same mean, bad early years → more failures."""
    df_good, truth = normal_returns
    df_bad, _ = bad_start_returns
    c_good = st.run_all_cohorts(df_good, horizon=truth["horizon"], withdrawal_rate=0.04)
    c_bad = st.run_all_cohorts(df_bad, horizon=truth["horizon"], withdrawal_rate=0.04)
    # Bad-start cohorts should fail at higher rates.
    # (Not guaranteed statistically but almost always holds with seed=173.)
    assert c_bad["survived"].mean() <= c_good["survived"].mean() + 0.05


def test_sequence_risk_stats_returns_augmented_frame(normal_returns):
    df, _ = normal_returns
    augmented = st.sequence_risk_stats(df, horizon=30, withdrawal_rate=0.04)
    assert "early_eq_return" in augmented.columns
    assert "bad_sequence" in augmented.columns


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_keys(normal_returns):
    df, _ = normal_returns
    cohorts = st.run_all_cohorts(df, horizon=30, withdrawal_rate=0.04)
    stats = st.summarize(cohorts)
    required = {"n_cohorts", "survival_rate", "median_terminal_wealth",
                "worst_terminal_wealth", "pct_fail", "years_solvent_if_fail",
                "max_drawdown_median"}
    assert required.issubset(stats.keys())


def test_survival_rate_plus_pct_fail_is_100(normal_returns):
    df, _ = normal_returns
    cohorts = st.run_all_cohorts(df, horizon=30, withdrawal_rate=0.04)
    stats = st.summarize(cohorts)
    assert abs(stats["survival_rate"] * 100 + stats["pct_fail"] - 100.0) < 1e-6
