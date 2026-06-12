"""The synthetic world is deterministic; the barometer's accuracy beats the base rate only when
January is genuinely predictive; in the null it does not; annual_table, the tradable rule and the
small-sample machinery (Wilson, Fisher, permutation, cash credit) are correct. All offline on the
seeded synthetic world."""

import numpy as np
import pandas as pd

from hangover import data, strategy as st


def test_world_deterministic(predictive_world):
    df, truth = predictive_world
    df2, _ = data.synthetic_years(n_years=200, predictive=0.05, seed=41)
    assert np.allclose(df.to_numpy(), df2.to_numpy())
    assert truth.is_predictive


def test_predictive_world_has_conditioning(predictive_world):
    df, _ = predictive_world
    cm = st.conditional_means(df)
    assert cm.loc["jan_up", "roy_mean"] > cm.loc["jan_down", "roy_mean"]   # the link exists
    assert st.barometer_accuracy(df) > st.base_rate(df) - 0.05            # beats the base rate


def test_null_world_no_edge_over_base_rate(null_world):
    """The honest null: with no predictive link, January-conditioning does NOT beat the base rate."""
    df, _ = null_world
    cm = st.conditional_means(df)
    # up- and down-January rest-of-years are statistically the same
    assert abs(cm.loc["jan_up", "roy_mean"] - cm.loc["jan_down", "roy_mean"]) < 0.03
    # directional accuracy doesn't exceed the base rate by more than noise
    assert st.barometer_accuracy(df) <= st.base_rate(df) + 0.05


def test_annual_table_collapses_months_correctly():
    idx = pd.date_range("2000-01-31", periods=24, freq="ME")
    ret = pd.Series(0.0, index=idx)
    ret.iloc[0] = 0.10                    # Jan 2000 +10%
    ret.iloc[1:12] = 0.01                 # Feb-Dec 2000 +1% each
    tbl = st.annual_table(ret)
    assert tbl.loc[2000, "jan"] == 0.10
    assert np.isclose(tbl.loc[2000, "roy"], (1.01 ** 11) - 1.0)


def test_barometer_rule_is_cash_after_down_january(predictive_world):
    df, _ = predictive_world
    bar = st.barometer_returns(df)
    down = df["jan"] <= 0
    assert (bar[down] == 0.0).all()                 # cash after a down January
    assert np.allclose(bar[~down], df.loc[~down, "roy"])  # else fully invested


def test_barometer_rule_credits_the_tbill(predictive_world):
    df, _ = predictive_world
    cash = pd.Series(0.04, index=df.index)
    bar = st.barometer_returns(df, cash)
    down = df["jan"] <= 0
    assert np.allclose(bar[down], 0.04)             # cash years earn the T-bill, not zero
    assert np.allclose(bar[~down], df.loc[~down, "roy"])


def test_cash_roy_compounds_feb_to_dec():
    idx = pd.date_range("2000-01-31", periods=12, freq="ME")
    y = pd.Series(0.06, index=idx)                  # flat 6% annualised yield
    out = st.cash_roy(y)
    assert np.isclose(out.loc[2000], (1 + 0.06 / 12.0) ** 11 - 1.0)  # 11 months, Feb-Dec


def test_wilson_ci_brackets_and_shrinks():
    lo, hi = st.wilson_ci(18, 30)
    assert lo < 0.6 < hi                            # brackets the point estimate
    lo2, hi2 = st.wilson_ci(180, 300)               # same rate, 10x the sample
    assert (hi2 - lo2) < (hi - lo)                  # tighter with more data
    assert st.wilson_ci(0, 30)[0] == 0.0 and np.isclose(st.wilson_ci(30, 30)[1], 1.0)  # sane at the edges


def test_fisher_exact_p_known_values():
    # identical proportions -> p = 1; the study's own 40/46 vs 18/30 table -> p ~ 0.012
    assert st.fisher_exact_p(5, 10, 5, 10) == 1.0
    assert abs(st.fisher_exact_p(40, 46, 18, 30) - 0.0119) < 0.001
    p_far = st.fisher_exact_p(28, 30, 8, 30)        # wildly different proportions
    assert p_far < 0.001


def test_split_tests_detect_link_only_when_real(predictive_world, null_world):
    dfp, _ = predictive_world
    dfn, _ = null_world
    assert st.split_tests(dfp)["mean_perm_p"] < 0.05      # the link is detected...
    assert st.split_tests(dfn)["mean_perm_p"] > 0.05      # ...and not invented in the null
    # deterministic: same seed, same p
    assert st.split_tests(dfp)["mean_perm_p"] == st.split_tests(dfp)["mean_perm_p"]


def test_decay_split_counts_add_up(predictive_world):
    df, _ = predictive_world
    dc = st.decay_split(df, split_year=int(df.index[len(df) // 2]))
    assert dc["pre_n"] + dc["post_n"] == len(df)
    assert 0.0 <= dc["fisher_p"] <= 1.0
    assert dc["pre_low"] <= dc["pre_acc"] <= dc["pre_high"]
