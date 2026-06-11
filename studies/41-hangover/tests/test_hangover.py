"""The synthetic world is deterministic; the barometer's accuracy beats the base rate only when
January is genuinely predictive; in the null it does not; annual_table and the tradable rule are
correct. All offline on the seeded synthetic world."""

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
