"""The synthetic world is deterministic; CAPE forecasts forward returns negatively when it should and
not in the null; the valuation buckets are monotone (cheap > expensive); the implied-return fit slopes
the right way. All offline on the seeded synthetic world."""

import numpy as np

from tide_table import data, strategy as st


def test_world_deterministic(predictive_world):
    df, truth = predictive_world
    df2, _ = data.synthetic_world(predicts=0.9, seed=56)
    assert np.allclose(df.to_numpy(), df2.to_numpy())
    assert truth.cape_informative


def test_cape_forecasts_when_it_should(predictive_world):
    df, _ = predictive_world
    fc = st.forward_corr(df["cape"], df["fwd10"])
    assert fc["corr_cape"] < 0.0     # high CAPE → low future returns
    assert fc["r2"] > 0.1            # and explanatory


def test_null_world_no_forecast(null_world):
    df, _ = null_world
    assert abs(st.forward_corr(df["cape"], df["fwd10"])["corr_cape"]) < 0.15


def test_buckets_monotone(predictive_world):
    df, _ = predictive_world
    b = st.bucket_returns(df["cape"], df["fwd10"])
    assert b.loc["cheap"] > b.loc["expensive"]   # cheap valuations precede higher returns


def test_implied_return_slope_positive_in_yield(predictive_world):
    """fwd return rises with the earnings yield (1/CAPE) — i.e. cheaper ⇒ higher expected return."""
    df, _ = predictive_world
    _, slope = st.implied_return(df["cape"], df["fwd10"])
    assert slope > 0.0
