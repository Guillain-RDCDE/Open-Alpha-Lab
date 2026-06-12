"""The synthetic world is deterministic; an inverted curve precedes lower forward equity returns when it
forecasts and not in the null; curve_slope is long minus short. All offline on the seeded world."""
import numpy as np
from inverted import data, strategy as st


def test_world_deterministic(predictive_world):
    df, truth = predictive_world
    df2, _ = data.synthetic_world(predicts=0.05, seed=66)
    assert np.allclose(df.to_numpy(), df2.to_numpy())
    assert truth.curve_informative


def test_inverted_precedes_lower_returns(predictive_world):
    df, _ = predictive_world
    c = st.conditional_forward(df["curve"], df["eq"], horizon=18)
    assert c["inverted_fwd"] < c["normal_fwd"]   # weaker returns after an inversion


def test_null_world_no_gap(null_world):
    df, _ = null_world
    c = st.conditional_forward(df["curve"], df["eq"], horizon=18)
    assert abs(c["gap"]) < 0.05


def test_curve_slope_is_long_minus_short():
    import pandas as pd
    idx = pd.date_range("2000-01-31", periods=3, freq="ME")
    lo = pd.Series([4.0, 3.0, 2.0], index=idx); sh = pd.Series([1.0, 3.5, 5.0], index=idx)
    s = st.curve_slope(lo, sh)
    assert np.allclose(s.to_numpy(), [3.0, -0.5, -3.0])


def test_inverted_share_reasonable(predictive_world):
    df, _ = predictive_world
    c = st.conditional_forward(df["curve"], df["eq"], 18)
    assert 0.0 < c["inverted_share"] < 0.6
