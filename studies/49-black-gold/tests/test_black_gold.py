"""The synthetic world is deterministic; the predictive regression recovers a significant negative
slope only when oil really forecasts equities; the null shows nothing; the timing rule beats
buy-and-hold when the signal is real; and the rule is in cash unless oil fell. Offline, seeded."""

import numpy as np

from black_gold import data, strategy as st


def test_world_deterministic(predictive_world):
    df, truth = predictive_world
    df2, _ = data.synthetic_world(oil_loads=-0.10, seed=49)
    assert np.allclose(df.to_numpy(), df2.to_numpy())
    assert truth.oil_predicts


def test_predictive_world_negative_significant_slope(predictive_world):
    df, _ = predictive_world
    reg = st.predict_regression(df["oil"], df["eq"])
    assert reg["slope"] < 0.0       # the Driesprong sign
    assert reg["tstat"] < -2.0      # and significant


def test_null_world_no_relationship(null_world):
    df, _ = null_world
    reg = st.predict_regression(df["oil"], df["eq"])
    assert abs(reg["tstat"]) < 2.0


def test_timing_beats_buyhold_when_signal_real(predictive_world):
    df, _ = predictive_world
    timing = st.summary(st.oil_timing(df["oil"], df["eq"]))
    bh = st.summary(st.buy_hold(df["eq"]))
    assert timing["sharpe"] > bh["sharpe"]   # real signal ⇒ timing adds value


def test_timing_in_cash_unless_oil_fell(predictive_world):
    df, _ = predictive_world
    t = st.oil_timing(df["oil"], df["eq"])
    rose = (df["oil"].shift(1) >= 0).reindex(t.index)
    assert (t[rose] == 0.0).all()            # cash after an up-oil month
