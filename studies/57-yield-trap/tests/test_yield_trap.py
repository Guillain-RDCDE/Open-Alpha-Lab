"""The synthetic world is deterministic; the spread is positive only when a premium exists; the null
shows nothing; the spread equals a minus b. All offline on the seeded synthetic world."""
import numpy as np
from yield_trap import data, strategy as st


def test_world_deterministic(premium_world):
    df, truth = premium_world
    df2, _ = data.synthetic_world(premium=0.04, seed=57)
    assert np.allclose(df.to_numpy(), df2.to_numpy())
    assert truth.has_premium


def test_premium_world_spread_positive(premium_world):
    df, _ = premium_world
    s = st.spread_stats(st.spread(df, "VYM", "SPY"))
    assert s["mean_ann"] > 0.0
    assert s["tstat"] > 1.5


def test_null_world_no_spread(null_world):
    df, _ = null_world
    assert abs(st.spread_stats(st.spread(df, "VYM", "SPY"))["tstat"]) < 2.0


def test_spread_is_a_minus_b(premium_world):
    df, _ = premium_world
    s = st.spread(df, "VYM", "SPY")
    assert np.allclose(s, (df["VYM"] - df["SPY"]).loc[s.index])


def test_leg_summary_runs(premium_world):
    df, _ = premium_world
    assert "sharpe" in st.leg_summary(df, "SPY")
