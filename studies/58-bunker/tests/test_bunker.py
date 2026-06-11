"""The synthetic world is deterministic; min-vol always cuts volatility (beta<1); the spread is
positive only with a real low-vol alpha; the null shows none. All offline on the seeded world."""
import numpy as np
from bunker import data, strategy as st


def test_world_deterministic(alpha_world):
    df, truth = alpha_world
    df2, _ = data.synthetic_world(lowvol_alpha=0.05, seed=58)
    assert np.allclose(df.to_numpy(), df2.to_numpy())
    assert truth.has_alpha


def test_minvol_cuts_volatility(alpha_world):
    df, _ = alpha_world
    assert st.vol_reduction(df, "USMV", "SPY") < 1.0   # lower vol by construction (beta<1)


def test_alpha_world_spread_positive(alpha_world):
    df, _ = alpha_world
    assert st.spread_stats(st.spread(df, "USMV", "SPY"))["mean_ann"] > 0.0


def test_null_world_no_alpha(null_world):
    df, _ = null_world
    assert abs(st.spread_stats(st.spread(df, "USMV", "SPY"))["tstat"]) < 2.0


def test_spread_is_usmv_minus_spy(alpha_world):
    df, _ = alpha_world
    s = st.spread(df, "USMV", "SPY")
    assert np.allclose(s, (df["USMV"] - df["SPY"]).loc[s.index])
