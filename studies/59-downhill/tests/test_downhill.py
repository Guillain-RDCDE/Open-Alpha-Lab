"""The synthetic world is deterministic; long duration is more volatile than cash; the term premium is
positive only when present; the null shows none. All offline on the seeded world."""
import numpy as np
from downhill import data, strategy as st


def test_world_deterministic(premium_world):
    df, truth = premium_world
    df2, _ = data.synthetic_world(premium=0.05, long_vol=0.025, seed=59)
    assert np.allclose(df.to_numpy(), df2.to_numpy())
    assert truth.has_premium


def test_duration_orders_volatility(premium_world):
    df, _ = premium_world
    v = {c: st.leg_summary(df, c)["vol_ann"] for c in ("IEF", "SHY", "BIL")}
    assert v["IEF"] > v["SHY"] > v["BIL"]   # longer duration → more volatile


def test_term_premium_positive(premium_world):
    df, _ = premium_world
    assert st.excess_stats(st.term_premium(df))["mean_ann"] > 0.0


def test_null_world_no_premium(null_world):
    df, _ = null_world
    assert abs(st.excess_stats(st.term_premium(df))["mean_ann"]) < 0.02


def test_term_premium_is_ief_minus_bil(premium_world):
    df, _ = premium_world
    s = st.term_premium(df)
    assert np.allclose(s, (df["IEF"] - df["BIL"]).loc[s.index])
