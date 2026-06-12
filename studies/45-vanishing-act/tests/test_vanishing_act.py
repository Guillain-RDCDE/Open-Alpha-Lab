"""The synthetic worlds are deterministic; SMB is positive only when a premium exists; the null shows
nothing; the decay world's premium really does fall from positive to negative (and the bootstrap flags
the difference as real); leg summaries are aligned by construction; and SMB = small − large. All
offline on the seeded synthetic worlds."""

import numpy as np
import pandas as pd

from vanishing_act import data, strategy as st


def test_world_deterministic(premium_world):
    df, truth = premium_world
    df2, _ = data.synthetic_smb(premium=0.05, decay=False, seed=45)
    assert np.allclose(df.to_numpy(), df2.to_numpy())
    assert truth.has_premium


def test_premium_world_smb_positive(premium_world):
    df, _ = premium_world
    s = st.smb_stats(st.smb(df, "small", "large"))
    assert s["mean_ann"] > 0.0
    assert s["tstat"] > 1.5         # a real, stable premium is detectable


def test_null_world_smb_zero(null_world):
    df, _ = null_world
    s = st.smb_stats(st.smb(df, "small", "large"))
    assert abs(s["tstat"]) < 2.0    # indistinguishable from zero


def test_decay_world_premium_reverses(decay_world):
    df, _ = decay_world
    split = st.window_split(st.smb(df, "small", "large"), cut_year=2007)
    pre = split.loc[split.index[0], "mean_ann"]
    post = split.loc[split.index[1], "mean_ann"]
    assert pre > 0 > post           # worked, then reversed — the engineered signature


def test_split_difference_separates_real_break_from_noise(null_world):
    """The block bootstrap must flag a strongly engineered ramp as a real pre/post difference and
    stay agnostic on the null — otherwise a 'turn' in the write-up is just a lucky window."""
    dfd, _ = data.synthetic_smb(premium=0.12, decay=True, seed=45)   # an unmissable break
    dfn, _ = null_world
    real = st.split_difference(st.smb(dfd, "small", "large"), cut_year=2007)
    noise = st.split_difference(st.smb(dfn, "small", "large"), cut_year=2007)
    assert real["diff_ann"] > 0 and real["p_value"] < 0.05
    assert noise["p_value"] > 0.05


def test_leg_summary_aligns_periods():
    """One leg with extra early history must not leak it into the comparison."""
    idx = pd.date_range("2000-01-31", periods=120, freq="ME")
    rng = np.random.default_rng(0)
    large = pd.Series(0.05 + 0.02 * rng.standard_normal(120), index=idx)  # strong early decade
    small = pd.Series(0.005 + 0.02 * rng.standard_normal(120), index=idx)
    small.iloc[:60] = np.nan                                              # small leg starts 5y later
    df = pd.DataFrame({"small": small, "large": large})
    legs = st.leg_summary(df, "small", "large")
    assert legs["n"] == 60
    assert legs["start"] == idx[60]
    # the large leg's stats must be computed on the common 60 months only
    expect = st.summary(large.iloc[60:])
    assert np.isclose(legs["large"]["sharpe"], expect["sharpe"])


def test_smb_is_small_minus_large(premium_world):
    df, _ = premium_world
    s = st.smb(df, "small", "large")
    assert np.allclose(s, (df["small"] - df["large"]).loc[s.index])
