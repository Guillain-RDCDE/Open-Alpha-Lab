"""The synthetic worlds are deterministic; SMB is positive only when a premium exists; the null shows
nothing; the decay world's premium really does fall from positive to negative; and SMB = small − large.
All offline on the seeded synthetic worlds."""

import numpy as np

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
    split = st.decay_split(st.smb(df, "small", "large"), cut_year=2007)
    pre = split.loc[split.index[0], "mean_ann"]
    post = split.loc[split.index[1], "mean_ann"]
    assert pre > 0 > post           # worked, then reversed — the Banz signature


def test_smb_is_small_minus_large(premium_world):
    df, _ = premium_world
    s = st.smb(df, "small", "large")
    assert np.allclose(s, (df["small"] - df["large"]).loc[s.index])
