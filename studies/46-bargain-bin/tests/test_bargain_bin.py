"""The synthetic worlds are deterministic; HML is positive only with a premium; the null shows nothing;
the regime world's premium really does go positive → negative → recover; and HML = value − growth.
All offline on the seeded synthetic worlds."""

import numpy as np

from bargain_bin import data, strategy as st


def test_world_deterministic(premium_world):
    df, truth = premium_world
    df2, _ = data.synthetic_hml(premium=0.06, regimes=False, idio_vol=0.04, seed=46)
    assert np.allclose(df.to_numpy(), df2.to_numpy())
    assert truth.has_premium


def test_premium_world_hml_positive(premium_world):
    df, _ = premium_world
    s = st.hml_stats(st.hml(df, "value", "growth"))
    assert s["mean_ann"] > 0.0
    assert s["tstat"] > 1.5


def test_null_world_hml_zero(null_world):
    df, _ = null_world
    s = st.hml_stats(st.hml(df, "value", "growth"))
    assert abs(s["tstat"]) < 2.0


def test_regime_world_has_a_lost_decade(regime_world):
    df, _ = regime_world
    split = st.regime_split(st.hml(df, "value", "growth"), breaks=(2009, 2017))
    pre = split.loc[split.index[0], "mean_ann"]
    mid = split.loc[split.index[1], "mean_ann"]
    assert pre > 0 > mid      # worked, then a negative lost-decade middle regime


def test_hml_is_value_minus_growth(premium_world):
    df, _ = premium_world
    s = st.hml(df, "value", "growth")
    assert np.allclose(s, (df["value"] - df["growth"]).loc[s.index])
