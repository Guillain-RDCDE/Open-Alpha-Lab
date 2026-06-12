"""The synthetic world is deterministic; the covered-call fund caps upside (upside capture < 1) and
trails the underlying when capped; the null matches the index; the spread equals fund minus underlying."""
import numpy as np
from premium_seller import data, strategy as st


def test_world_deterministic(capped_world):
    df, truth = capped_world
    df2, _ = data.synthetic_world(cap=0.5, premium=0.005, seed=62)
    assert np.allclose(df.to_numpy(), df2.to_numpy())
    assert truth.is_capped


def test_capped_fund_caps_upside(capped_world):
    df, _ = capped_world
    cap = st.capture(df, "QYLD", "QQQ")
    assert cap["upside_capture"] < 0.9                     # keeps less than the full up-move


def test_capped_fund_trails_underlying(capped_world):
    df, _ = capped_world
    assert st.spread_stats(st.spread(df, "QYLD", "QQQ"))["mean_ann"] < 0.0


def test_null_world_matches_index(null_world):
    df, _ = null_world
    assert abs(st.spread_stats(st.spread(df, "QYLD", "QQQ"))["mean_ann"]) < 0.02


def test_spread_is_fund_minus_underlying(capped_world):
    df, _ = capped_world
    s = st.spread(df, "QYLD", "QQQ")
    assert np.allclose(s, (df["QYLD"] - df["QQQ"]).loc[s.index])
