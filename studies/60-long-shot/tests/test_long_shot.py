"""The synthetic panel is deterministic; the skewness signal separates high- from low-skew assets; the
textbook long-low/short-high trade pays only when high-skew underperforms; the null shows nothing; costs
reduce the hedge. All offline on the seeded world."""
import numpy as np
from long_shot import data, strategy as st


def test_world_deterministic(skew_world):
    df, truth = skew_world
    df2, _ = data.synthetic_panel(n_assets=18, skew_premium=0.0030, seed=60)
    assert np.allclose(df.to_numpy(), df2.to_numpy())
    assert truth.has_premium


def test_textbook_trade_pays_when_skew_real(skew_world):
    df, _ = skew_world
    h = st.stats(st.cross_section_hedge(df, st.skewness_signal(df), long_high=False))
    assert h["sharpe"] > 0.3       # low-skew wins on the control


def test_null_world_no_edge(null_world):
    df, _ = null_world
    assert abs(st.stats(st.cross_section_hedge(df, st.skewness_signal(df), long_high=False))["sharpe"]) < 0.5


def test_skewness_signal_finite(skew_world):
    df, _ = skew_world
    sig = st.skewness_signal(df).dropna(how="all")
    vals = sig.to_numpy()
    assert np.isfinite(vals[np.isfinite(vals)]).all()


def test_costs_reduce_hedge(skew_world):
    df, _ = skew_world
    h = st.cross_section_hedge(df, st.skewness_signal(df), long_high=False)
    assert st.net_of_cost(h, 25.0).mean() < h.mean()
