"""The synthetic panel is deterministic; nearness is bounded in (0,1]; in a trending world the
nearness hedge pays and is highly correlated with the momentum hedge (same factor); in the null it
earns nothing; and costs reduce the hedge. All offline on the seeded synthetic world."""

import numpy as np

from high_water import data, strategy as st


def test_world_deterministic(trend_world):
    df, truth = trend_world
    df2, _ = data.synthetic_panel(trend=0.12, seed=50)
    assert np.allclose(df.to_numpy(), df2.to_numpy())
    assert truth.has_trend


def test_nearness_is_bounded(trend_world):
    df, _ = trend_world
    near = st.nearness(df).to_numpy()
    near = near[np.isfinite(near)]
    assert (near > 0).all() and (near <= 1.0 + 1e-9).all()   # price never exceeds its own running max


def test_trend_world_nearness_pays_and_tracks_momentum(trend_world):
    df, _ = trend_world
    near_h = st.stats(st.cross_section_hedge(df, st.nearness(df)))
    assert near_h["sharpe"] > 0.3              # near-high stocks keep rising in a trending world
    assert st.signal_overlap(df) > 0.5         # and it's basically the momentum factor


def test_null_world_no_edge(null_world):
    df, _ = null_world
    assert abs(st.stats(st.cross_section_hedge(df, st.nearness(df)))["sharpe"]) < 0.5


def test_costs_reduce_hedge(trend_world):
    df, _ = trend_world
    h = st.cross_section_hedge(df, st.nearness(df))
    assert st.net_of_cost(h, 20.0).mean() < h.mean()
