"""The synthetic panel is deterministic; nearness is bounded in (0,1]; in a trending world the
nearness hedge pays and is highly correlated with the 12-2 momentum hedge (same factor); in the null
it earns nothing; the momentum control skips the most recent month (the 12-2 convention); costs are
counted one-way (3.2x NAV per month); and the survivor panel is guarded behind an explicit opt-in.
All offline on the seeded synthetic world."""

import numpy as np
import pytest

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


def test_costs_counted_one_way(trend_world):
    """Turnover is one-way: ~80% of a two-sided book replaced = 3.2x NAV traded per month."""
    df, _ = trend_world
    h = st.cross_section_hedge(df, st.nearness(df))
    c = 10.0
    assert np.isclose(h.mean() - st.net_of_cost(h, cost_bps=c).mean(), 3.2 * c / 1e4)


def test_momentum_is_12_2(trend_world):
    """The control follows the Jegadeesh-Titman 12-2 convention: the most recent month is skipped,
    so this month's return never moves this month's momentum score."""
    df, _ = trend_world
    mom = st.momentum(df)                       # default skip=1
    bumped = df.copy()
    bumped.iloc[-1] = bumped.iloc[-1] + 0.10    # shock the latest month only
    assert np.allclose(
        mom.iloc[-1].dropna(), st.momentum(bumped).iloc[-1].dropna()
    )  # 12-2 ignores it...
    naive = st.momentum(df, skip=0)
    assert not np.allclose(
        naive.iloc[-1].dropna(), st.momentum(bumped, skip=0).iloc[-1].dropna()
    )  # ...while the naive trailing-12 does not


def test_fetch_panel_is_guarded(tmp_path):
    """The survivor panel refuses to load without the explicit opt-in (cache or no cache)."""
    from quantlab.hf_data import SurvivorshipBiasError

    with pytest.raises(SurvivorshipBiasError):
        data.fetch_panel(cache_dir=str(tmp_path))
    # opted in, with no cache and no fetch, it degrades to an empty frame (offline-safe)
    assert data.fetch_panel(cache_dir=str(tmp_path), allow_survivorship_bias=True).empty
