"""The synthetic panel is deterministic; the long-low/short-high hedge pays only when fast growers
truly underperform; in the null it earns ~nothing; the quantile split and alignment are correct.
All offline on the seeded synthetic world."""

import numpy as np
import pandas as pd

from growth_spurt import data, strategy as st


def test_world_deterministic(penalty_world):
    ag, fwd, truth = penalty_world
    ag2, fwd2, _ = data.synthetic_panel(growth_penalty=0.06, seed=44)
    assert np.allclose(ag.to_numpy(), ag2.to_numpy())
    assert np.allclose(fwd.to_numpy(), fwd2.to_numpy())
    assert truth.has_penalty


def test_penalty_world_hedge_pays(penalty_world):
    ag, fwd, _ = penalty_world
    h = st.quantile_hedge(ag, fwd, q=0.2)
    s = st.summary(h["hedge"])
    assert s["mean"] > 0.0       # low-growth beats high-growth
    assert s["sharpe"] > 0.3     # and reliably so when the effect is real


def test_null_world_no_hedge(null_world):
    ag, fwd, _ = null_world
    h = st.quantile_hedge(ag, fwd, q=0.2)
    assert abs(st.summary(h["hedge"])["mean"]) < 0.03   # nothing to harvest in the null


def test_hedge_is_low_minus_high(penalty_world):
    ag, fwd, _ = penalty_world
    h = st.quantile_hedge(ag, fwd, q=0.2)
    assert np.allclose(h["hedge"], h["low_growth"] - h["high_growth"])


def test_quantile_split_picks_extremes():
    """With a clean monotone growth ranking, the bottom-q and top-q names are the right ones."""
    years = [2010]
    n = 30
    firms = [f"F{i:02d}" for i in range(n)]
    ag = pd.DataFrame([np.linspace(0.0, 0.9, n)], index=pd.Index(years, name="year"), columns=firms)
    # forward return decreasing in growth → low-growth firms have the highest returns
    fwd = pd.DataFrame([np.linspace(0.9, 0.0, n)], index=pd.Index(years, name="year"), columns=firms)
    h = st.quantile_hedge(ag, fwd, q=0.2)
    # bottom-20% growth names earn the most, top-20% the least → hedge strongly positive
    assert h.loc[2010, "hedge"] > 0.5
