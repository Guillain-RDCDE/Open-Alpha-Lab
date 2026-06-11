"""The synthetic world is deterministic; the winter premium shows up only when present; the null has
no seasonal; the sell-in-may rule is in cash all summer; and the month classification is correct.
All offline on the seeded synthetic world."""

import numpy as np
import pandas as pd

from summer_lull import data, strategy as st


def test_world_deterministic(seasonal_world):
    s, truth = seasonal_world
    s2, _ = data.synthetic_monthly(n_years=120, winter_premium_bp=100.0, seed=55)
    assert np.allclose(s.to_numpy(), s2.to_numpy())
    assert truth.has_seasonal


def test_winter_premium_recovered(seasonal_world):
    s, _ = seasonal_world
    d = st.seasonal_split(s)
    assert d["winter_bp"] > d["summer_bp"] + 20.0
    assert d["welch_t"] > 2.0


def test_null_has_no_seasonal(null_world):
    s, _ = null_world
    assert abs(st.seasonal_split(s)["welch_t"]) < 2.0


def test_sell_in_may_is_cash_all_summer(seasonal_world):
    s, _ = seasonal_world
    sim = st.sell_in_may(s)
    summer = ~st.is_winter(sim.index).to_numpy()
    assert (sim[summer] == 0.0).all()                        # cash May-Oct
    assert np.allclose(sim[~summer], s[st.is_winter(s.index).to_numpy()])  # invested Nov-Apr


def test_winter_classification():
    idx = pd.date_range("2020-01-31", periods=12, freq="ME")
    w = st.is_winter(idx)
    assert w.loc[idx[0]]      # January is winter
    assert w.loc[idx[10]]     # November is winter
    assert not w.loc[idx[5]]  # June is summer
