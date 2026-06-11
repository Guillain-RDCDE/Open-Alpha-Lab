"""The synthetic panel is deterministic; the same-month hedge works only when a same-month seasonal
exists; the other-month control does not; the null has no edge; costs reduce the hedge and break-even
is positive when gross is. All offline on the seeded synthetic world."""

import numpy as np

from groundhog import data, strategy as st


def test_world_deterministic(seasonal_world):
    df, truth = seasonal_world
    df2, _ = data.synthetic_panel(seasonality=0.02, seed=48)
    assert np.allclose(df.to_numpy(), df2.to_numpy())
    assert truth.has_seasonality


def test_same_month_works_control_does_not(seasonal_world):
    """The decisive test: the same-month predictor pays; the other-month control doesn't."""
    df, _ = seasonal_world
    same = st.stats(st.seasonal_hedge(df, same_month=True))
    ctrl = st.stats(st.seasonal_hedge(df, same_month=False))
    assert same["sharpe"] > 0.5            # the seasonal is real and strong on the control
    assert same["sharpe"] > ctrl["sharpe"] + 0.4   # and specific to the same month


def test_null_world_no_edge(null_world):
    df, _ = null_world
    assert abs(st.stats(st.seasonal_hedge(df, same_month=True))["sharpe"]) < 0.5


def test_costs_reduce_and_breakeven_positive(seasonal_world):
    df, _ = seasonal_world
    h = st.seasonal_hedge(df, same_month=True)
    gross = h.mean()
    net = st.net_of_cost(h, cost_bps=20.0).mean()
    assert net < gross
    assert st.breakeven_cost_bps(h) > 0    # a positive gross edge has a positive break-even


def test_predictor_uses_only_past():
    """seasonal_hedge starts after warmup and never reads the current month — a smoke check on shape."""
    df, _ = data.synthetic_panel(seasonality=0.02, n_years=12, seed=1)
    h = st.seasonal_hedge(df, same_month=True, warmup=60)
    assert h.index.min() >= df.index[60]   # nothing before warmup
