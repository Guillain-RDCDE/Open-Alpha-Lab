"""The synthetic panel is deterministic; the same-month hedge works only when a same-month seasonal
exists; the other-month control does not; the null has no edge; costs are counted one-way (3.2x NAV
per month) and break-even is positive when gross is; the survivor panel is guarded behind an explicit
opt-in. All offline on the seeded synthetic world."""

import numpy as np
import pytest

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


def test_costs_counted_one_way(seasonal_world):
    """Turnover is one-way: ~80% of a two-sided book replaced = 3.2x NAV traded, so the monthly
    drag at cost c is 3.2*c and break-even = mean/3.2 — twice as strict as a round-trip count."""
    df, _ = seasonal_world
    h = st.seasonal_hedge(df, same_month=True)
    c = 10.0
    drag = h.mean() - st.net_of_cost(h, cost_bps=c).mean()
    assert np.isclose(drag, 3.2 * c / 1e4)
    assert np.isclose(st.breakeven_cost_bps(h), h.mean() * 1e4 / 3.2)


def test_borrow_drags_the_short_leg(seasonal_world):
    """A 50 bp/yr borrow on a 1x-NAV short leg costs 50/12 bp per month of hedge return."""
    df, _ = seasonal_world
    h = st.seasonal_hedge(df, same_month=True)
    net = st.net_of_borrow(h, borrow_bps_per_year=50.0)
    assert np.isclose(h.mean() - net.mean(), 50.0 / 1e4 / 12)


def test_fetch_panel_is_guarded(tmp_path):
    """The survivor panel refuses to load without the explicit opt-in (cache or no cache)."""
    from quantlab.hf_data import SurvivorshipBiasError

    with pytest.raises(SurvivorshipBiasError):
        data.fetch_panel(cache_dir=str(tmp_path))
    # opted in, with no cache and no fetch, it degrades to an empty frame (offline-safe)
    assert data.fetch_panel(cache_dir=str(tmp_path), allow_survivorship_bias=True).empty


def test_predictor_uses_only_past():
    """seasonal_hedge starts after warmup and never reads the current month — a smoke check on shape."""
    df, _ = data.synthetic_panel(seasonality=0.02, n_years=12, seed=1)
    h = st.seasonal_hedge(df, same_month=True, warmup=60)
    assert h.index.min() >= df.index[60]   # nothing before warmup
