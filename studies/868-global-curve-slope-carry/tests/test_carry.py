"""Offline, fixed-seed tests for the global-curve-slope-carry machinery.

The synthetic panel is deterministic; the realized-yield carry proxy has the right
definition and no look-ahead; the cross-sectional carry book recovers a planted carry
(positive mean, Newey-West t lights up) and stays silent on the null; the duration scaling
re-ranks the cross-section; costs reduce the net; the inference primitives behave. Every
test is offline synthetic-only — the one test that touches the real cache is skipped when
the cache is absent.
"""
import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from curve_slope_carry import data, strategy as st  # noqa: E402

CACHE = data.cache_path()


# --------------------------------------------------------------------------- #
# Synthetic determinism + signal definition
# --------------------------------------------------------------------------- #
def test_world_deterministic(edge_world):
    p2 = data.synthetic_panel(edge=0.010, seed=868)
    assert np.allclose(edge_world.to_numpy(), p2.to_numpy())


def test_synthetic_index_no_overflow(edge_world):
    # Month-end index stays well inside the pandas ns-Timestamp horizon (< 2262).
    assert edge_world.index.max() < pd.Timestamp("2100-01-01")
    assert edge_world.shape == (360, 6)


def test_realized_yield_definition():
    # A clean geometric series: level_t = (1+g)**t => monthly return == g => realized
    # yield == g*12 once the trailing window is full.
    g = 0.004
    n = 60
    lvl = pd.DataFrame({"A": (1.0 + g) ** np.arange(n)},
                       index=pd.date_range("2000-01-31", periods=n, freq="ME"))
    ry = st.realized_yield(lvl, window=36)
    assert np.isnan(ry["A"].iloc[3])                 # not enough history early
    assert abs(ry["A"].iloc[50] - g * 12.0) < 1e-9


def test_carry_signal_duration_reranks(edge_world):
    # Dividing by per-column durations is not a no-op: it changes the cross-sectional values.
    durs = {c: 1.0 + 3.0 * i for i, c in enumerate(edge_world.columns)}
    raw = st.carry_signal(edge_world, durations=None)
    ytd = st.carry_signal(edge_world, durations=durs)
    a, b = raw.to_numpy(), ytd.to_numpy()
    m = ~np.isnan(a) & ~np.isnan(b)
    assert not np.allclose(a[m], b[m])


def test_book_is_dollar_neutral(edge_world):
    bt = st.carry_book(edge_world)
    W = bt.attrs["W"].to_numpy()
    # each row's signed weights sum to ~0 (long +1 gross, short -1 gross)
    assert np.allclose(np.nansum(W, axis=1), 0.0, atol=1e-9)
    assert (bt["n_long"] > 0).all() and (bt["n_short"] > 0).all()


def test_positions_are_point_in_time(edge_world):
    # The signal feeding month t is shifted one month (known at t-1); a shock to the final
    # month's price cannot change any earlier book return.
    bt1 = st.carry_book(edge_world)
    shocked = edge_world.copy()
    shocked.iloc[-1] = shocked.iloc[-1] * 1.5
    bt2 = st.carry_book(shocked)
    common = bt1.index.intersection(bt2.index)[:-1]
    assert np.allclose(bt1.loc[common, "ret"].to_numpy(),
                       bt2.loc[common, "ret"].to_numpy())


# --------------------------------------------------------------------------- #
# Planted effect recovered / null silent
# --------------------------------------------------------------------------- #
def test_planted_carry_recovered(edge_world):
    s = st.carry_stats(st.carry_book(edge_world))
    assert s["mean_bps"] > 0
    assert s["t_nw"] > 3.0          # the carry detector lights up on a planted effect
    assert s["sharpe"] > 0.5
    assert s["long_bps"] > s["short_bps"]   # high-carry leg out-yields the low-carry leg


def test_null_world_no_signal(null_world):
    s = st.carry_stats(st.carry_book(null_world))
    assert abs(s["t_nw"]) < 2.5     # nothing to find when every market yields the same


def test_null_silent_over_many_seeds():
    res = st.synthetic_mean_t(data, edge=0.0, n_seeds=20, base_seed=868)
    assert abs(res["mean_t"]) < 1.0
    assert res["fire_frac"] <= 0.15   # essentially never fires on the null


def test_placebo_separates_planted_from_null(edge_world, null_world):
    pe = st.placebo_pvalue(edge_world, n_perm=400)
    pn = st.placebo_pvalue(null_world, n_perm=400)
    assert pe["p_value"] < 0.05        # planted world sits in the right tail
    assert pn["p_value"] > 0.10        # null world does not


# --------------------------------------------------------------------------- #
# Costs
# --------------------------------------------------------------------------- #
def test_costs_reduce_net(edge_world):
    gross = st.timer_stats(edge_world, cost_bps=0.0, borrow_bps_yr=0.0)["net_bps"]
    net = st.timer_stats(edge_world, cost_bps=10.0, borrow_bps_yr=75.0)["net_bps"]
    assert net < gross


def test_timer_gross_matches_book(edge_world):
    tm = st.timer_stats(edge_world, cost_bps=0.0, borrow_bps_yr=0.0)
    s = st.carry_stats(st.carry_book(edge_world))
    assert abs(tm["gross_bps"] - s["mean_bps"]) < 1e-6


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=6) - st.one_sample_t(x)) < 0.6


def test_welch_t_detects_mean_gap():
    rng = np.random.default_rng(1)
    a = rng.normal(0.02, 0.05, 500)
    b = rng.normal(0.00, 0.05, 500)
    assert st.welch_t(a, b) > 3.0


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(55, 100)
    assert lo < 0.55 < hi


# --------------------------------------------------------------------------- #
# Real cache (skipped offline)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not os.path.exists(CACHE), reason="real cache absent offline CI")
def test_real_panel_loads_and_is_monthly():
    panel = data.load_panel()
    assert panel.index.max() <= pd.Timestamp(data.AS_OF)
    assert set(["SHY", "IEF", "TLT", "BWX", "IGOV", "BNDX"]).issuperset(set(panel.columns))
    bt = st.carry_book(panel, durations=data.DURATIONS)
    assert len(bt) > 100
