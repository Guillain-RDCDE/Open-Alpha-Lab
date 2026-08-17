"""Window arithmetic, cost/borrow accounting, inference primitives, and the study's spine.

Everything is offline and synthetic. The spine: on a tape with a *planted* post-first-cut
duration rally the event study recovers the planted size and clears |t| >= 3 once events
are pooled across worlds; on the null it stays quiet. The tests also pin the study's two
honesty invariants — exactly one execution lag, and costs/borrow that can only subtract.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from first_cut import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Window arithmetic — the one execution lag
# --------------------------------------------------------------------------- #
def test_entry_is_the_session_after_the_announcement():
    idx = pd.bdate_range("2020-01-01", periods=400)
    ev = idx[100]
    entry, exit_ = st.window_bounds(idx, ev, 6)
    assert entry == idx[101]           # exactly one session of lag, never zero
    assert entry > ev


def test_exit_is_the_last_session_on_or_before_the_horizon():
    idx = pd.bdate_range("2020-01-01", periods=500)
    ev = idx[100]
    entry, exit_ = st.window_bounds(idx, ev, 3)
    target = ev + pd.DateOffset(months=3)
    assert exit_ <= target
    assert idx[idx.get_loc(exit_) + 1] > target


def test_incomplete_window_is_dropped_not_truncated():
    idx = pd.bdate_range("2020-01-01", periods=200)
    ev = idx[-5]
    assert st.window_bounds(idx, ev, 12) is None


def test_event_before_the_tape_is_dropped(planted):
    prices, events, _ = planted
    early = pd.DatetimeIndex([prices.index[0] - pd.DateOffset(years=3)])
    tbl = st.event_table(prices["duration"], prices["cash"], early, 6)
    assert len(tbl) == 0


def test_in_window_mask_matches_the_event_table(planted):
    prices, events, _ = planted
    idx = prices.index
    mask = st.in_window_mask(idx, events, 6)
    tbl = st.event_table(prices["duration"], prices["cash"], events, 6)
    # Every table row's exit day must be flagged, and the day before every entry must not.
    for _, r in tbl.iterrows():
        assert mask.loc[r["exit"]] == 1.0
        assert mask.loc[r["entry"]] == 0.0


def test_no_lookahead_perturbing_the_future_leaves_past_events_alone(planted):
    prices, events, _ = planted
    tbl = st.event_table(prices["duration"], prices["cash"], events[:2], 6)
    tail = prices.copy()
    cut = tail.index[-500]
    tail.loc[tail.index >= cut, "duration"] *= 3.0
    tbl2 = st.event_table(tail["duration"], tail["cash"], events[:2], 6)
    assert np.allclose(tbl["excess_pct"].to_numpy(), tbl2["excess_pct"].to_numpy())


# --------------------------------------------------------------------------- #
# Cost & borrow accounting
# --------------------------------------------------------------------------- #
def test_costs_only_subtract_and_are_two_one_way_legs(planted):
    prices, events, _ = planted
    gross = st.event_table(prices["duration"], prices["cash"], events, 6, cost_bps=0.0)
    net = st.event_table(prices["duration"], prices["cash"], events, 6, cost_bps=25.0)
    assert np.allclose(gross["excess_pct"], net["excess_pct"])          # gross unchanged
    delta = (gross["excess_net_pct"] - net["excess_net_pct"]).to_numpy()
    assert np.allclose(delta, 2 * 25.0 * 1e-4 * 100.0)                  # exactly 2 legs


def test_cost_sweep_is_monotone_decreasing(planted):
    prices, events, _ = planted
    sw = st.cost_sweep(prices["duration"], prices["cash"], events, horizon_months=6)
    vals = sw["mean_excess_net_pct"].to_numpy()
    assert np.all(np.diff(vals) <= 1e-12)


def test_borrow_only_subtracts_from_the_short_leg(planted):
    prices, events, _ = planted
    sw = st.borrow_sweep(prices["duration"], prices["front"], events, horizon_months=6)
    vals = sw["mean_ls_net_pct"].to_numpy()
    assert np.all(np.diff(vals) < 0.0)
    assert sw["n_events"].nunique() == 1


def test_long_short_gross_is_borrow_free(planted):
    prices, events, _ = planted
    a = st.long_short_table(prices["duration"], prices["front"], events, 6, borrow_bps_ann=0.0)
    b = st.long_short_table(prices["duration"], prices["front"], events, 6, borrow_bps_ann=300.0)
    assert np.allclose(a["ls_gross_pct"], b["ls_gross_pct"])
    assert (a["ls_net_pct"] > b["ls_net_pct"]).all()


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_recovers_a_positive_mean():
    rng = np.random.default_rng(1)
    assert st.newey_west_t(0.001 + rng.normal(0, 0.01, 5000)) > 3
    assert abs(st.newey_west_t(rng.normal(0, 0.01, 5000))) < 3


def test_one_sample_t_is_nan_on_a_single_observation():
    assert np.isnan(st.one_sample_t(np.array([1.0])))


def test_block_bootstrap_ci_brackets_the_point(planted):
    prices, events, _ = planted
    daily = st.conditional_daily(prices["duration"], prices["cash"], events, 6)
    ci = st.block_bootstrap_mean_ci(daily["e_dur"], n_boot=400, seed=924)
    assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]


def test_block_bootstrap_scale_is_linear(planted):
    prices, events, _ = planted
    daily = st.conditional_daily(prices["duration"], prices["cash"], events, 6)
    a = st.block_bootstrap_mean_ci(daily["e_dur"], n_boot=200, seed=924, scale=1.0)
    b = st.block_bootstrap_mean_ci(daily["e_dur"], n_boot=200, seed=924, scale=252.0)
    assert np.isclose(b["mean"], a["mean"] * 252.0)


def test_placebo_pvalue_endpoints():
    means = np.arange(100, dtype=float)
    assert st.placebo_pvalue(1000.0, means) == 0.0
    assert st.placebo_pvalue(-1.0, means) == 1.0
    assert np.isnan(st.placebo_pvalue(float("nan"), means))


# --------------------------------------------------------------------------- #
# The study's spine — the harness finds a planted effect and stays quiet on the null
# --------------------------------------------------------------------------- #
def _pooled(worlds, horizon=6):
    vals = []
    for prices, events, _ in worlds:
        tbl = st.event_table(prices["duration"], prices["cash"], events, horizon, 5.0)
        vals.extend(tbl["excess_net_pct"].tolist())
    return np.array(vals)


def test_planted_effect_is_recovered_in_size(planted):
    prices, events, truth = planted
    d = st.synthetic_detect(prices, events, horizon_months=6)
    planted_pct = truth["planted_6m_excess"] * 100.0
    assert abs(d["mean_excess_pct"] - planted_pct) < 0.5 * planted_pct
    assert d["in_ann_pct"] > d["out_ann_pct"]


def test_pooled_planted_panel_clears_the_bar(planted_panel):
    vals = _pooled(planted_panel)
    assert len(vals) >= 50
    assert vals.mean() > 5.0
    assert st.one_sample_t(vals) > 3.0


def test_pooled_null_panel_is_quiet(null_panel):
    vals = _pooled(null_panel)
    assert abs(vals.mean()) < 3.0
    assert abs(st.one_sample_t(vals)) < 2.5


def test_null_five_event_t_fires_at_about_the_nominal_rate(null_panel):
    """A five-event t-test on the null should exceed |t| = 2 only rarely."""
    ts = []
    for prices, events, _ in null_panel:
        tbl = st.event_table(prices["duration"], prices["cash"], events, 6, 5.0)
        ts.append(st.one_sample_t(tbl["excess_net_pct"].to_numpy()))
    ts = np.abs(np.array(ts))
    assert (ts > 2.0).sum() <= 2          # 12 worlds, nominal ~5% => 0-2 is normal


def test_five_event_test_is_underpowered_even_when_the_effect_is_real(planted_panel):
    """The point the study exists to make: N=5 misses a genuine +9% effect most of the time."""
    ts = []
    for prices, events, _ in planted_panel:
        tbl = st.event_table(prices["duration"], prices["cash"], events, 6, 5.0)
        ts.append(st.one_sample_t(tbl["excess_net_pct"].to_numpy()))
    ts = np.array(ts)
    assert (ts > 2.0).mean() < 0.8        # power well short of certainty at N=5
    assert ts.mean() > 1.0                # but the effect is genuinely there


def test_null_daily_hac_t_is_small(null_world):
    prices, events, _ = null_world
    d = st.synthetic_detect(prices, events, horizon_months=6)
    assert abs(d["t_event_leg_hac"]) < 2.5


def test_horizon_sweep_shape(planted):
    prices, events, _ = planted
    sw = st.horizon_sweep(prices["duration"], prices["cash"], events)
    assert list(sw.index) == list(st.HORIZONS)
    assert sw["n_events"].min() >= 3
    assert np.isfinite(sw["mean_excess_pct"]).all()


def test_era_split_partitions_the_events(planted):
    prices, events, _ = planted
    mid = str(prices.index[len(prices) // 2].date())
    eras = st.era_split(prices["duration"], prices["cash"], events,
                        split=mid, horizon_months=6)
    tbl = st.event_table(prices["duration"], prices["cash"], events, 6)
    assert eras["early"]["n_events"] + eras["late"]["n_events"] == len(tbl)


def test_placebo_distribution_is_centred_on_the_all_date_mean(null_world):
    prices, events, _ = null_world
    pl = st.placebo(prices["duration"], prices["cash"], n_events=5,
                    horizon_months=6, n_draws=300, seed=924)
    assert pl["n_eligible"] > 1000
    assert abs(pl["means"].mean() - pl["all_mean_pct"]) < 1.0


def test_conditional_daily_gate_and_costs(planted):
    prices, events, _ = planted
    d = st.conditional_daily(prices["duration"], prices["cash"], events, 6, cost_bps=0.0)
    assert set(np.unique(d["in_window"])) <= {0.0, 1.0}
    flat = d.loc[d["in_window"] == 0.0, "e_event"]
    assert np.allclose(flat.to_numpy(), 0.0)     # no cost => flat days earn exactly zero
    live = d.loc[d["in_window"] == 1.0]
    assert np.allclose(live["e_event"].to_numpy(), live["e_dur"].to_numpy())


def test_all_cuts_control_runs_on_synthetic(planted):
    """The control path must work on any calendar, including a denser one."""
    prices, events, _ = planted
    dense = pd.DatetimeIndex(sorted(set(list(events) + list(prices.index[::400]))))
    cmp = st.compare(prices["duration"], prices["cash"], dense, horizon_months=3)
    assert cmp["n_events"] > len(events)
    assert 0.0 <= cmp["in_frac"] <= 1.0
