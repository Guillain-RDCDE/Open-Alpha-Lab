"""Fully-offline, deterministic tests for Study 844 — Madden-Cover-Curse.

No network: everything here runs on the seeded synthetic world, a synthetic price panel
built on a business-day index, and pure functions. The single real-tape test is skipped
when the git-ignored `_cache/` is absent (i.e. on CI).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from madden_curse import data as dt, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Calendar shape
# --------------------------------------------------------------------------- #
def test_calendar_shape():
    # 20 launches: 10 Madden (EA) + 10 NBA 2K (TTWO), 2015->2024.
    assert len(dt.EVENTS) == 20
    pubs = [p for _, p, _, _ in dt.EVENTS]
    assert pubs.count("EA") == 10 and pubs.count("TTWO") == 10
    dates = [pd.Timestamp(d) for d, _, _, _ in dt.EVENTS]
    # each parseable and unique; span the intended years
    assert len(set(dates)) == len(dates)
    years = {d.year for d in dates}
    assert min(years) == 2015 and max(years) == 2024
    # every event carries a non-empty cover athlete label
    assert all(athlete.strip() for _, _, _, athlete in dt.EVENTS)


# --------------------------------------------------------------------------- #
# Inference primitives — hand calcs
# --------------------------------------------------------------------------- #
def test_one_sample_t_matches_hand_calc():
    x = np.array([0.01, -0.02, 0.03, 0.00, 0.015])
    r = st.one_sample_t(x)
    assert r["n"] == 5
    assert abs(r["mean"] - x.mean()) < 1e-12
    se = x.std(ddof=1) / np.sqrt(5)
    assert abs(r["t"] - x.mean() / se) < 1e-9


def test_newey_west_matches_one_sample_on_iid():
    # at 0 lags the HAC t collapses to the plain one-sample t up to the ddof (1/n vs
    # 1/(n-1)) variance convention; on 500 iid points that factor is ~1.001.
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 1.0, 500)
    nw = st.newey_west_t(x, lags=0)
    ss = st.one_sample_t(x)["t"]
    assert abs(nw / ss - np.sqrt(len(x) / (len(x) - 1))) < 1e-9


def test_welch_t_zero_when_means_equal():
    a = np.array([0.01, -0.01, 0.02, -0.02])
    b = np.array([0.03, -0.03, 0.04, -0.04])  # same mean (0), wider spread
    assert abs(st.welch_t(a, b)) < 1e-9


def test_wilson_interval_brackets_point_estimate():
    lo, hi = st.wilson_interval(9, 20)
    assert lo < 9 / 20 < hi
    assert 0.0 <= lo <= hi <= 1.0


def test_hit_rate_counts_positive_events():
    x = np.array([0.02, -0.01, 0.03, -0.05, 0.0])  # 2 strictly-positive of 5
    hr = st.hit_rate(x)
    assert hr["k"] == 2 and hr["n"] == 5
    assert abs(hr["rate"] - 0.4) < 1e-12
    assert hr["lo"] < 0.4 < hr["hi"]


# --------------------------------------------------------------------------- #
# Synthetic world — determinism, null, planted recovery
# --------------------------------------------------------------------------- #
def test_synthetic_world_deterministic(null_world):
    a, b, keys = null_world
    a2, b2, keys2 = dt.synthetic_world(drift=0.0, seed=844)
    assert keys == keys2
    assert np.allclose(a.values, a2.values) and np.allclose(b.values, b2.values)
    assert len(keys) == 20  # 20 synthetic launches, mirroring the real calendar size


def test_synthetic_null_is_quiet():
    # drift = 0 -> the launch-drift detector should not fire; mean |t| small across seeds
    ts = np.array([st.synthetic_detect(drift=0.0, seed=844 + s, k=10)["t"] for s in range(20)])
    assert abs(ts.mean()) < 0.75
    # honest small-sample FPR bound at n=20 events with a 10-day AR
    assert (np.abs(ts) >= 2).sum() <= 4


def test_synthetic_planted_drift_lifts_t_monotonically():
    t0 = st.synthetic_detect(drift=0.0, seed=844, k=10)["t"]
    t1 = st.synthetic_detect(drift=0.02, seed=844, k=10)["t"]
    t2 = st.synthetic_detect(drift=0.04, seed=844, k=10)["t"]
    assert t0 < t1 < t2
    assert t2 > 1.8  # a 4% planted launch drift is clearly detectable


def test_synthetic_planted_drift_recovered_unit_slope():
    # each +1% of planted drift should raise the measured post-window mean AR by ~+1%.
    m0 = st.synthetic_detect(drift=0.0, seed=844, k=10)["mean"]
    m4 = st.synthetic_detect(drift=0.04, seed=844, k=10)["mean"]
    assert abs((m4 - m0) - 0.04) < 5e-3


# --------------------------------------------------------------------------- #
# Event-table machinery on a synthetic PRICE panel (no network, no real cache)
# --------------------------------------------------------------------------- #
def _synthetic_price_panel(post_drift: float = 0.0, seed: int = 7) -> dict[str, pd.Series]:
    """Build EA/TTWO/SPY adjusted-close Series on a business-day index spanning the real
    event calendar, optionally injecting a positive publisher AR in the 5 sessions after
    each launch anchor (to prove build_event_table's post-window AR picks it up)."""
    idx = pd.bdate_range("2014-06-02", periods=3200)  # ~2014-06 -> ~2026-10, safe horizon
    rng = np.random.default_rng(seed)
    spy_r = rng.normal(0.0003, 0.008, len(idx))
    prices = {"SPY": pd.Series(100.0 * np.exp(np.cumsum(spy_r)), index=idx)}
    for publisher in dt.PUBLISHERS:
        r = 0.6 * spy_r + rng.normal(0.0, 0.012, len(idx))
        prices[publisher] = pd.Series(50.0 * np.exp(np.cumsum(r)), index=idx)
    if post_drift != 0.0:
        for launch, publisher, _, _ in dt.EVENTS:
            common = prices[publisher].index
            pos = common.searchsorted(pd.Timestamp(launch), side="right") - 1
            # bump the publisher's price path for 5 sessions after the anchor
            prices[publisher].iloc[pos + 1:pos + 6] *= (1.0 + post_drift)
    return prices


def test_build_event_table_resolves_all_events_offline():
    prices = _synthetic_price_panel(post_drift=0.0)
    ev = st.build_event_table(prices)
    assert len(ev) == len(dt.EVENTS)
    inc = ev[ev["included"]]
    assert len(inc) == 20  # every launch has synthetic coverage
    # each event anchored to its own publisher; columns present
    assert set(inc["publisher"]) == {"EA", "TTWO"}
    for col in ("pre_s", "pre_l", "post_s", "post_l"):
        assert inc[col].notna().all()


def test_event_table_picks_up_planted_post_drift():
    base = st.build_event_table(_synthetic_price_panel(post_drift=0.0))
    bumped = st.build_event_table(_synthetic_price_panel(post_drift=0.03))
    b_inc = base[base["included"]]
    u_inc = bumped[bumped["included"]]
    # a planted +3% post-launch bump lifts the mean post-window AR clearly
    assert st.one_sample_t(u_inc["post_s"].values)["mean"] > \
        st.one_sample_t(b_inc["post_s"].values)["mean"] + 0.02


def test_net_is_below_gross():
    ev = st.build_event_table(_synthetic_price_panel(post_drift=0.01), cost_bps=10.0)
    inc = ev[ev["included"]]
    assert (inc["post_s_net"] < inc["post_s"]).all()
    assert (inc["pre_l_net"] < inc["pre_l"]).all()


def test_placebo_runs_and_pvalue_in_range():
    prices = _synthetic_price_panel(post_drift=0.0)
    ev = st.build_event_table(prices)
    pl = st.placebo_pvalue(ev, prices, "post_l", k=10, n_seeds=3, n_draws_per_seed=50,
                           tail="right")
    assert 0.0 <= pl["p_value"] <= 1.0
    assert pl["n_draws"] == 150


def test_car_path_is_zero_at_anchor():
    prices = _synthetic_price_panel(post_drift=0.0)
    ev = st.build_event_table(prices)
    car = st.car_path(ev, prices)
    assert abs(car.loc[0]) < 1e-9          # normalised to 0 at the launch
    assert car.index[0] == -st.PRE_LONG_K and car.index[-1] == st.POST_LONG_K


# --------------------------------------------------------------------------- #
# Real tape — skipped offline (cache is git-ignored / absent on CI)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not dt.have_real(), reason="real cache absent offline CI")
def test_real_panel_resolves_all_launches():
    prices = dt.load_real()
    ev = st.build_event_table(prices)
    assert int(ev["included"].sum()) == len(dt.EVENTS)
