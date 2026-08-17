"""Estimator invariants and the study's spine — all offline and synthetic.

The spine: on a panel with a *planted* creation-halt premium the estimator must find it
(positive announcement z, positive drift while suspended, negative fade on resumption);
on the null panel it must find nothing. Around that sit the mechanical invariants — one
execution lag and no look-ahead, placebo windows that never overlap the event they
control for, costs and borrow that only ever reduce a net number, and the identity that
makes this long-short spread an excess-of-cash quantity by construction.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from creation_halt import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# The spread
# --------------------------------------------------------------------------- #
def test_spread_is_zero_for_identical_legs():
    idx = pd.bdate_range("2020-01-02", periods=200)
    p = pd.Series(100 * np.exp(np.cumsum(np.full(200, 0.001))), index=idx)
    s = st.pair_spread(p, p, direction=1)
    assert float(s.abs().max()) < 1e-12


def test_spread_direction_flips_the_sign():
    idx = pd.bdate_range("2020-01-02", periods=200)
    rng = np.random.default_rng(0)
    f = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200))), index=idx)
    g = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 200))), index=idx)
    assert np.allclose(st.pair_spread(f, g, +1).to_numpy(),
                       -st.pair_spread(f, g, -1).to_numpy())


def test_spread_uses_only_common_sessions():
    """A 24/7 leg must be sampled on the fund's sessions, not padded with phantom days."""
    idx = pd.bdate_range("2020-01-02", periods=100)
    daily = pd.date_range("2020-01-02", periods=140, freq="D")
    rng = np.random.default_rng(1)
    f = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 100))), index=idx)
    g = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 140))), index=daily)
    s = st.pair_spread(f, g, +1)
    assert len(s) == len(idx.intersection(daily)) - 1
    assert s.index.isin(idx).all()


def test_excess_of_cash_identity_is_exact():
    idx = pd.bdate_range("2018-01-02", periods=400)
    rng = np.random.default_rng(2)
    f = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400))), index=idx)
    g = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, 400))), index=idx)
    cash = pd.Series(100 * (1 + 0.04 / 252) ** np.arange(400), index=idx)
    assert st.excess_of_cash_check(f, g, cash, +1) < 1e-12


# --------------------------------------------------------------------------- #
# The execution lag and the window helpers
# --------------------------------------------------------------------------- #
def test_window_starts_the_day_after_the_event():
    idx = pd.bdate_range("2021-01-04", periods=60)
    s = pd.Series(np.arange(60, dtype=float), index=idx)
    total, i0, i1 = st.window_sum(s, idx[10], 5)
    assert i0 == 11 and i1 == 16
    assert total == float(np.arange(11, 16).sum())


def test_announcement_day_move_is_excluded_by_the_lag():
    """A one-day spike ON the announcement day must not enter the measured window."""
    idx = pd.bdate_range("2021-01-04", periods=120)
    s = pd.Series(np.zeros(120), index=idx)
    s.iloc[60] = 0.25                       # the announcement-day jump
    ev = {"key": "T", "halt": str(idx[60].date()), "resume": str(idx[100].date())}
    car, _, _ = st.window_sum(s, ev["halt"], 10)
    assert car == pytest.approx(0.0)


def test_window_sum_is_nan_when_the_window_runs_off_the_tape():
    idx = pd.bdate_range("2021-01-04", periods=30)
    s = pd.Series(np.zeros(30), index=idx)
    car, _, _ = st.window_sum(s, idx[27], 10)
    assert np.isnan(car)


def test_placebo_windows_exclude_the_event_window():
    idx = pd.bdate_range("2021-01-04", periods=400)
    s = pd.Series(np.zeros(400), index=idx)
    s.iloc[200:210] = 1.0                   # the "event" lives at positions 200-209
    pl = st.placebo_sums(s, 10, exclude=[(200, 210), (390, 400)])
    assert pl.size > 100
    assert float(np.abs(pl).max()) == 0.0   # no placebo window touches the event


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_recovers_a_positive_mean_and_stays_quiet_on_noise():
    rng = np.random.default_rng(3)
    assert st.newey_west_t(0.001 + rng.normal(0, 0.01, 5000)) > 3
    assert abs(st.newey_west_t(rng.normal(0, 0.01, 5000))) < 3


def test_block_bootstrap_ci_brackets_the_point_estimate():
    rng = np.random.default_rng(4)
    x = 0.0005 + rng.normal(0, 0.01, 1500)
    ci = st.block_bootstrap_mean_ci(x, n_boot=400, seed=918)
    assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]


def test_event_resample_ci_is_wide_with_few_events():
    ci = st.event_resample_ci([1.0, -0.5, 0.2, 3.0, -1.0], n_boot=2000, seed=918)
    assert ci["ci_low"] < ci["mean"] < ci["ci_high"]
    assert ci["n_events"] == 5


def test_welch_and_one_sample_t_are_finite():
    rng = np.random.default_rng(5)
    a, b = rng.normal(0.001, 0.01, 300), rng.normal(0, 0.01, 900)
    assert np.isfinite(st.welch_t(a, b))
    assert np.isfinite(st.one_sample_t(a))


# --------------------------------------------------------------------------- #
# Costs and borrow
# --------------------------------------------------------------------------- #
def _one_pair(signal_strength=1.0):
    frames, evs = data.synthetic_panel(n_events=1, signal_strength=signal_strength, seed=918)
    return st.build_spreads_from_frames(frames, evs), evs


def test_costs_and_borrow_only_reduce_the_net():
    spreads, evs = _one_pair()
    base = st.trade_event(spreads[evs[0]["key"]], evs[0], cost_bps=0.0, borrow_ann=0.0)
    dearer = st.trade_event(spreads[evs[0]["key"]], evs[0], cost_bps=25.0, borrow_ann=0.0)
    borrowed = st.trade_event(spreads[evs[0]["key"]], evs[0], cost_bps=0.0, borrow_ann=30.0)
    assert base["gross_pct"] == pytest.approx(dearer["gross_pct"])
    assert dearer["net_pct"] < base["net_pct"]
    assert borrowed["net_pct"] < base["net_pct"]


def test_daily_rebalancing_is_charged_and_scales_with_the_hold():
    """exp(sum x) is a continuously dollar-neutral position; holding one flat costs turnover.

    Charging only the four entry/exit crossings understates a long hold badly, so the
    rebalancing leg must exist, must scale with the number of sessions held, and must be
    inside net_pct.
    """
    spreads, evs = _one_pair()
    key = evs[0]["key"]
    tr = st.trade_event(spreads[key], evs[0], cost_bps=10.0, borrow_ann=0.0)
    assert tr["rebal_pct"] > 0.0
    assert tr["net_pct"] == pytest.approx(
        tr["gross_pct"] - tr["cost_pct"] - tr["rebal_pct"] - tr["borrow_pct"])
    short = st.trade_event(spreads[key], evs[0], cost_bps=10.0, borrow_ann=0.0,
                           hold="blind", blind_days=10)
    long_ = st.trade_event(spreads[key], evs[0], cost_bps=10.0, borrow_ann=0.0,
                           hold="blind", blind_days=200)
    assert long_["rebal_pct"] > short["rebal_pct"]
    assert st.trade_event(spreads[key], evs[0], cost_bps=0.0,
                          borrow_ann=0.0)["rebal_pct"] == pytest.approx(0.0)


def test_blind_hold_never_uses_the_resumption_date():
    """The no-look-ahead variant must be invariant to where we think the halt ended."""
    spreads, evs = _one_pair()
    key = evs[0]["key"]
    moved = dict(evs[0])
    moved["resume"] = str((pd.Timestamp(moved["resume"]) + pd.Timedelta(days=400)).date())
    a = st.trade_event(spreads[key], evs[0], hold="blind", blind_days=60)
    b = st.trade_event(spreads[key], moved, hold="blind", blind_days=60)
    assert a["gross_pct"] == pytest.approx(b["gross_pct"])
    # ...whereas the hindsight-exit variant is NOT invariant, which is the whole point.
    c = st.trade_event(spreads[key], evs[0], hold="halt")
    d = st.trade_event(spreads[key], moved, hold="halt")
    assert c["n_days"] != d["n_days"]


def test_fee_drag_is_signed_and_one_sided_against_an_unfeed_ruler():
    fees = {"F": 2.0, "SPOT": 0.0, "TWIN": 2.0}
    long_ev = {"fund": "F", "proxy": "SPOT", "direction": +1}
    short_ev = {"fund": "F", "proxy": "SPOT", "direction": -1}
    matched = {"fund": "F", "proxy": "TWIN", "direction": +1}
    # a 2%/yr fee against an unfeed ruler drags the +1 spread down by 2/252 %/day
    assert st.fee_drag_bps(long_ev, fees) == pytest.approx(-2.0 / 252 * 1e2)
    # ...and flips sign with the event direction, so a redemption freeze is flattered
    assert st.fee_drag_bps(short_ev, fees) == pytest.approx(+2.0 / 252 * 1e2)
    # two funds with the same fee cancel exactly
    assert st.fee_drag_bps(matched, fees) == pytest.approx(0.0)


def test_regime_control_excludes_the_post_resumption_fade():
    """Leaving the fade in the control depresses the baseline and flatters the gap."""
    idx = pd.bdate_range("2015-01-01", periods=600)
    x = pd.Series(np.zeros(600), index=idx)
    x.iloc[100:200] = 0.002          # the halt drift
    x.iloc[200:220] = -0.010         # the fade, right after resumption
    ev = {"key": "T", "fund": "F", "proxy": "P", "direction": +1,
          "halt": str(idx[99].date()), "resume": str(idx[199].date())}
    contaminated = st.regime_drift(x, ev, fade_buffer=0, fees={})
    clean = st.regime_drift(x, ev, fade_buffer=20, fees={})
    assert contaminated["out_bps"] < 0          # dragged down by the fade
    assert clean["out_bps"] == pytest.approx(0.0)
    assert clean["out_bps"] > contaminated["out_bps"]
    assert clean["in_bps"] == pytest.approx(contaminated["in_bps"])


def test_multiplicity_bar_is_reported_for_the_design():
    assert st.family_size(5) == 30
    assert st.bonferroni_percentile(30) == pytest.approx(1 - 0.05 / 30)


def test_placebo_percentile_is_also_reported_on_non_overlapping_windows():
    """An overlapping pool has ~k times fewer independent draws than rows.

    Quoting "percentile 0.999 of 1,978" when the windows overlap by 19/20 implies a
    resolution the tape does not contain, so the estimator must also return the
    non-overlapping percentile, whose denominator is ~n/k.
    """
    frames, evs = data.synthetic_panel(n_events=1, signal_strength=1.0, seed=918)
    spreads = st.build_spreads_from_frames(frames, evs)
    r = st.event_car(spreads[evs[0]["key"]], evs[0], k=20)
    assert r["n_placebo_indep"] < r["n_placebo"] / 10
    assert r["n_placebo_indep"] == pytest.approx(r["n_placebo"] / 20, rel=0.2)
    assert 0.0 <= r["pct_indep"] <= 1.0
    f = st.resumption_fade(spreads[evs[0]["key"]], evs[0], k=20)
    assert f["n_placebo_indep"] < f["n_placebo"] / 10
    # the planted premium still shows up on the honest pool
    assert r["pct_indep"] > 0.9 and f["pct_indep"] < 0.1


def test_trade_sweep_is_monotone_in_borrow():
    spreads, evs = _one_pair()
    sw = st.trade_sweep(spreads, evs, cost_grid=(10.0,), borrow_grid=(0.0, 5.0, 20.0))
    means = sw.sort_values("borrow_ann_pct")["mean_net_pct"].to_numpy()
    assert means[0] > means[1] > means[2]


def test_fade_trade_is_the_reverse_position():
    spreads, evs = _one_pair()
    key = evs[0]["key"]
    fade = st.trade_event(spreads[key], evs[0], cost_bps=0.0, borrow_ann=0.0, hold="fade")
    raw, _, _ = st.window_sum(spreads[key], evs[0]["resume"], fade["n_days"])
    assert fade["gross_pct"] / 100.0 == pytest.approx(np.exp(-raw) - 1.0, rel=1e-9)


# --------------------------------------------------------------------------- #
# The study's spine — the machinery finds a planted halt and invents none
# --------------------------------------------------------------------------- #
def test_planted_panel_gives_a_positive_pooled_announcement_z(planted):
    frames, evs = planted
    spreads = st.build_spreads_from_frames(frames, evs)
    pooled = st.pooled_z(st.event_study(spreads, evs, k=20))
    assert pooled["mean_z"] > 0.5
    assert pooled["n_positive"] >= 5


def test_planted_panel_drifts_while_suspended(planted):
    frames, evs = planted
    spreads = st.build_spreads_from_frames(frames, evs)
    reg = st.regime_table(spreads, evs)
    assert reg["in_bps"].mean() > 5.0
    assert (reg["in_bps"] > reg["out_bps"]).all()


def test_planted_panel_fades_on_resumption(planted):
    frames, evs = planted
    spreads = st.build_spreads_from_frames(frames, evs)
    fad = st.fade_study(spreads, evs, k=20)
    assert fad["z"].mean() < -1.0
    assert (fad["z"] < 0).sum() >= 5


def test_null_panel_shows_no_announcement_effect(null):
    frames, evs = null
    spreads = st.build_spreads_from_frames(frames, evs)
    pooled = st.pooled_z(st.event_study(spreads, evs, k=20))
    assert abs(pooled["mean_z"]) < 0.8


def test_null_panel_shows_no_drift_and_no_fade(null):
    frames, evs = null
    spreads = st.build_spreads_from_frames(frames, evs)
    reg = st.regime_table(spreads, evs)
    fad = st.fade_study(spreads, evs, k=20)
    assert abs(reg["in_bps"].mean()) < 6.0
    assert abs(fad["z"].mean()) < 1.0


def test_null_is_centred_across_seeds():
    ts = np.array([st.synthetic_detect(0.0, seed=918 + 11 * s)["pooled_t"] for s in range(8)])
    assert abs(ts.mean()) < 1.5
    assert (np.abs(ts) >= 2).sum() <= 2


def test_planted_beats_null_on_the_same_seeds():
    pl = np.array([st.synthetic_detect(1.0, seed=918 + 11 * s)["mean_in_bps"] for s in range(6)])
    nl = np.array([st.synthetic_detect(0.0, seed=918 + 11 * s)["mean_in_bps"] for s in range(6)])
    assert pl.mean() > nl.mean() + 5.0
    assert (pl > nl).all()


def test_detector_scales_with_the_planted_premium():
    weak = st.synthetic_detect(1.0, seed=918, premium_per_day_bps=3.0)["mean_in_bps"]
    strong = st.synthetic_detect(1.0, seed=918, premium_per_day_bps=30.0)["mean_in_bps"]
    assert strong > weak + 10.0


# --------------------------------------------------------------------------- #
# Robustness helpers return usable shapes
# --------------------------------------------------------------------------- #
def test_jackknife_drops_one_event_at_a_time(planted):
    frames, evs = planted
    spreads = st.build_spreads_from_frames(frames, evs)
    tbl = st.event_study(spreads, evs, k=20)
    jk = st.jackknife(tbl)
    assert len(jk) == len(tbl)
    assert (jk["n"] == len(tbl) - 1).all()


def test_resume_date_sweep_covers_every_shift(planted):
    frames, evs = planted
    spreads = st.build_spreads_from_frames(frames, evs)
    sw = st.resume_date_sweep(spreads, evs, shifts=(-5, 0, 5), k=20)
    assert list(sw.index) == [-5, 0, 5]
    assert np.isfinite(sw["mean_fade_z"].to_numpy()).all()


def test_spread_sharpe_matches_a_hand_computation():
    rng = np.random.default_rng(6)
    x = pd.Series(rng.normal(0.0004, 0.01, 1000))
    assert st.spread_sharpe(x) == pytest.approx(
        x.mean() / x.std(ddof=1) * np.sqrt(252), rel=1e-9)


def test_ruler_split_separates_the_two_kinds(planted):
    frames, evs = planted
    for i, ev in enumerate(evs):
        ev["ruler"] = "exact" if i % 2 == 0 else "curve-mismatched"
    spreads = st.build_spreads_from_frames(frames, evs)
    split = st.ruler_split(st.event_study(spreads, evs, k=20), evs)
    assert split["exact"]["n"] == 3 and split["curve-mismatched"]["n"] == 3
