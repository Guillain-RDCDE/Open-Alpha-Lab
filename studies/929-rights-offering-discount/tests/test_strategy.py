"""Event-study logic, portfolio invariants, and the study's spine — all offline/synthetic.

The spine: on a panel with a *planted* rights-offering effect the market-model event
study recovers all three legs (announcement drop, subscription drift, post-expiry
bounce) and the deep-band split; on the matched null it stays quiet across seeds.
Costs and borrow only ever reduce return; the single execution lag is enforced.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rights_offering import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_recovers_positive_mean():
    rng = np.random.default_rng(1)
    x = 0.001 + rng.normal(0, 0.01, 5000)
    assert st.newey_west_t(x) > 3
    assert abs(st.newey_west_t(rng.normal(0, 0.01, 5000))) < 3


def test_one_sample_and_welch_agree_on_obvious_cases():
    rng = np.random.default_rng(2)
    a = 0.05 + rng.normal(0, 0.01, 200)
    b = rng.normal(0, 0.01, 200)
    assert st.one_sample_t(a) > 5
    assert st.welch_t(a, b) > 5
    assert abs(st.welch_t(b, b.copy())) < 1e-9


def test_wilson_interval_brackets_point():
    lo, hi = st.wilson_interval(60, 100)
    assert lo < 0.60 < hi


def test_market_model_recovers_planted_beta():
    rng = np.random.default_rng(3)
    mkt = rng.normal(0, 0.01, 800)
    y = 0.0003 + 1.4 * mkt + rng.normal(0, 0.002, 800)
    alpha, beta, n = st.market_model(y, mkt)
    assert abs(beta - 1.4) < 0.05
    assert abs(alpha - 0.0003) < 5e-4
    assert n == 800


def test_market_model_refuses_short_windows():
    alpha, beta, n = st.market_model(np.zeros(10), np.zeros(10))
    assert np.isnan(alpha) and np.isnan(beta)


# --------------------------------------------------------------------------- #
# Event-panel mechanics
# --------------------------------------------------------------------------- #
def test_event_panel_columns_and_finiteness(planted):
    px, ev, _ = planted
    panel = st.event_panel(px, ev)
    assert len(panel) >= 20
    for w in st.WINDOWS:
        assert f"car_{w}" in panel.columns
        assert np.isfinite(panel[f"car_{w}"]).all()
    assert (panel["n_est"] > 100).all()


def test_event_panel_drops_events_off_the_tape(planted):
    px, ev, _ = planted
    bad = ev + (("N00", "1990-01-02", "month", "deep", "cef"),
                ("ZZZZ", str(px.index[1000].date()), "month", "deep", "cef"))
    assert len(st.event_panel(px, bad)) == len(st.event_panel(px, ev))


def test_estimation_window_excludes_the_event(planted):
    """Perturbing prices only inside the event windows must not move the betas."""
    px, ev, _ = planted
    base = st.event_panel(px, ev)
    px2 = px.copy()
    idx = px.index
    for tk, ann, _p, _b, _k in ev[:5]:
        pos = int(idx.searchsorted(pd.Timestamp(ann)))
        px2.iloc[pos:pos + 40, px2.columns.get_loc(tk)] *= 1.5
    pert = st.event_panel(px2, ev)
    merged = base.merge(pert, on=["ticker", "anchor"], suffixes=("_b", "_p"))
    assert np.allclose(merged["beta_est_b"], merged["beta_est_p"])


def test_coverage_counts_are_consistent(planted):
    px, ev, _ = planted
    cov = st.coverage(px, ev)
    assert cov["n_listed"] == len(ev)
    assert cov["n_usable"] + cov["dropped"] == cov["n_listed"]


# --------------------------------------------------------------------------- #
# Portfolio invariants — one lag, costs and borrow only ever hurt
# --------------------------------------------------------------------------- #
def test_portfolio_enters_the_day_after_the_anchor():
    """With hold=(1, 5) the anchor session itself must carry no position."""
    px, ev, _ = data.synthetic_panel(n_names=1, n_events_per_name=1, n_years=6, seed=7)
    bt = st.calendar_time_portfolio(px, ev, hold=(1, 5), cost_bps=0.0)
    # bt drops the first (NaN-return) session, so locate the anchor in bt's own index
    pos = int(bt.index.searchsorted(pd.Timestamp(ev[0][1])))
    assert bt["n_active"].iloc[pos] == 0
    assert bt["n_active"].iloc[pos + 1] == 1
    assert bt["n_active"].iloc[pos + 6] == 0


def test_flat_days_earn_the_cash_leg(planted):
    px, ev, _ = planted
    bt = st.calendar_time_portfolio(px, ev, hold=(1, 20), cost_bps=0.0)
    flat = bt[bt["n_active"] == 0]
    assert len(flat) > 100
    assert (flat["r_port"] - flat["r_cash"]).abs().max() < 1e-12


def test_costs_and_borrow_only_reduce_return(planted):
    px, ev, _ = planted
    means = []
    for c in (0.0, 25.0, 100.0):
        bt = st.calendar_time_portfolio(px, ev, hold=(1, 40), cost_bps=c)
        means.append(bt["r_port"].mean())
    assert means[0] >= means[1] >= means[2]
    shorts = []
    for b in (0.0, 300.0, 800.0):
        bt = st.calendar_time_portfolio(px, ev, hold=(1, 40), cost_bps=25.0,
                                        side=-1, borrow_bps_ann=b)
        shorts.append(bt["r_port"].mean())
    assert shorts[0] >= shorts[1] >= shorts[2]


def test_tradability_reports_both_arms(planted):
    px, ev, _ = planted
    t = st.tradability(px, ev, hold=(1, 40), cost_bps=25.0)
    for k in ("sharpe_adv", "t_diff", "t_port", "invested_frac"):
        assert np.isfinite(t[k])
    assert 0.0 < t["invested_frac"] <= 1.0
    for k in ("alpha_ann_pct", "beta", "t_alpha"):
        assert np.isfinite(t["alpha"][k])


def test_short_book_is_cash_collateralised(planted):
    """Costs must not be one-sided: the short leg pays borrow AND earns its rebate.

    A short book is collateralised, so on a flat day it earns cash exactly like the long
    book, and on an invested day it earns cash minus borrow rather than nothing.
    """
    px, ev, _ = planted
    bt = st.calendar_time_portfolio(px, ev, hold=(1, 20), cost_bps=0.0, side=-1,
                                    borrow_bps_ann=0.0)
    flat = bt[bt["n_active"] == 0]
    assert len(flat) > 100
    assert (flat["r_port"] - flat["r_cash"]).abs().max() < 1e-12
    live = bt[bt["n_active"] > 0]
    assert (live["r_port"] - live["r_cash"]).abs().max() > 0   # it still takes risk
    # and the borrow charge is exactly the annualised rate on invested days
    bt2 = st.calendar_time_portfolio(px, ev, hold=(1, 20), cost_bps=0.0, side=-1,
                                     borrow_bps_ann=252.0)
    drop = (bt["r_port"] - bt2["r_port"])[bt["n_active"] > 0]
    assert np.allclose(drop.to_numpy(), 252.0 * 1e-4 / st.TRADING_DAYS)


def test_alpha_vs_market_prices_out_borrowed_beta():
    """A pure levered-market series has a big Sharpe and no alpha; the stat must say so."""
    rng = np.random.default_rng(7)
    mkt = rng.normal(0.0004, 0.01, 3000)
    lev = 0.6 * mkt + rng.normal(0.0, 0.001, 3000)   # no alpha, just rented beta + noise
    a = st.alpha_vs_market(lev, mkt)
    assert abs(a["beta"] - 0.6) < 0.02
    assert abs(a["alpha_ann_pct"]) < 1.0 and abs(a["t_alpha"]) < 2.0
    b = st.alpha_vs_market(lev + 0.0004, mkt)   # now with real alpha (+10%/yr)
    assert b["alpha_ann_pct"] > 5.0 and b["t_alpha"] > 3.0
    assert abs(b["beta"] - a["beta"]) < 1e-9    # alpha must not leak into beta


# --------------------------------------------------------------------------- #
# Bootstraps, jackknives, permutation
# --------------------------------------------------------------------------- #
def test_bootstrap_mean_ci_brackets_point(planted):
    px, ev, _ = planted
    panel = st.event_panel(px, ev)
    ci = st.bootstrap_mean_ci(panel["car_post_expiry"].to_numpy(), n_boot=800, seed=929)
    assert ci["ci_low"] <= ci["mean_pct"] <= ci["ci_high"]


def test_block_bootstrap_mean_ci_brackets_point():
    rng = np.random.default_rng(4)
    x = rng.normal(0.0005, 0.01, 2000)
    ci = st.block_bootstrap_mean_ci(x, n_boot=500, seed=929)
    assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]


def test_jackknives_run_and_bracket_the_point(planted):
    px, ev, _ = planted
    panel = st.event_panel(px, ev)
    j = st.jackknife(panel, "announce")
    assert j["min_mean_pct"] <= j["mean_pct"] <= j["max_mean_pct"]
    ji = st.jackknife_issuer(panel, "announce")
    assert len(ji) >= 5 and ji["t"].is_monotonic_increasing


def test_permutation_test_is_calibrated_on_random_labels(planted):
    """Random labels must not produce a small permutation p-value."""
    px, ev, _ = planted
    panel = st.event_panel(px, ev).copy()
    rng = np.random.default_rng(11)
    panel["discount_band"] = np.where(rng.random(len(panel)) < 0.5, "deep", "moderate")
    p = st.permutation_discount_test(panel, "post_expiry", n_perm=1500, seed=5)
    assert p["p_two_sided"] > 0.02


# --------------------------------------------------------------------------- #
# The study's spine — the harness recovers a planted effect and is quiet on the null
# --------------------------------------------------------------------------- #
def test_planted_effect_is_recovered(planted):
    px, ev, truth = planted
    d = st.synthetic_detect(px, ev)
    assert d["announce_mean_pct"] < -2.0 and d["announce_t"] < -3.0
    assert d["subscription_mean_pct"] < -3.0 and d["subscription_t"] < -2.0
    assert d["post_expiry_mean_pct"] > 2.0 and d["post_expiry_t"] > 2.0


def test_planted_deep_band_is_the_stronger_one(planted):
    """Deep-discount names carry 1.5x the planted effect, so the split must find it."""
    px, ev, _ = planted
    panel = st.event_panel(px, ev)
    d = st.discount_split(panel, "post_expiry")
    assert d["mean_deep_pct"] > d["mean_rest_pct"]
    p = st.permutation_discount_test(panel, "post_expiry", n_perm=2000, seed=929)
    assert p["p_two_sided"] < 0.20


def test_null_is_centered_across_seeds():
    """No planted effect: window means must sit near zero and rarely clear |t| = 2."""
    means, ts = [], []
    for s in range(10):
        px, ev, _ = data.synthetic_panel(signal_strength=0.0, seed=929 + s)
        d = st.synthetic_detect(px, ev)
        means.append(d["post_expiry_mean_pct"])
        ts.append(d["post_expiry_t"])
    means, ts = np.array(means), np.array(ts)
    assert abs(means.mean()) < 1.0
    assert (np.abs(ts) >= 2.0).sum() <= 2


def test_null_subscription_window_is_centered_across_seeds():
    """The widest window is the one the real tape trips on — guard it explicitly.

    A single null seed *can* throw |t| = 2 over 28 days in this universe (that is the
    point made in docs/results.md), so the guard is on the average across seeds and on
    the firing *rate*, not on any one seed.
    """
    means, ts = [], []
    for s in range(12):
        px, ev, _ = data.synthetic_panel(signal_strength=0.0, seed=929 + s)
        d = st.synthetic_detect(px, ev)
        means.append(d["subscription_mean_pct"])
        ts.append(d["subscription_t"])
    means, ts = np.array(means), np.array(ts)
    assert abs(means.mean()) < 1.5          # no systematic drift in the estimator
    assert (np.abs(ts) >= 2.0).sum() <= 3   # ~chance, not a biased detector


def test_null_announcement_window_is_quiet():
    means = []
    for s in range(10):
        px, ev, _ = data.synthetic_panel(signal_strength=0.0, seed=929 + s)
        means.append(st.synthetic_detect(px, ev)["announce_mean_pct"])
    assert abs(np.mean(means)) < 0.75


def test_placebo_is_centered_on_the_null():
    """On a null panel the observed mean must not stand out against random anchors."""
    px, ev, _ = data.synthetic_panel(n_names=8, n_events_per_name=2, n_years=10,
                                     signal_strength=0.0, seed=929)
    pl = st.placebo(px, ev, window="post_expiry", n_draws=60, seed=929)
    assert pl["n_draws"] >= 40
    assert abs(pl["z"]) < 3.0


def test_placebo_draw_range_is_honoured():
    """The date range of the pseudo-anchors is a choice, and it must actually bind.

    Restricting the draw to a sub-period has to change the placebo distribution — that
    is the whole point of the era-matched version used on the real tape.
    """
    px, ev, _ = data.synthetic_panel(n_names=8, n_events_per_name=2, n_years=10,
                                     signal_strength=0.0, seed=929)
    idx = px.index
    lo, hi = str(idx[400].date()), str(idx[1400].date())
    wide = st.placebo(px, ev, window="post_expiry", n_draws=60, seed=929)
    narrow = st.placebo(px, ev, window="post_expiry", n_draws=60, seed=929,
                        lo_date=lo, hi_date=hi)
    assert narrow["n_draws"] >= 40
    assert narrow["placebo_sd_pct"] != wide["placebo_sd_pct"]
    assert narrow["lo_date"] == lo and narrow["hi_date"] == hi
    # a range too narrow to hold a window must refuse rather than pretend
    assert st.placebo(px, ev, n_draws=5, lo_date=lo, hi_date=str(idx[420].date()))[
        "n_draws"] == 0


def test_clustered_placebo_sees_a_planted_effect_and_not_a_null():
    """Sliding the whole list keeps the clustering; it must still have power."""
    px, ev, _ = data.synthetic_panel(n_names=8, n_events_per_name=3, n_years=12,
                                     signal_strength=0.0, seed=929)
    pl0 = st.clustered_placebo(px, ev, window="post_expiry", n_draws=120, seed=929,
                               max_shift_years=4)
    assert pl0["n_draws"] >= 20 and abs(pl0["z"]) < 3.0
    px1, ev1, _ = data.synthetic_panel(n_names=8, n_events_per_name=3, n_years=12,
                                       signal_strength=1.0, seed=929)
    pl1 = st.clustered_placebo(px1, ev1, window="post_expiry", n_draws=120, seed=929,
                               max_shift_years=4)
    assert pl1["z"] > pl0["z"]


def test_anchor_jitter_and_timetable_sweeps_run(planted):
    px, ev, _ = planted
    rows = st.anchor_jitter(px, ev, window="post_expiry", shifts=(-5, 0, 5))
    assert len(rows) == 3 and all(np.isfinite(r["mean_pct"]) for r in rows)
    tt = st.timetable_sweep(px, ev, expiry_days_grid=(28, 40))
    assert len(tt) == 2 and all(np.isfinite(r["subscription_mean_pct"]) for r in tt)


def test_era_cut_returns_both_halves(planted):
    px, ev, _ = planted
    idx = px.index
    mid = str(idx[len(idx) // 2].date())
    eras = st.era_cut(px, ev, split=mid, window="post_expiry")
    assert eras["early"] is not None and eras["late"] is not None
    for e in eras.values():
        assert np.isfinite(e["mean_pct"]) and np.isfinite(e["t"])


def test_cost_and_borrow_sweeps_are_monotone(planted):
    px, ev, _ = planted
    cs = st.cost_sweep(px, ev, hold=(1, 40), grid=(0.0, 25.0, 100.0))
    assert cs[0]["mean_diff_bps"] >= cs[1]["mean_diff_bps"] >= cs[2]["mean_diff_bps"]
    bs = st.borrow_sweep(px, ev, hold=(1, 40), grid=(0.0, 300.0, 800.0))
    assert bs[0]["sharpe_port"] >= bs[1]["sharpe_port"] >= bs[2]["sharpe_port"]
