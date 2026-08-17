"""Event-study logic, inference primitives, and the study's spine — all offline/synthetic.

The spine: on a panel of staggered IPOs carrying a *planted* first-year slide the
estimator recovers roughly the planted magnitude with a decisively negative *t*; on the
null panel it stays quiet across seeds. Around that, the mechanical invariants: the
offering-price entry needs raw closes, costs and borrow only ever reduce the mirror
trade's return, the mirror trade respects its one-day execution lag, and the
time-to-discount proxy fires on a planted slide and not on a flat one.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cef_ipo import data, strategy as st  # noqa: E402


def _frame(panel):
    """Same reshape as ``conftest._as_frame`` — for panels a fixture does not build."""
    cols, ipos = {}, []
    for key, px in panel.items():
        cols[key] = px["fund"]
        cols[key + "_B"] = px["bench"]
        ipos.append((key, 20.0, "synthetic", key + "_B"))
    prices = pd.DataFrame(cols).sort_index()
    prices.index.name = "date"
    return prices, tuple(ipos), prices[[k for k, _, _, _ in ipos]].copy()


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_one_sample_t_recovers_a_negative_mean():
    rng = np.random.default_rng(0)
    x = -0.10 + rng.normal(0, 0.10, 40)
    assert st.one_sample_t(x) < -3
    assert abs(st.one_sample_t(rng.normal(0, 0.10, 40))) < 3


def test_newey_west_recovers_positive_mean():
    rng = np.random.default_rng(1)
    assert st.newey_west_t(0.001 + rng.normal(0, 0.01, 5000)) > 3
    assert abs(st.newey_west_t(rng.normal(0, 0.01, 5000))) < 3


def test_welch_and_wilson_behave():
    rng = np.random.default_rng(2)
    a = rng.normal(0.0, 1.0, 200)
    b = rng.normal(1.0, 1.0, 200)
    assert st.welch_t(a, b) < -5
    lo, hi = st.wilson_interval(25, 28)
    assert lo < 25 / 28 < hi


def test_cluster_bootstrap_brackets_the_mean():
    rng = np.random.default_rng(3)
    v = -0.08 + rng.normal(0, 0.05, 30)
    g = np.repeat(np.arange(6), 5)
    ci = st.cluster_bootstrap_ci(v, g, n_boot=800, seed=931)
    assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]
    assert ci["n_clusters"] == 6


def test_cluster_bootstrap_is_wider_than_ignoring_clusters():
    """Clustered data: resampling whole groups must not pretend to more information."""
    rng = np.random.default_rng(4)
    shock = rng.normal(0, 0.10, 6)                     # one shock per cluster
    v = np.repeat(shock, 5) + rng.normal(0, 0.01, 30)  # near-identical inside a cluster
    g = np.repeat(np.arange(6), 5)
    clustered = st.cluster_bootstrap_ci(v, g, n_boot=1500, seed=931)
    naive = st.cluster_bootstrap_ci(v, np.arange(30), n_boot=1500, seed=931)
    assert (clustered["ci_high"] - clustered["ci_low"]) > (naive["ci_high"] - naive["ci_low"])


# --------------------------------------------------------------------------- #
# The event study — mechanics
# --------------------------------------------------------------------------- #
def test_offer_entry_requires_raw_closes(planted_panel):
    px = planted_panel[0]["F00"]
    with pytest.raises(ValueError):
        st.fund_event_table(px["fund"], px["bench"], entry="offer")


def test_close_entry_needs_no_assumption(planted_panel):
    px = planted_panel[0]["F00"]
    rec = st.fund_event_table(px["fund"], px["bench"], entry="close")
    assert rec["day1_gap_pct"] == 0.0
    assert np.isfinite(rec["abn_12m"])


def test_offer_entry_moves_the_return_by_exactly_the_day_one_gap(planted_panel):
    px = planted_panel[0]["F00"]
    raw = px["fund"] * 1.0
    a = st.fund_event_table(px["fund"], px["bench"], offer=20.0, raw_fund=raw, entry="offer")
    b = st.fund_event_table(px["fund"], px["bench"], entry="close")
    gap = 1.0 + a["day1_gap_pct"] / 100.0
    assert np.isclose(1.0 + a["fund_12m"], gap * (1.0 + b["fund_12m"]))


def test_build_table_covers_every_fund(planted_frame):
    prices, ipos, raw = planted_frame
    tbl = st.build_table(prices, ipos, raw=raw)
    assert len(tbl) == len(ipos)
    assert {"abn_1m", "abn_3m", "abn_6m", "abn_12m", "vintage"}.issubset(tbl.columns)
    assert tbl["ipo_date"].is_monotonic_increasing


def test_horizon_summary_shape(planted_frame):
    prices, ipos, raw = planted_frame
    summ = st.horizon_summary(st.build_table(prices, ipos, raw=raw))
    assert list(summ.index) == ["1m", "3m", "6m", "12m"]
    assert (summ["n_funds"] == len(ipos)).all()


def test_benchmark_is_aligned_not_leaked(planted_panel):
    """The benchmark leg is read from the fund's IPO date on — never from before it."""
    px = planted_panel[0]["F03"]
    early = px["bench"].copy()
    early.iloc[:5] *= 5.0          # perturb bars before this fund's own start
    shifted = px["fund"].iloc[10:]
    a = st.fund_event_table(shifted, px["bench"], entry="close")
    b = st.fund_event_table(shifted, early, entry="close")
    assert np.isclose(a["abn_12m"], b["abn_12m"])


# --------------------------------------------------------------------------- #
# Time-to-discount proxy
# --------------------------------------------------------------------------- #
def test_time_to_threshold_fires_on_a_planted_slide(planted_panel):
    px = planted_panel[0]["F00"]
    d = st.time_to_threshold(px["fund"], px["bench"], threshold_pct=5.0, entry="close")
    assert np.isfinite(d) and 0 < d < 504


def test_time_to_threshold_is_quiet_without_a_slide():
    """A fund that tracks its benchmark exactly never reaches the threshold."""
    idx = pd.bdate_range("2015-01-02", periods=600)
    bench = pd.Series(100 * (1.0 + 0.0002) ** np.arange(600), index=idx)
    fund = bench * 0.2
    d = st.time_to_threshold(fund, bench, threshold_pct=5.0, entry="close")
    assert not np.isfinite(d)


def test_time_to_discount_stats_counts_non_crossers(planted_frame):
    prices, ipos, raw = planted_frame
    tbl = st.time_to_discount_table(prices, ipos, raw, threshold_pct=5.0)
    s = st.time_to_discount_stats(tbl)
    assert s["n_funds"] == len(ipos)
    assert 0.0 <= s["share_crossed"] <= 1.0
    assert s["q1_days"] <= s["median_days"] <= s["q3_days"]


# --------------------------------------------------------------------------- #
# Calendar-time portfolio
# --------------------------------------------------------------------------- #
def test_calendar_time_spread_is_negative_on_a_planted_panel(planted_frame):
    prices, ipos, raw = planted_frame
    sp = st.calendar_time_spread(prices, ipos, window=252)
    s = st.calendar_time_stats(sp)
    assert s["ann_pct"] < 0 and s["tstat_hac"] < -2


def test_calendar_time_spread_is_quiet_on_the_null(null_frame):
    prices, ipos, raw = null_frame
    s = st.calendar_time_stats(st.calendar_time_spread(prices, ipos, window=252))
    assert abs(s["tstat_hac"]) < 2


def test_calendar_time_spread_respects_the_execution_lag(planted_frame):
    """start_day=1 drops the IPO day itself — the one execution lag, applied once."""
    prices, ipos, raw = planted_frame
    lagged = st.calendar_time_spread(prices, ipos, window=252, start_day=1)
    same_day = st.calendar_time_spread(prices, ipos, window=252, start_day=0)
    assert len(lagged) <= len(same_day)


def test_calendar_time_book_never_earns_the_ipo_close_move(planted_frame):
    """The book is established at the t+1 close, so its first P&L day is the fund's THIRD.

    The daily return dated at age *k* is earned by whoever was already positioned at the
    close of age *k−1*. Entering at the t+1 close (``start_day=1``) therefore means ages
    2 … ``window`` — never age 1, the IPO-close → next-close move that nobody holding to
    the stated convention could have had. An off-by-one here silently hands the portfolio
    a day it did not own.
    """
    prices, ipos, raw = planted_frame
    tk = ipos[0][0]
    dates = prices[tk].dropna().index
    for w in (21, 63, 252):
        sp = st.calendar_time_spread(prices, ipos[:1], window=w, start_day=1)
        assert len(sp) == w - 1
        assert sp.index[0] == dates[2]
        assert sp.index[-1] == dates[w]


def test_block_bootstrap_brackets_the_mean(planted_frame):
    prices, ipos, raw = planted_frame
    sp = st.calendar_time_spread(prices, ipos, window=252)
    b = st.block_bootstrap_mean_ci(sp, n_boot=500, seed=931)
    assert b["ci_low"] <= b["ann_pct"] <= b["ci_high"]


# --------------------------------------------------------------------------- #
# The beta control — "is the hole just leverage?"
# --------------------------------------------------------------------------- #
def test_seasoned_beta_recovers_planted_leverage():
    panel, _ = data.synthetic_panel(n_funds=6, beta=1.6, signal_strength=0.0, seed=931)
    betas = [st.seasoned_beta(px["fund"], px["bench"]) for px in panel.values()]
    assert np.isfinite(betas).all()
    assert abs(float(np.mean(betas)) - 1.6) < 0.10


def test_seasoned_beta_is_nan_without_enough_seasoned_history():
    """A fund with no life after the event window cannot have a seasoned beta invented."""
    px, _ = data.synthetic_daily(n_days=260, seed=931)
    assert not np.isfinite(st.seasoned_beta(px["fund"], px["bench"]))


def test_beta_adjustment_removes_a_planted_leverage_bias():
    """Levered funds, NO planted IPO hole: beta-1 is biased, the beta-fitted CAR is not."""
    panel, _ = data.synthetic_panel(beta=1.6, signal_strength=0.0, seed=931)
    prices, ipos, _ = _frame(panel)
    s = st.beta_adjusted_summary(st.beta_adjusted_table(prices, ipos))
    assert abs(s["mean_beta"] - 1.6) < 0.10
    assert abs(s["betafit_mean_pct"]) < abs(s["beta1_mean_pct"])
    assert abs(s["betafit_mean_pct"]) < 2.0


def test_beta_adjusted_car_recovers_the_planted_hole_at_beta_one(planted_frame):
    """When there is no leverage, adjusting for beta must not erase a real hole."""
    prices, ipos, raw = planted_frame
    s = st.beta_adjusted_summary(st.beta_adjusted_table(prices, ipos))
    assert abs(s["mean_beta"] - 1.0) < 0.10
    assert s["betafit_t"] < -3 and s["beta1_t"] < -3
    assert abs(s["betafit_mean_pct"] - s["beta1_mean_pct"]) < 1.0
    assert -14.0 < s["betafit_mean_pct"] < -6.0


def test_market_model_car_respects_the_execution_lag(planted_frame):
    """The CAR window starts one day after the entry close, like the mirror trade."""
    prices, ipos, raw = planted_frame
    f, b = prices[ipos[0][0]], prices[ipos[0][3]]
    a = st.market_model_car(f, b, horizon=63, lag_days=1)
    c = st.market_model_car(f, b, horizon=63, lag_days=0)
    assert np.isfinite(a) and np.isfinite(c) and not np.isclose(a, c)
    assert not np.isfinite(st.market_model_car(f, b, horizon=10_000))


# --------------------------------------------------------------------------- #
# The seasoned mirror & the placebo
# --------------------------------------------------------------------------- #
def test_seasoned_window_is_quiet_when_the_slide_is_over(planted_frame):
    """The planted slide stops after 252 days, so the seasoned window must be ~flat."""
    prices, ipos, raw = planted_frame
    se = st.seasoned_table(prices, ipos, start_day=252, end_day=756)["abn_seasoned"].dropna()
    assert len(se) > 5
    assert abs(st.one_sample_t(se)) < 2


def test_placebo_is_centred_near_zero_on_a_planted_panel(planted_frame):
    """Fake (seasoned) event dates must not reproduce the IPO-window damage."""
    prices, ipos, raw = planted_frame
    pl = st.pseudo_ipo_placebo(prices, ipos, n_draws=60, seed=931)
    assert abs(pl["mean_pct"]) < 5.0
    assert pl["n_draws"] == 60


# --------------------------------------------------------------------------- #
# Era cut / sweeps
# --------------------------------------------------------------------------- #
def test_era_cut_returns_both_halves(planted_frame):
    prices, ipos, raw = planted_frame
    tbl = st.build_table(prices, ipos, raw=raw)
    split = int(np.median(tbl["vintage"]))
    eras = st.era_cut(tbl, split_year=split)
    assert set(eras) == {"early", "late"}
    for e in eras.values():
        assert e["n_funds"] > 0 and np.isfinite(e["mean_abn_pct"])


def test_entry_sweep_reports_both_conventions(planted_frame):
    prices, ipos, raw = planted_frame
    rows = st.entry_sweep(prices, ipos, raw)
    assert [r["entry"] for r in rows] == ["offer", "close"]
    assert all(np.isfinite(r["mean_abn_pct"]) for r in rows)


def test_mapping_sweep_handles_a_single_benchmark(planted_frame):
    prices, ipos, raw = planted_frame
    one = {tk: ipos[0][3] for tk, _, _, _ in ipos}
    rows = st.mapping_sweep(prices, ipos, raw, {"per-fund": dict(), "single": one})
    assert len(rows) == 2 and all(r["n_funds"] > 0 for r in rows)


# --------------------------------------------------------------------------- #
# The tradable mirror — costs and borrow can only hurt
# --------------------------------------------------------------------------- #
def test_short_trade_is_profitable_on_a_planted_slide(planted_frame):
    prices, ipos, raw = planted_frame
    r = st.short_trade(prices, ipos, cost_bps=0.0, borrow_bps_year=0.0)
    assert r["mean_net_pct"] > 0 and r["tstat"] > 2


def test_costs_and_borrow_monotonically_reduce_the_mirror(planted_frame):
    prices, ipos, raw = planted_frame
    nets = [st.short_trade(prices, ipos, cost_bps=c, borrow_bps_year=b)["mean_net_pct"]
            for c, b in ((0.0, 0.0), (20.0, 0.0), (20.0, 500.0), (50.0, 2000.0))]
    assert nets[0] >= nets[1] >= nets[2] >= nets[3]


def test_borrow_charge_is_exactly_pro_rata(planted_frame):
    prices, ipos, raw = planted_frame
    a = st.short_trade(prices, ipos, horizon=252, cost_bps=0.0, borrow_bps_year=0.0)
    b = st.short_trade(prices, ipos, horizon=252, cost_bps=0.0, borrow_bps_year=1000.0)
    assert np.isclose(a["mean_net_pct"] - b["mean_net_pct"], 10.0, atol=1e-6)


def test_mirror_uses_the_next_close_not_the_ipo_close(planted_frame):
    prices, ipos, raw = planted_frame
    lag1 = st.short_trade(prices, ipos, lag_days=1, cost_bps=0.0, borrow_bps_year=0.0)
    lag0 = st.short_trade(prices, ipos, lag_days=0, cost_bps=0.0, borrow_bps_year=0.0)
    assert not np.isclose(lag1["mean_net_pct"], lag0["mean_net_pct"])


def test_cost_borrow_sweep_shape(planted_frame):
    prices, ipos, raw = planted_frame
    rows = st.cost_borrow_sweep(prices, ipos)
    assert len(rows) == 5 and all("per_fund" not in r for r in rows)


# --------------------------------------------------------------------------- #
# The study's spine — the detector recovers a planted hole and stays quiet on the null
# --------------------------------------------------------------------------- #
def test_detector_recovers_the_planted_slide(planted_panel):
    panel, truth = planted_panel
    d = st.synthetic_detect(panel)
    planted_pct = -truth["planted_slide"] * 100
    assert d["tstat"] < -3
    assert abs(d["mean_abn_pct"] - planted_pct) < 4.0   # ~ the planted magnitude
    assert d["share_negative"] > 0.7


def test_detector_is_quiet_on_the_null(null_panel):
    panel, _ = null_panel
    d = st.synthetic_detect(panel)
    assert abs(d["tstat"]) < 2.0


def test_detector_null_across_seeds_is_centred():
    ts = np.array([
        st.synthetic_detect(data.synthetic_panel(signal_strength=0.0, seed=931 + s)[0])["tstat"]
        for s in range(6)
    ])
    assert abs(ts.mean()) < 1.5
    assert (np.abs(ts) >= 2.0).sum() <= 1


def test_detector_scales_with_signal_strength():
    half = st.synthetic_detect(data.synthetic_panel(signal_strength=0.5, seed=931)[0])
    full = st.synthetic_detect(data.synthetic_panel(signal_strength=1.0, seed=931)[0])
    assert full["mean_abn_pct"] < half["mean_abn_pct"] < 0.0
