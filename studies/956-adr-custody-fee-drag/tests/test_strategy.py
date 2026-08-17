"""Estimator logic, break handling, and the study's spine — all offline/synthetic.

The spine: on a world with a **planted** custody fee the trend estimator recovers the
planted annual drag to within a basis point or two, is unmoved by a planted ADS-ratio
step, reports a flat price-ratio placebo, and stays quiet when the fee is switched off.
The coverage screen throws out a home leg whose adjusted close carries no dividends —
the London defect that would otherwise manufacture a 5 %/yr "fee".
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from adr_drag import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_newey_west_recovers_positive_mean():
    rng = np.random.default_rng(1)
    x = 0.001 + rng.normal(0, 0.01, 5000)
    assert st.newey_west_t(x, lags=10) > 3
    assert abs(st.newey_west_t(rng.normal(0, 0.01, 5000), lags=10)) < 3


def test_nw_ols_recovers_a_known_slope():
    rng = np.random.default_rng(0)
    n = 3000
    t = np.arange(n, dtype=float)
    y = 2.0 + 0.003 * t + rng.normal(0, 1.0, n)
    X = np.column_stack([np.ones(n), t])
    beta, se = st.nw_ols(X, y, lags=20)
    assert abs(beta[1] - 0.003) < 5e-4
    assert (se > 0).all()


def test_block_bootstrap_ci_brackets_the_mean():
    rng = np.random.default_rng(2)
    x = 0.5 + rng.normal(0, 1.0, 4000)
    lo, hi = st.block_bootstrap_ci(x, n_boot=400, block=21, seed=956)
    assert lo < 0.5 < hi


def test_choose_matches_binomial_coefficients():
    assert st._choose(10, 0) == pytest.approx(1.0)
    assert st._choose(10, 5) == pytest.approx(252.0)
    assert st._choose(10, 10) == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
# Log ratios and the level-shift detector
# --------------------------------------------------------------------------- #
def test_log_ratio_legs(planted):
    df, _ = planted
    x = st.log_ratio(df, "tr")
    assert len(x) == len(df) and np.isfinite(x.to_numpy()).all()
    with pytest.raises(ValueError):
        st.log_ratio(df, "nope")


def test_level_shift_ignores_a_one_day_spike():
    n = 800
    x = pd.Series(np.zeros(n), index=pd.bdate_range("2010-01-04", periods=n))
    x.iloc[400] = 0.5                      # a single non-synchronous close, reverts at once
    assert st.n_breaks(x, break_thresh=0.10) == 0


def test_level_shift_catches_a_permanent_step():
    n = 800
    x = pd.Series(np.zeros(n), index=pd.bdate_range("2010-01-04", periods=n))
    x.iloc[400:] = 0.5                     # an ADS-ratio change
    bp = st.break_points(x, break_thresh=0.10)
    assert len(bp) == 1 and abs(bp[0] - 400) <= 12


def test_segment_index_drops_short_stubs():
    n = 800
    x = pd.Series(np.zeros(n), index=pd.bdate_range("2010-01-04", periods=n))
    x.iloc[700:] = 0.5
    seg = st.segment_index(x, break_thresh=0.10, min_days=250)
    assert (seg == -1).sum() >= 90         # the 100-day tail stub is excluded
    assert set(np.unique(seg)) <= {-1, 0, 1}


def test_break_detector_finds_the_planted_ads_ratio_change(broken_pair):
    df, _ = broken_pair
    x = st.log_ratio(df, "px")
    bp = st.break_points(x, break_thresh=0.10)
    assert len(bp) == 1 and abs(bp[0] - len(df) // 2) <= 12


# --------------------------------------------------------------------------- #
# The trend estimator
# --------------------------------------------------------------------------- #
def test_trend_drag_recovers_a_planted_linear_drift():
    n = 4000
    t = np.arange(n) / st.TRADING_DAYS
    idx = pd.bdate_range("2005-01-03", periods=n)
    rng = np.random.default_rng(3)
    x = pd.Series(-0.004 * t + rng.normal(0, 0.005, n), index=idx)
    out = st.trend_drag(x, break_thresh=1.0)
    assert abs(out["drag"] - 0.004) < 5e-4
    assert out["t"] > 2 and out["n_segments"] == 1


def test_trend_drag_is_immune_to_a_level_step():
    n = 4000
    t = np.arange(n) / st.TRADING_DAYS
    idx = pd.bdate_range("2005-01-03", periods=n)
    x = pd.Series(-0.004 * t, index=idx)
    x.iloc[n // 2:] += 0.7                 # a big ADS-ratio step
    out = st.trend_drag(x, break_thresh=0.10)
    assert out["n_segments"] == 2
    assert abs(out["drag"] - 0.004) < 5e-4


def test_trend_drag_short_sample_returns_nan():
    idx = pd.bdate_range("2020-01-02", periods=50)
    out = st.trend_drag(pd.Series(np.zeros(50), index=idx))
    assert np.isnan(out["drag"])


def test_bootstrap_ci_brackets_the_point_estimate(planted):
    df, _ = planted
    _, _, gap = st.income_series(df)
    ci = st.bootstrap_drag_ci(gap, break_thresh=1.0, n_boot=200, seed=956)
    assert ci["ci_low"] <= ci["drag"] <= ci["ci_high"]
    assert ci["n_boot"] > 0


# --------------------------------------------------------------------------- #
# The coverage screen — the London defect
# --------------------------------------------------------------------------- #
def test_coverage_screen_passes_a_healthy_pair(planted):
    df, _ = planted
    scr = st.coverage_screen(df)
    assert scr["pass"] is True
    assert 0.02 < scr["local_yield"] < 0.05
    assert 0.5 < scr["yield_ratio"] < 1.0


def test_coverage_screen_rejects_a_split_only_home_leg(no_dividend_pair):
    df, _ = no_dividend_pair
    scr = st.coverage_screen(df)
    assert scr["pass"] is False
    assert scr["local_yield"] < 1e-9


def test_screen_frames_reports_every_name(planted, no_dividend_pair):
    frames = {"GOOD": planted[0], "LSE": no_dividend_pair[0]}
    kept, report = st.screen_frames(frames)
    assert set(report.index) == {"GOOD", "LSE"}
    assert set(kept) == {"GOOD"}


def test_screen_does_not_depend_on_the_adr_leg(planted):
    """The gate reads the HOME leg only, so it cannot select on the answer."""
    df, _ = planted
    hacked = df.copy()
    hacked["adr_tr"] = hacked["adr_px"]        # destroy the ADR's income leg entirely
    assert st.coverage_screen(hacked)["pass"] == st.coverage_screen(df)["pass"]


# --------------------------------------------------------------------------- #
# The study's spine — recovery on the planted world, silence on the null
# --------------------------------------------------------------------------- #
def test_planted_income_gap_is_recovered(planted):
    df, truth = planted
    out = st.decompose_pair(df, wht=truth["wht"])
    assert abs(out["income_gap"] - truth["planted_gap_per_year"]) < 5e-4
    assert out["income_gap_t"] > 2


def test_planted_custody_residual_is_recovered(planted):
    """With the *correct* withholding assumption the residual is the planted fee."""
    df, truth = planted
    out = st.decompose_pair(df, wht=truth["wht"])
    assert abs(out["custody"] - truth["custody_drag_per_year"]) < 8e-4


def test_price_placebo_is_flat_on_the_planted_world(planted):
    df, _ = planted
    out = st.decompose_pair(df, wht=0.15)
    assert abs(out["price_drift"]) < 2e-3


def test_null_world_shows_no_drag(null_pair):
    df, _ = null_pair
    out = st.decompose_pair(df, wht=0.0)
    assert abs(out["income_gap"]) < 5e-4


def test_pooled_recovers_the_planted_drag(planted_panel):
    frames, truth = planted_panel
    whts = {k: truth["per_name"][k]["wht"] for k in frames}
    d = st.synthetic_detect(frames, whts)
    assert abs(d["income_gap"]["mean"] - truth["planted_gap_per_year"]) < 5e-4
    assert d["income_gap"]["share_positive"] == 1.0


def test_pooled_null_is_quiet_across_seeds():
    means = []
    for s in range(6):
        frames, truth = data.synthetic_panel(n_names=6, n_years=10, drag_bps_per_year=25.0,
                                             signal_strength=0.0, seed=956 + 11 * s)
        whts = {k: 0.0 for k in frames}
        means.append(st.synthetic_detect(frames, whts)["income_gap"]["mean"])
    means = np.array(means)
    assert abs(means.mean()) < 5e-4
    assert (np.abs(means) >= 1e-3).sum() <= 1


def test_drag_survives_a_planted_ratio_break(broken_pair):
    df, truth = broken_pair
    out = st.decompose_pair(df, wht=truth["wht"], break_thresh=0.10)
    assert out["n_segments"] == 2
    assert abs(out["drag_total"] - truth["planted_gap_per_year"]) < 3e-3


# --------------------------------------------------------------------------- #
# Panel plumbing, sweeps and the traded leg
# --------------------------------------------------------------------------- #
def test_panel_table_and_pooled(planted_panel):
    frames, truth = planted_panel
    whts = {k: truth["per_name"][k]["wht"] for k in frames}
    tbl = st.panel_table(frames, whts)
    assert len(tbl) == len(frames)
    for c in ("drag_total", "income_gap", "custody", "custody_cents"):
        assert c in tbl.columns
    p = st.pooled(tbl, "income_gap")
    assert p["n"] == len(frames) and np.isfinite(p["t"])


def test_pooled_handles_a_degenerate_subset(planted_panel):
    frames, truth = planted_panel
    whts = {k: truth["per_name"][k]["wht"] for k in frames}
    tbl = st.panel_table(frames, whts)
    p = st.pooled(tbl, "income_gap", subset=[tbl.index[0]])
    assert np.isnan(p["mean"]) or p["n"] == 1


def test_name_bootstrap_ci_and_sign_test(planted_panel):
    frames, truth = planted_panel
    whts = {k: truth["per_name"][k]["wht"] for k in frames}
    tbl = st.panel_table(frames, whts)
    bs = st.name_bootstrap(tbl, "income_gap", n_boot=1000)
    assert bs["ci_low"] <= bs["mean"] <= bs["ci_high"]
    assert bs["n_positive"] == bs["n"]
    assert bs["sign_p"] < 0.01


def test_leave_one_out_shape(planted_panel):
    frames, truth = planted_panel
    whts = {k: truth["per_name"][k]["wht"] for k in frames}
    tbl = st.panel_table(frames, whts)
    loo = st.leave_one_out(tbl, "income_gap")
    assert len(loo) == len(tbl)
    assert (loo["n"] == len(tbl) - 1).all()


def _small_panel():
    frames, truth = data.synthetic_panel(n_names=4, n_years=10, drag_bps_per_year=25.0,
                                         signal_strength=1.0, seed=956)
    return frames, {k: truth["per_name"][k]["wht"] for k in frames}


def test_withholding_sweep_moves_the_residual_monotonically():
    frames, whts = _small_panel()
    rows = st.withholding_sweep(frames, whts, scales=(0.0, 0.5, 1.0, 1.5))
    means = [r["custody_mean"] for r in rows]
    assert means[0] > means[1] > means[2] > means[3]


def test_break_threshold_sweep_shape():
    frames, whts = _small_panel()
    rows = st.break_threshold_sweep(frames, whts, thresholds=(0.10, 0.25))
    assert len(rows) == 2 and all(np.isfinite(r["drag_mean"]) for r in rows)


def test_era_cut_returns_both_halves():
    frames, whts = _small_panel()
    idx = list(frames.values())[0].index
    eras = st.era_cut(frames, whts, split=str(idx[len(idx) // 2].date()), min_days=250)
    assert eras["early"] is not None and eras["late"] is not None
    for e in eras.values():
        assert np.isfinite(e["income_gap"]["mean"])


def test_switch_race_costs_and_lag(planted_panel):
    frames, truth = planted_panel
    idx = list(frames.values())[0].index
    cash = pd.Series(100 * (1.0 + 0.02 / 252) ** np.arange(len(idx)), index=idx)
    free = st.switch_race(frames, cash, fx_cost_bps=0.0)
    costed = st.switch_race(frames, cash, fx_cost_bps=50.0, foreign_custody_bps_per_year=40.0)
    assert free["ann_diff"] > costed["ann_diff"]      # friction only ever subtracts
    assert free["n_days"] > 0 and np.isfinite(free["t_diff"])


def test_switch_cost_sweep_is_monotone_in_friction(planted_panel):
    frames, truth = planted_panel
    idx = list(frames.values())[0].index
    cash = pd.Series(100 * (1.0 + 0.02 / 252) ** np.arange(len(idx)), index=idx)
    rows = st.switch_cost_sweep(frames, cash,
                                grid=((0.0, 0.0), (30.0, 10.0), (60.0, 40.0)))
    diffs = [r["ann_diff"] for r in rows]
    assert diffs[0] > diffs[1] > diffs[2]
