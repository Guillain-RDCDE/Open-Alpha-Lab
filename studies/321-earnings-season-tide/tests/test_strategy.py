"""The calendar contrast, its HAC inference, the overlay engine, and the study's two
spines: (1) the contrast clears the bar only when an in-window tide is really planted, and
(2) the in-window overlay does not out-Sharpe buy-and-hold on a null tape."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from earnings_season_tide import data  # noqa: E402
from earnings_season_tide import strategy as st  # noqa: E402


# ---- the contrast & its inference -----------------------------------------
def test_contrast_keys_and_counts(null_tape):
    bars, truth = null_tape
    c = st.calendar_contrast(bars)
    assert set(c) >= {"n_in", "n_out", "mean_in_bps", "mean_out_bps",
                      "diff_bps", "tstat_diff", "ann_in_pct"}
    # in + out covers every return bar (one bar lost to the diff).
    assert c["n_in"] + c["n_out"] == len(bars) - 1
    assert c["n_in"] > 0 and c["n_out"] > 0


def test_diff_is_in_minus_out(tide_tape):
    bars, _ = tide_tape
    c = st.calendar_contrast(bars)
    assert np.isclose(c["diff_bps"], c["mean_in_bps"] - c["mean_out_bps"], atol=1e-6)


def test_contrast_significant_only_with_tide(null_tape, tide_tape):
    """Spine #1: a planted in-window tide clears the bar; a pure random walk does not."""
    c_null = st.calendar_contrast(null_tape[0])
    c_tide = st.calendar_contrast(tide_tape[0])
    assert abs(c_null["tstat_diff"]) < 2.0     # correct null
    assert c_tide["tstat_diff"] > 2.0          # genuine tide recovered
    assert c_tide["diff_bps"] > c_null["diff_bps"]


def test_hac_tstat_finite_and_signed(tide_tape):
    bars, _ = tide_tape
    r = st.log_returns(bars)
    mask = data.in_earnings_window(r.index).reindex(r.index).to_numpy()
    t = st._hac_diff_tstat(r.to_numpy(), mask)
    assert np.isfinite(t) and t > 0


def test_bootstrap_ci_brackets_point_estimate(tide_tape):
    bars, _ = tide_tape
    c = st.calendar_contrast(bars)
    lo, hi = st.block_bootstrap_ci(bars, n_boot=500, seed=1)
    assert lo < c["diff_bps"] < hi
    assert lo < hi


def test_bootstrap_null_ci_includes_zero(null_tape):
    bars, _ = null_tape
    lo, hi = st.block_bootstrap_ci(bars, n_boot=500, seed=1)
    assert lo < 0.0 < hi


# ---- the overlay engine ----------------------------------------------------
def test_overlay_columns_and_in_window(null_tape):
    bars, _ = null_tape
    ov = st.overlay_returns(bars, cost_bps=1.0)
    assert list(ov.columns) == ["ret_gross", "ret_net", "in_window", "traded"]
    # out-of-window gross return is the cash rate (0 by default).
    assert np.allclose(ov.loc[~ov["in_window"], "ret_gross"].to_numpy(), 0.0)


def test_overlay_net_is_gross_minus_cost(null_tape):
    bars, _ = null_tape
    ov = st.overlay_returns(bars, cost_bps=3.0)
    expected = ov["ret_gross"] - ov["traded"].astype(float) * 3.0e-4
    assert np.allclose(ov["ret_net"], expected)


def test_overlay_cost_only_on_trade_days(null_tape):
    bars, _ = null_tape
    ov = st.overlay_returns(bars, cost_bps=5.0)
    not_traded = ~ov["traded"]
    assert np.allclose(ov.loc[not_traded, "ret_net"].to_numpy(),
                       ov.loc[not_traded, "ret_gross"].to_numpy())


def test_cost_monotonically_lowers_overlay_mean(null_tape):
    bars, _ = null_tape
    means = [st.overlay_returns(bars, cost_bps=c)["ret_net"].mean()
             for c in (0.0, 1.0, 2.0)]
    assert means[0] > means[1] > means[2]


def test_time_in_market_is_small(null_tape):
    """Four ~2-week windows a year is roughly a fifth of trading days."""
    bars, _ = null_tape
    race = st.race_vs_buyhold(bars, cost_bps=1.0)
    assert 0.10 < race["time_in_market"] < 0.30


def test_overlay_does_not_beat_buyhold_on_null(null_tape):
    """Spine #2: with no tide, the in-window overlay cannot out-Sharpe buy-and-hold."""
    bars, _ = null_tape
    race = st.race_vs_buyhold(bars, cost_bps=1.0)
    # On a driftless random walk both are ~0; the overlay must not manufacture an edge.
    assert race["overlay_sharpe_net"] <= race["buyhold_sharpe"] + 0.25


def test_turnover_is_positive(null_tape):
    bars, _ = null_tape
    race = st.race_vs_buyhold(bars, cost_bps=1.0)
    assert race["turnover_per_yr"] > 0


# ---- per-window breakdown --------------------------------------------------
def test_per_window_has_four_rows(tide_tape):
    bars, _ = tide_tape
    pw = st.per_window_means(bars)
    assert len(pw) == 4
    assert set(pw.columns) == {"window", "n_days", "mean_bps", "tstat"}
    assert (pw["n_days"] > 0).all()
