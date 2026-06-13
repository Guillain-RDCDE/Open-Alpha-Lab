"""Signals, ledger invariants, random control, and the study's spine:
follow-morning beats random only when continuation is genuinely planted."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from power_hour import data, strategy as st  # noqa: E402


# ---- correlation diagnostic ----------------------------------------------
def test_corr_null_near_zero(null_panel):
    panel, _ = null_panel
    res = st.morning_last_corr(panel)
    assert abs(res["corr"]) < 0.12
    assert abs(res["tstat"]) < 3.0  # not significant on a null tape


def test_corr_trending_positive(trending_panel):
    panel, _ = trending_panel
    res = st.morning_last_corr(panel)
    assert res["corr"] > 0.1
    assert res["tstat"] > 1.0


def test_corr_reversal_negative(reversal_panel):
    panel, _ = reversal_panel
    res = st.morning_last_corr(panel)
    assert res["corr"] < -0.1


# ---- ledger invariants ---------------------------------------------------
def test_ledger_columns(null_panel):
    panel, _ = null_panel
    led = st.build_ledger(panel, arm="follow", cost_bps=1.0)
    assert list(led.columns) == ["date", "morning_ret", "last_ret", "dir", "ret_gross", "ret_net"]


def test_follow_direction_matches_morning_sign(null_panel):
    panel, _ = null_panel
    led = st.build_ledger(panel, arm="follow", cost_bps=0.0)
    m = led["morning_ret"].to_numpy()
    d = led["dir"].to_numpy()
    # Direction must equal sign(morning_ret) for non-zero mornings
    mask = m != 0
    assert (d[mask] == np.sign(m[mask]).astype(int)).all()


def test_fade_direction_opposite_morning(null_panel):
    panel, _ = null_panel
    led = st.build_ledger(panel, arm="fade", cost_bps=0.0)
    m = led["morning_ret"].to_numpy()
    d = led["dir"].to_numpy()
    mask = m != 0
    assert (d[mask] == -np.sign(m[mask]).astype(int)).all()


def test_random_arm_balanced(null_panel):
    panel, _ = null_panel
    led = st.build_ledger(panel, arm="random", seed=1)
    dirs = led["dir"].to_numpy()
    assert set(np.unique(dirs)) == {-1, 1}
    assert 0.35 < (dirs == 1).mean() < 0.65


def test_net_equals_gross_minus_cost(null_panel):
    panel, _ = null_panel
    cost = 2.5
    led = st.build_ledger(panel, arm="follow", cost_bps=cost)
    assert np.allclose(led["ret_net"], led["ret_gross"] - cost * 1e-4)


def test_gross_equals_dir_times_last(null_panel):
    panel, _ = null_panel
    led = st.build_ledger(panel, arm="follow", cost_bps=0.0)
    expected = led["dir"].to_numpy() * led["last_ret"].to_numpy()
    assert np.allclose(led["ret_gross"].to_numpy(), expected)


def test_cost_monotonically_lowers_mean(null_panel):
    panel, _ = null_panel
    means = [
        st.summarize(st.build_ledger(panel, arm="follow", cost_bps=c), "ret_net")["mean_bps"]
        for c in (0.0, 1.0, 2.0)
    ]
    assert means[0] > means[1] > means[2]


# ---- the spine: signal only where continuation is planted ----------------
def test_follow_beats_random_only_with_continuation(null_panel, trending_panel):
    """The spine: follow-morning edges the random coin only when we plant continuation."""
    def edge(panel):
        follow = st.summarize(st.build_ledger(panel, arm="follow", cost_bps=0), "ret_gross")
        rand = st.summarize(st.build_ledger(panel, arm="random", cost_bps=0, seed=1), "ret_gross")
        return follow["mean_bps"] - rand["mean_bps"]

    assert edge(trending_panel[0]) > 0.0       # planted signal → follow wins
    assert edge(null_panel[0]) < 1.0           # no signal → no real edge


def test_fade_wins_under_reversal(reversal_panel):
    """With planted reversal, the fade arm (against-morning) should have positive mean."""
    panel, _ = reversal_panel
    fade = st.summarize(st.build_ledger(panel, arm="fade", cost_bps=0), "ret_gross")
    assert fade["mean_bps"] > 0.0


def test_summarize_tstat_finite(null_panel):
    panel, _ = null_panel
    s = st.summarize(st.build_ledger(panel, arm="follow", cost_bps=1.0))
    assert np.isfinite(s["tstat"])
    assert np.isfinite(s["mean_bps"])
