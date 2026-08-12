"""Strategy invariants, the tercile-sort mechanics, the inference primitives, and the spine:
the ROIC long-short lights up *only when a premium is planted*."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from roic_premium import data  # noqa: E402
from roic_premium import strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# inference primitives
# --------------------------------------------------------------------------- #
def test_one_sample_t_zero_mean():
    rng = np.random.default_rng(0)
    assert abs(st.one_sample_t(rng.normal(0, 1, 5000))) < 3.0


def test_one_sample_t_positive_mean():
    rng = np.random.default_rng(1)
    assert st.one_sample_t(rng.normal(0.5, 1, 5000)) > 10.0


def test_welch_t_symmetry():
    a = np.array([1.0, 2, 3, 4, 5]); b = np.array([2.0, 3, 4, 5, 6])
    assert st.welch_t(a, b) == pytest.approx(-st.welch_t(b, a))


def test_newey_west_reduces_to_iid_at_zero_lags():
    rng = np.random.default_rng(2)
    x = rng.normal(0.1, 1, 400)
    t_nw0 = st.newey_west_t(x, lags=0)
    se = np.std(x, ddof=0) / np.sqrt(len(x))
    assert t_nw0 == pytest.approx(x.mean() / se, rel=1e-6)


def test_newey_west_widens_se_under_positive_autocorr():
    rng = np.random.default_rng(3)
    e = rng.normal(0, 1, 2000)
    x = np.empty(2000); x[0] = e[0]
    for i in range(1, 2000):
        x[i] = 0.6 * x[i - 1] + e[i]
    x += 0.1
    assert abs(st.newey_west_t(x, lags=10)) < abs(st.one_sample_t(x))


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(60, 100)
    assert lo < 0.60 < hi


def test_bucketize_equal_frequency():
    b = st._bucketize(np.arange(30.0), 3)
    assert set(np.bincount(b)) == {10}
    assert b[0] == 0 and b[-1] == 2


# --------------------------------------------------------------------------- #
# calendar-time long-short spine
# --------------------------------------------------------------------------- #
def test_calendar_ls_columns(null_panel):
    prices, ev = null_panel
    ls = st.calendar_ls(prices, ev, staleness_days=120)
    for c in ("ls", "long", "short", "n", "turnover"):
        assert c in ls.columns
    assert len(ls) > 10


def test_calendar_ls_null_is_quiet(null_panel):
    prices, ev = null_panel
    ls = st.calendar_ls(prices, ev, staleness_days=120)
    assert abs(st.calendar_ls_stats(ls)["t_nw"]) < 3.0


def test_calendar_ls_recovers_planted_premium(planted_panel):
    prices, ev = planted_panel
    ls = st.calendar_ls(prices, ev, staleness_days=120)
    stt = st.calendar_ls_stats(ls)
    assert stt["mean_bps"] > 0            # high-ROIC beats low-ROIC
    assert stt["t_nw"] > 2.0              # and it clears the bar


def test_calendar_ls_turnover_in_unit_range(planted_panel):
    prices, ev = planted_panel
    ls = st.calendar_ls(prices, ev, staleness_days=120)
    assert (ls["turnover"] >= 0).all() and (ls["turnover"] <= 1.0).all()


def test_costs_reduce_edge(planted_panel):
    prices, ev = planted_panel
    ls = st.calendar_ls(prices, ev, staleness_days=120)
    gross = st.calendar_ls_stats(ls)["mean_bps"]
    net = st.calendar_ls_net(ls, cost_bps=20.0, borrow_bps_ann=100.0)["net_mean_bps"]
    assert net < gross


# --------------------------------------------------------------------------- #
# pooled event drift + placebo
# --------------------------------------------------------------------------- #
def test_event_summary_null_placebo_not_extreme(null_panel):
    prices, ev = null_panel
    s = st.event_summary(prices, ev, horizon=63, n_draws=1000)
    assert 0.02 < s["p_placebo"] < 0.98


def test_event_summary_planted_significant(planted_panel):
    prices, ev = planted_panel
    s = st.event_summary(prices, ev, horizon=63, n_draws=1000)
    assert s["ls_mean"] > 0 and s["t"] > 2.0
    assert s["p_placebo"] < 0.05


def test_placebo_reproducible(planted_panel):
    prices, ev = planted_panel
    fr = st.event_drift_frame(prices, ev, horizon=63)
    a = st.placebo_pvalue(fr, n_draws=500, seed=7)["draws"]
    b = st.placebo_pvalue(fr, n_draws=500, seed=7)["draws"]
    assert np.allclose(a, b)


def test_bucket_means_monotone_when_planted(planted_panel):
    prices, ev = planted_panel
    fr = st.event_drift_frame(prices, ev, horizon=63)
    m = st.bucket_means(fr, 3)
    assert m[0] < m[2]                     # low ROIC under-returns high ROIC


# --------------------------------------------------------------------------- #
# 3rd axis — ROIC vs ROE vs GP contrast
# --------------------------------------------------------------------------- #
def test_contrast_reports_all_signals(planted_panel):
    prices, ev = planted_panel
    c = st.contrast(prices, ev, staleness_days=120)
    for k in ("roic", "roic_chg", "roe", "gp", "roic_roe_rank_corr"):
        assert k in c
    # ROIC drives returns in the planted world; its long-short clears the bar
    assert c["roic"]["t_nw"] > 2.0
    # ROIC and (leverage-scaled) ROE are strongly rank-correlated in the synthetic world
    assert c["roic_roe_rank_corr"] > 0.5


# --------------------------------------------------------------------------- #
# synthetic detector wrapper
# --------------------------------------------------------------------------- #
def test_synthetic_detect_spine(null_panel, planted_panel):
    p0, e0 = null_panel
    p1, e1 = planted_panel
    assert abs(st.synthetic_detect(p0, e0)["t_nw"]) < 3.0
    assert st.synthetic_detect(p1, e1)["t_nw"] > 2.0


# --------------------------------------------------------------------------- #
# real-cache smoke — skipped when the git-ignored cache is absent (offline CI)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not data.have_real(), reason="real cache absent offline CI")
def test_real_calendar_ls_runs():
    px, ev = data.load_real()
    ls = st.calendar_ls(px, ev, signal_col="roic", n_buckets=3, min_names=6, staleness_days=200)
    s = st.calendar_ls_stats(ls)
    assert s["n_months"] > 24
    assert np.isfinite(s["t_nw"])
