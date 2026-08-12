"""Strategy invariants, the tercile-sort mechanics, the inference primitives, and the spine:
the accrual-quality long-short lights up *only when a quality->return relation is planted*."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from accrual_quality import strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# inference primitives
# --------------------------------------------------------------------------- #
def test_one_sample_t_zero_mean():
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, 5000)
    assert abs(st.one_sample_t(x)) < 3.0


def test_one_sample_t_positive_mean():
    rng = np.random.default_rng(1)
    x = rng.normal(0.5, 1, 5000)
    assert st.one_sample_t(x) > 10.0


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
    x = np.arange(30.0)
    b = st._bucketize(x, 3)
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
    stt = st.calendar_ls_stats(ls)
    assert abs(stt["t_nw"]) < 3.0


def test_calendar_ls_recovers_planted_relation(planted_panel):
    prices, ev = planted_panel
    ls = st.calendar_ls(prices, ev, staleness_days=120)
    stt = st.calendar_ls_stats(ls)
    assert stt["mean_bps"] > 0            # high-quality beats low-quality
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
    assert 0.01 < s["p_placebo"] < 0.99


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


# --------------------------------------------------------------------------- #
# persistence axis
# --------------------------------------------------------------------------- #
def test_persistence_planted_gap():
    import pandas as pd
    rng = np.random.default_rng(5)
    n = 900
    aq = rng.uniform(0.0, 0.05, n)                 # residual vol (poor quality = high)
    roa = rng.normal(0.03, 0.02, n)
    # good-quality (low aq) earnings persist; poor-quality earnings are near-random walk down
    persist = np.where(aq < np.median(aq), 0.9, 0.2)
    roa_next = persist * roa + rng.normal(0, 0.01, n)
    ev = pd.DataFrame({"aq_vol": aq, "roa": roa, "roa_next": roa_next,
                       "earn_vol": aq})
    out = st.persistence_by_quality(ev)
    assert out["good_slope"] > out["poor_slope"]   # good quality persists more
    assert out["slope_gap"] > 0


def test_persistence_thin_returns_nan():
    import pandas as pd
    ev = pd.DataFrame({"aq_vol": [0.1, 0.2], "roa": [0.01, 0.02],
                       "roa_next": [0.01, 0.02], "earn_vol": [0.1, 0.2]})
    out = st.persistence_by_quality(ev)
    assert np.isnan(out["good_slope"])


# --------------------------------------------------------------------------- #
# synthetic detector wrapper
# --------------------------------------------------------------------------- #
def test_synthetic_detect_spine(null_panel, planted_panel):
    p0, e0 = null_panel
    p1, e1 = planted_panel
    assert abs(st.synthetic_detect(p0, e0)["t_nw"]) < 3.0
    assert st.synthetic_detect(p1, e1)["t_nw"] > 2.0
