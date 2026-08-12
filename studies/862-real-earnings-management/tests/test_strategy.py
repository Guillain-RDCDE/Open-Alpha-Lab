"""Strategy invariants, the tercile-sort mechanics, the inference primitives, and the spine:
the REM long-short lights up *only when a lead is planted*."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from real_earn_mgmt import strategy as st  # noqa: E402


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
    assert abs(stt["t_nw"]) < 3.0            # no planted lead -> no strong HAC t


def test_calendar_ls_recovers_planted_lead(planted_panel):
    prices, ev = planted_panel
    ls = st.calendar_ls(prices, ev, staleness_days=120)
    stt = st.calendar_ls_stats(ls)
    assert stt["mean_bps"] > 0               # high-REM beats low-REM (planted sign)
    assert stt["t_nw"] > 2.0                 # and it clears the bar


def test_calendar_ls_turnover_in_unit_range(planted_panel):
    prices, ev = planted_panel
    ls = st.calendar_ls(prices, ev, staleness_days=120)
    assert (ls["turnover"] >= 0).all() and (ls["turnover"] <= 1.0).all()


def test_costs_reduce_absolute_edge(planted_panel):
    prices, ev = planted_panel
    ls = st.calendar_ls(prices, ev, staleness_days=120)
    gross = st.calendar_ls_stats(ls)["mean_bps"]
    net = st.calendar_ls_net(ls, cost_bps=20.0, borrow_bps_ann=100.0)["net_mean_bps"]
    assert abs(net) < abs(gross)             # friction shrinks the magnitude of the edge


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


# --------------------------------------------------------------------------- #
# leads-operating axis (gross-margin reversal)
# --------------------------------------------------------------------------- #
def test_leads_operating_planted_reversal():
    import pandas as pd
    rng = np.random.default_rng(5)
    n = 400
    rem = rng.normal(0, 1.0, n)
    gm = rng.normal(0.4, 0.05, n)
    # high REM (overproduction) -> next-quarter gross margin falls back: negative Δgm
    next_gm = gm - 0.03 * rem + rng.normal(0, 0.01, n)
    ev = pd.DataFrame({"rem": rem, "gm": gm, "next_gm": next_gm})
    out = st.leads_operating(ev)
    assert out["slope"] < 0 and out["t"] < -5.0
    assert out["spread"] < 0                  # top-REM tercile has the worse Δgm


def test_leads_operating_no_relation():
    import pandas as pd
    rng = np.random.default_rng(6)
    n = 400
    ev = pd.DataFrame({"rem": rng.normal(0, 1, n),
                       "gm": rng.normal(0.4, 0.05, n),
                       "next_gm": rng.normal(0.4, 0.05, n)})
    out = st.leads_operating(ev)
    assert abs(out["t"]) < 3.0


# --------------------------------------------------------------------------- #
# synthetic detector wrapper
# --------------------------------------------------------------------------- #
def test_synthetic_detect_spine(null_panel, planted_panel):
    p0, e0 = null_panel
    p1, e1 = planted_panel
    assert abs(st.synthetic_detect(p0, e0)["t_nw"]) < 3.0
    assert st.synthetic_detect(p1, e1)["t_nw"] > 2.0
