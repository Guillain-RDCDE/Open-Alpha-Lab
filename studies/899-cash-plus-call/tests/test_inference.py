"""Offline tests for the inference primitives — HAC t, spanning alpha, bootstrap, Wilson, capture."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from cash_call import strategy as st  # noqa: E402


def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=10) - st.one_sample_t(x)) < 0.6


def test_spanning_alpha_recovers_zero_when_pure_beta():
    # y = beta*x + iid noise, no intercept -> alpha ~ 0, |t_alpha| small, beta recovered.
    rng = np.random.default_rng(2)
    x = rng.normal(0.0003, 0.01, 5000)
    y = 0.7 * x + rng.normal(0.0, 0.002, 5000)
    out = st.spanning_alpha(y, x)
    assert abs(out["beta"] - 0.7) < 0.05
    assert abs(out["t_alpha"]) < 2.0


def test_spanning_alpha_detects_planted_intercept():
    rng = np.random.default_rng(3)
    x = rng.normal(0.0003, 0.01, 5000)
    y = 1.0 * x + 0.0004 + rng.normal(0.0, 0.002, 5000)   # +4 bps/day alpha
    out = st.spanning_alpha(y, x)
    assert out["t_alpha"] > 4.0
    assert out["alpha_ann"] > 0.05


def test_bootstrap_ci_brackets_point(calm_world):
    frame, _ = calm_world
    r = st.race(frame)
    bs = st.bootstrap_sharpe_diff(r["tt_net"], r["bh_net"], r["cash_net"], n_boot=500)
    lo, hi = bs["ci95"]
    assert lo <= bs["point"] <= hi
    assert 0.0 <= bs["frac_a_wins"] <= 1.0


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(60, 100)
    assert lo < 0.60 < hi


def test_capture_symmetric_for_linear_book():
    # A constant-mix (linear) book captures up and down moves symmetrically -> asymmetry ~ 0.
    rng = np.random.default_rng(5)
    spy = rng.normal(0.0004, 0.011, 4000)
    cash = np.full(4000, 0.03 / 252)
    lin = st.constant_mix(spy, cash, 0.5)
    cap = st.capture(lin, spy)
    assert abs(cap["asymmetry"]) < 0.05                  # linear -> up-capture ~ down-capture


def test_max_drawdown_zero_on_monotone_series():
    up = pd.Series(np.full(500, 0.001))     # never draws down
    s = st.stats(up)
    assert s["max_dd"] > -1e-9


def test_sortino_finite_and_positive_on_positive_drift():
    rng = np.random.default_rng(7)
    r = rng.normal(0.0006, 0.01, 3000)
    s = st.stats(pd.Series(r))
    assert np.isfinite(s["sortino"]) and s["sortino"] > 0


def test_calendar_year_returns_compound():
    idx = pd.bdate_range("2020-01-01", "2021-12-31", name="Date")
    r = pd.Series(np.full(len(idx), 0.0004), index=idx)
    cy = st.calendar_year_returns(r)
    assert set(cy.index) == {2020, 2021}
    assert (cy > 0).all()


def test_calendar_year_drawdowns_are_nonpositive():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2019-01-01", "2021-12-31", name="Date")
    r = pd.Series(rng.normal(0.0002, 0.01, len(idx)), index=idx)
    dd = st.calendar_year_drawdowns(r)
    assert (dd <= 0).all()
    assert set(dd.index) == {2019, 2020, 2021}
