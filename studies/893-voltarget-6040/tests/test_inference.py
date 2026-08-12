"""Offline tests for the inference primitives — HAC t, spanning alpha, bootstrap, Wilson."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from vt6040 import strategy as st  # noqa: E402


def test_newey_west_matches_one_sample_on_iid():
    rng = np.random.default_rng(0)
    x = rng.normal(0.001, 0.01, 4000)
    assert abs(st.newey_west_t(x, lags=10) - st.one_sample_t(x)) < 0.6


def test_spanning_alpha_recovers_zero_when_pure_beta():
    # y = beta*x + iid noise, no intercept -> alpha ~ 0, |t_alpha| small, beta recovered.
    rng = np.random.default_rng(2)
    x = rng.normal(0.0003, 0.01, 5000)
    y = 1.3 * x + rng.normal(0.0, 0.002, 5000)
    out = st.spanning_alpha(y, x)
    assert abs(out["beta"] - 1.3) < 0.05
    assert abs(out["t_alpha"]) < 2.0


def test_spanning_alpha_detects_planted_intercept():
    rng = np.random.default_rng(3)
    x = rng.normal(0.0003, 0.01, 5000)
    y = 1.0 * x + 0.0004 + rng.normal(0.0, 0.002, 5000)   # +4 bps/day alpha
    out = st.spanning_alpha(y, x)
    assert out["t_alpha"] > 4.0
    assert out["alpha_ann"] > 0.05


def test_bootstrap_ci_brackets_point(edge_world):
    frame, _ = edge_world
    ret = st.to_returns(frame)
    r = st.race(ret)
    bs = st.bootstrap_sharpe_diff(r["vt_net"], r["static_net"], r["cash_net"], n_boot=500)
    lo, hi = bs["ci95"]
    assert lo <= bs["point"] <= hi
    assert 0.0 <= bs["frac_a_wins"] <= 1.0


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(60, 100)
    assert lo < 0.60 < hi


def test_max_drawdown_zero_on_monotone_series():
    up = pd.Series(np.full(500, 0.001))     # never draws down
    s = st.stats(up)
    assert s["max_dd"] > -1e-9


def test_calendar_year_returns_compound():
    idx = pd.bdate_range("2020-01-01", "2021-12-31", name="Date")
    r = pd.Series(np.full(len(idx), 0.0), index=idx)
    r.iloc[:] = 0.0004
    cy = st.calendar_year_returns(r)
    assert set(cy.index) == {2020, 2021}
    assert (cy > 0).all()
