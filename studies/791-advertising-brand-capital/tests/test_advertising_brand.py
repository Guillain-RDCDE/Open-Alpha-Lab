"""Offline, deterministic tests for Study 791 — Advertising-Brand-Capital.

No network: every test runs on the synthetic control or on small hand-built panels. Fixed
seeds throughout. These guard the inference machinery (the positive control fires only where
an effect is planted) and the honesty rails (survivorship guard, one-month execution lag,
costs one-way x NAV, shorts pay borrow).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from advertising_brand import data, strategy as st  # noqa: E402

MPY = st.MONTHS_PER_YEAR


# --------------------------------------------------------------------------- #
# Synthetic positive control — null must not fire, planted edge must be recovered
# --------------------------------------------------------------------------- #
def test_synthetic_null_is_flat():
    """edge = 0: the long-short is statistically indistinguishable from zero across seeds."""
    ts = []
    for s in range(10):
        sig, rets, _, _ = data.synthetic_panel(edge=0.0, seed=791 + s)
        b = st.signal_books(sig, rets, frac=1 / 3)
        ts.append(st.hac_tstat(b["long_short"])["tstat"])
    ts = np.asarray(ts)
    assert abs(ts.mean()) < 1.0          # centred on zero
    assert (np.abs(ts) >= 2).sum() <= 1  # essentially never a false positive


def test_synthetic_planted_edge_recovered():
    """A large planted premium must light the detector up (t clears 2), sign correct."""
    sig, rets, _, truth = data.synthetic_panel(edge=0.08, seed=791)
    b = st.signal_books(sig, rets, frac=1 / 3)
    d = st.hac_tstat(b["long_short"])
    assert d["mean_ann"] > 0.03          # recovered a positive premium
    assert d["tstat"] > 2.0              # and it is significant


def test_synthetic_edge_monotone():
    """More planted edge => larger recovered spread."""
    spreads = []
    for edge in (0.0, 0.04, 0.08):
        sig, rets, _, _ = data.synthetic_panel(edge=edge, seed=791)
        b = st.signal_books(sig, rets, frac=1 / 3)
        spreads.append(b["long_short"].mean() * MPY)
    assert spreads[0] < spreads[1] < spreads[2]


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_hac_tstat_zero_mean():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(0, 0.05, 300))
    assert abs(st.hac_tstat(x)["tstat"]) < 2.0


def test_welch_and_one_sample_signs():
    a = np.array([0.02, 0.03, 0.025, 0.028])
    b = np.array([-0.01, 0.0, -0.005, 0.002])
    assert st.welch_t(a, b) > 0
    assert st.one_sample_t(a) > 0
    assert st.one_sample_t(-a) < 0


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(60, 100)
    assert lo < 0.60 < hi
    assert 0.0 <= lo < hi <= 1.0


# --------------------------------------------------------------------------- #
# Portfolio-sort mechanics
# --------------------------------------------------------------------------- #
def test_tertile_members_split():
    row = pd.Series({"a": 0.10, "b": 0.08, "c": 0.05, "d": 0.03, "e": 0.01, "f": 0.00})
    lo, sh = st.tertile_members(row, frac=1 / 3)
    assert lo == ["a", "b"]           # top two (descending)
    assert sh == ["e", "f"]           # bottom two (the last two of a descending sort)


def test_execution_lag_is_one_month():
    """Signal at month t earns month t+1's return: the leg series is indexed by the HOLDING
    months (months[1:]) — the first month is signal-only (never a holding return), and each
    holding return is the *next* month's return of the members the *prior* signal selected."""
    months = pd.date_range("2020-01-31", periods=4, freq="ME")
    cols = ["a", "b", "c"]
    # constant ranking a > b > c, so 'a' is the long tertile every month
    sig = pd.DataFrame({"a": 3.0, "b": 2.0, "c": 1.0}, index=months)
    rets = pd.DataFrame({"a": [0.01, 0.02, 0.03, 0.04], "b": 0.0, "c": 0.0}, index=months)
    lr, lt = st._leg_returns(sig, rets, frac=1 / 3, which="long")
    # holding months are exactly months[1:]; the first month is signal-only
    assert list(lr.index) == list(months[1:])
    assert months[0] not in lr.index
    # the holding return at month t+1 is that month's return of 'a' (chosen by signal at t)
    assert lr.loc[months[1]] == pytest.approx(0.02)
    assert lr.loc[months[3]] == pytest.approx(0.04)


def test_costs_reduce_returns_and_borrow_hits_short():
    sig, rets, _, _ = data.synthetic_panel(edge=0.05, seed=1)
    gross = st.signal_books(sig, rets, frac=1 / 3, cost_bps=0.0, borrow_bps=0.0)
    net = st.signal_books(sig, rets, frac=1 / 3, cost_bps=20.0, borrow_bps=200.0)
    assert net["ls_net"].mean() < gross["long_short"].mean()
    # borrow is charged once per month on the short leg only
    diff = (gross["long_short"] - net["ls_net"]).mean()
    assert diff > 0


def test_placebo_null_centered_near_zero():
    sig, rets, _, _ = data.synthetic_panel(edge=0.0, seed=5)
    null = st.placebo_null(sig, rets, frac=1 / 3, n_shuffles=60, seed=5)
    assert abs(np.nanmean(null)) < 0.02      # a shuffled signal pays ~nothing


def test_placebo_recovers_real_in_tail():
    sig, rets, _, _ = data.synthetic_panel(edge=0.08, seed=7)
    b = st.signal_books(sig, rets, frac=1 / 3)
    real_ls = b["long_short"].mean() * MPY
    null = st.placebo_null(sig, rets, frac=1 / 3, n_shuffles=100, seed=7)
    assert st.percentile_of(real_ls, null) > 90.0


# --------------------------------------------------------------------------- #
# Signal construction & honesty rails
# --------------------------------------------------------------------------- #
def test_build_signal_is_adv_over_sales_with_lag():
    """adv_sales at a month uses the most-recent fiscal year known with a 1-year lag."""
    adv = pd.DataFrame({"X": {2018: 10.0, 2019: 20.0}}); adv.index.name = "fy"
    rev = pd.DataFrame({"X": {2018: 100.0, 2019: 100.0}}); rev.index.name = "fy"
    months = pd.date_range("2020-06-30", periods=1, freq="ME")
    rets = pd.DataFrame({"X": [0.0]}, index=months)
    real = {"adv": adv, "rev": rev, "returns": rets}
    sig = data.build_signal(real, report_lag=1)
    # in 2020, fy <= 2019 known -> uses FY2019 => 20/100 = 0.20
    assert sig.loc[months[0], "X"] == pytest.approx(0.20)


def test_missing_advertising_is_nan_not_zero():
    """A firm-year with no advertising figure goes UNRANKED (NaN), not floored to 0."""
    adv = pd.DataFrame({"X": {2019: 10.0}, "Y": {2019: np.nan}}); adv.index.name = "fy"
    rev = pd.DataFrame({"X": {2019: 100.0}, "Y": {2019: 100.0}}); rev.index.name = "fy"
    months = pd.date_range("2021-06-30", periods=1, freq="ME")
    rets = pd.DataFrame({"X": [0.0], "Y": [0.0]}, index=months)
    sig = data.build_signal({"adv": adv, "rev": rev, "returns": rets})
    assert not np.isnan(sig.loc[months[0], "X"])
    assert np.isnan(sig.loc[months[0], "Y"])


def test_load_real_survivorship_guard():
    """The panel loader refuses to hand back a current-membership basket without the opt-in."""
    if not data.have_real():
        pytest.skip("no cache present (offline CI without the pinned parquet)")
    with pytest.raises(PermissionError):
        data.load_real(allow_survivorship_bias=False)
    real = data.load_real(allow_survivorship_bias=True)
    assert "adv" in real and "returns" in real


def test_fingerprint_deterministic():
    sig, rets, _, _ = data.synthetic_panel(edge=0.03, seed=3)
    assert data.fingerprint(sig, rets) == data.fingerprint(sig, rets)
