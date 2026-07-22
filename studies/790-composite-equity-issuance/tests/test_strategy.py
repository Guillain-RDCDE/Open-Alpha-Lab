"""Offline, deterministic tests for Study 790 — Composite Equity Issuance.

Fixed seeds; no network. These pin the machinery: the composite-issuance identity, the
inference primitives, the sort direction, cost accounting, point-in-time discipline, and the
synthetic positive control's null-vs-planted behaviour.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from composite_issuance import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Composite-issuance identity
# --------------------------------------------------------------------------- #
def test_composite_issuance_is_log_share_growth_when_no_dividends():
    """With raw == adj (no dividends) the composite measure equals the log share-count growth
    over the lookback exactly (the price legs cancel)."""
    idx = pd.period_range("2010", periods=8, freq="Y").to_timestamp(how="end")
    px = pd.DataFrame({"A": np.linspace(100, 200, 8)}, index=idx)
    shares = pd.DataFrame({"A": [1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7]}, index=idx)
    iota = st.composite_issuance(px, px, shares, lookback=5)
    expected = np.log(shares["A"].iloc[5] / shares["A"].iloc[0])
    assert iota["A"].iloc[5] == pytest.approx(expected, rel=1e-12)
    # first `lookback` rows are undefined
    assert iota["A"].iloc[:5].isna().all()


def test_composite_issuance_split_neutral():
    """A pure stock split (raw price halves, shares double, adj unchanged) must leave the
    composite measure unchanged — it is not issuance."""
    idx = pd.period_range("2010", periods=7, freq="Y").to_timestamp(how="end")
    raw = pd.DataFrame({"A": [100.0] * 7}, index=idx)
    adj = pd.DataFrame({"A": [100.0] * 7}, index=idx)
    shares = pd.DataFrame({"A": [1.0] * 7}, index=idx)
    base = st.composite_issuance(raw, adj, shares, lookback=5)["A"].iloc[5]
    # apply a 2-for-1 split at t=3: raw halves, shares double from then on
    raw2 = raw.copy(); adj2 = adj.copy(); sh2 = shares.copy()
    raw2.loc[raw2.index[3:], "A"] = 50.0
    sh2.loc[sh2.index[3:], "A"] = 2.0
    split = st.composite_issuance(raw2, adj2, sh2, lookback=5)["A"].iloc[5]
    assert base == pytest.approx(0.0, abs=1e-12)
    assert split == pytest.approx(0.0, abs=1e-12)


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_one_sample_t_sign_and_zero():
    assert st.one_sample_t(np.array([0.09, 0.11, 0.10, 0.10])) > 2   # tight positive mean
    assert st.one_sample_t(np.array([-0.1, -0.2, -0.15, -0.05])) < 0
    assert np.isnan(st.one_sample_t(np.array([1.0])))                 # < 2 obs -> NaN


def test_newey_west_zero_lag_matches_population_t():
    """At lag 0 the HAC t reduces to mean / (population-sd / sqrt(n)) (1/n normalisation) — the
    same sign and magnitude as a plain t up to the ddof convention."""
    x = np.array([0.02, -0.01, 0.03, -0.02, 0.01, 0.0, 0.02])
    n = len(x)
    pop_t = x.mean() / (x.std(ddof=0) / np.sqrt(n))
    assert st.newey_west_t(x, lags=0) == pytest.approx(pop_t, rel=1e-9)
    # positive-lag HAC t keeps the sign of the mean
    assert np.sign(st.newey_west_t(x, lags=1)) == np.sign(x.mean())


def test_wilson_interval_brackets_phat():
    lo, hi = st.wilson_interval(4, 11)
    assert 0.0 <= lo < 4 / 11 < hi <= 1.0


# --------------------------------------------------------------------------- #
# Sort direction + costs
# --------------------------------------------------------------------------- #
def test_sort_direction_long_low_short_high():
    """A planted world where low-issuance names earn more must produce a POSITIVE long-short."""
    raw, adj, sh = data.synthetic_panel(edge=0.15, seed=790, n_years=12)
    ls = st.long_short_series(raw, adj, sh, frac=0.3)
    assert ls["ls_ret"].mean() > 0
    assert (ls["n_long"] > 0).all() and (ls["n_short"] > 0).all()


def test_costs_only_worsen_a_negative_spread():
    raw, adj, sh = data.synthetic_panel(edge=0.0, seed=791, n_years=12)
    gross = st.long_short_series(raw, adj, sh, frac=0.3)["ls_ret"].mean()
    c = st.net_of_costs(raw, adj, sh, frac=0.3, cost_bps=10.0, borrow_ann_bps=50.0)
    assert c["net_mean"] < c["gross_mean"]  # costs always subtract
    assert c["gross_mean"] == pytest.approx(gross, rel=1e-12)


# --------------------------------------------------------------------------- #
# Synthetic positive control — the machinery proof
# --------------------------------------------------------------------------- #
def test_synthetic_null_does_not_fire_planted_does():
    rows = st.synthetic_control(data.synthetic_panel, edges=(0.0, 0.12), n_seeds=25)
    null = next(r for r in rows if r["edge"] == 0.0)
    planted = next(r for r in rows if r["edge"] == 0.12)
    assert abs(null["mean_t"]) < 1.0          # no false positive from a null world
    assert planted["mean_t"] > 3.0            # a real low-issuance edge lights up, RIGHT sign


def test_synthetic_deterministic():
    a = st.synthetic_control(data.synthetic_panel, edges=(0.0,), n_seeds=10)[0]["mean_t"]
    b = st.synthetic_control(data.synthetic_panel, edges=(0.0,), n_seeds=10)[0]["mean_t"]
    assert a == pytest.approx(b, rel=1e-12)


# --------------------------------------------------------------------------- #
# Point-in-time discipline
# --------------------------------------------------------------------------- #
def test_point_in_time_shares_no_lookahead():
    """A fact filed AFTER a year-end must not be visible at that year-end."""
    recs = [
        (pd.Timestamp("2019-12-31"), pd.Timestamp("2020-02-01"), 100.0),  # filed Feb 2020
        (pd.Timestamp("2020-12-31"), pd.Timestamp("2021-02-01"), 120.0),
    ]
    ye = pd.DatetimeIndex([pd.Timestamp("2019-12-31"), pd.Timestamp("2020-12-31")])
    s = data._point_in_time_shares(recs, ye)
    # at 2019-12-31 the 2019 10-K (filed 2020-02-01) is NOT yet known -> NaN
    assert np.isnan(s.loc[pd.Timestamp("2019-12-31")])
    # at 2020-12-31 the first fact IS known (filed 2020-02-01) -> 100.0
    assert s.loc[pd.Timestamp("2020-12-31")] == pytest.approx(100.0)
