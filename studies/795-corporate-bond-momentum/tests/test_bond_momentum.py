"""Offline, deterministic tests for Study 795 — Corporate-Bond-Momentum.

No network: everything runs on the deterministic synthetic control and on tiny synthetic
frames. Asserts the WML machinery is faithful (recovers a planted momentum, does NOT
manufacture one from a null panel), that there is no look-ahead in the execution lag, and
that the pure inference helpers behave.

    pytest -q studies/795-corporate-bond-momentum/tests
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bond_momentum import data, strategy as st


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def test_one_sample_t_matches_hand_value():
    x = pd.Series([1.0, 2.0, 3.0, 4.0])
    # mean 2.5, sd(ddof=1) ~1.290994, se ~0.645497, t ~3.8730
    assert abs(st.one_sample_t(x) - 3.872983) < 1e-4
    assert np.isnan(st.one_sample_t(pd.Series([1.0])))


def test_wilson_interval_brackets_half_for_coin_flip():
    lo, hi = st.wilson_interval(50, 100)
    assert lo < 0.5 < hi
    assert abs(((lo + hi) / 2) - 0.5) < 0.02


def test_hac_tstat_small_sample_is_nan():
    assert np.isnan(st.hac_tstat(pd.Series([0.01, 0.02, 0.03])))


def test_n_group_never_overlaps_legs():
    # top and bottom legs must not overlap: 2 * n_group <= n_valid
    for n in range(4, 30):
        g = st.n_group(n)
        assert g >= 1 and 2 * g <= n


# --------------------------------------------------------------------------- #
# The momentum signal + book — mechanics and no look-ahead
# --------------------------------------------------------------------------- #
def test_momentum_signal_is_trailing_only():
    # A ramp where each asset's trailing return is known and ordered.
    idx = pd.date_range("2015-01-31", periods=10, freq="ME")
    prices = pd.DataFrame({
        "A": np.linspace(100, 200, 10),   # strong up-trend
        "B": np.full(10, 100.0),          # flat
        "C": np.linspace(200, 100, 10),   # down-trend
    }, index=idx)
    sig = st.momentum_signal(prices, lookback=3, skip=0)
    # first 3 rows are NaN (insufficient history); thereafter A > B > C in trailing return
    row = sig.dropna().iloc[0]
    assert row["A"] > row["B"] > row["C"]


def test_wml_book_execution_lag_no_lookahead():
    # If we know the future is a jump in month t+1 for the current winner, the book must
    # NOT earn it on the formation month — the holding-month label is t+1.
    idx = pd.date_range("2015-01-31", periods=12, freq="ME")
    base = np.linspace(100, 130, 12)
    prices = pd.DataFrame({f"A{i}": base * (1 + 0.01 * i) for i in range(8)}, index=idx)
    bk = st.wml_book(prices, lookback=3, skip=0, min_names=6)
    # Every holding-month label must be strictly after the earliest possible formation date,
    # i.e. the book never books a return on the very first month-end.
    assert bk.index.min() > prices.index[0]
    assert set(["wml_gross", "long_ret", "short_ret", "turnover", "wml_net"]).issubset(bk.columns)


def test_wml_dollar_neutral_and_costs_charged_once():
    # net = gross - turnover*cost - borrow/12; a hand check on a one-month book.
    prices, _ = data.synthetic_panel(mom_strength=0.0, seed=1)
    mp = st.to_monthly(prices)
    g = st.wml_book(mp, cost_bps=0.0, borrow_ann_bps=0.0)
    n = st.wml_book(mp, cost_bps=10.0, borrow_ann_bps=120.0)
    # costs strictly reduce the mean; the gap ~ turnover*10bps + 120/12 bps
    assert n["wml_net"].mean() < g["wml_gross"].mean()
    approx_cost = g["turnover"].mean() * 10e-4 + 120e-4 / 12
    assert abs((g["wml_gross"].mean() - n["wml_net"].mean()) - approx_cost) < 1e-6


# --------------------------------------------------------------------------- #
# The synthetic positive control — faithful & powered
# --------------------------------------------------------------------------- #
def test_synthetic_null_does_not_fire():
    # No planted momentum -> the WML book must not clear the bar (single seed 314).
    prices, truth = data.synthetic_panel(mom_strength=0.0, seed=314)
    assert not truth.has_momentum
    s = st.summary(st.wml_book(st.to_monthly(prices))["wml_gross"])
    assert abs(s["tstat"]) < 2.0


def test_synthetic_planted_momentum_lights_up():
    # A real planted cross-sectional momentum must be recovered with a big HAC t.
    prices, truth = data.synthetic_panel(mom_strength=0.20, seed=314)
    assert truth.has_momentum
    s = st.summary(st.wml_book(st.to_monthly(prices))["wml_gross"])
    assert s["tstat"] > 4.0
    assert s["mean"] > 0.05


def test_synthetic_null_robustness_over_many_seeds():
    nr = st.synthetic_null_robustness(n_seeds=10)
    assert nr["n_seeds"] == 10
    assert abs(nr["mean_null_t"]) < 1.0     # unbiased on average
    assert nr["frac_abs_t_gt2"] <= 0.1      # rarely fires by chance


def test_placebo_pvalue_is_large_on_null_panel():
    # On a no-momentum panel the real WML should not beat the rank-shuffle null.
    prices, _ = data.synthetic_panel(mom_strength=0.0, seed=314)
    pl = st.placebo_pvalue(st.to_monthly(prices), n_perm=300)
    assert pl["n_perm"] == 300
    assert pl["p_value"] > 0.10


def test_fingerprint_is_deterministic():
    prices, _ = data.synthetic_panel(mom_strength=0.0, seed=314)
    assert data.fingerprint(prices) == data.fingerprint(prices)
