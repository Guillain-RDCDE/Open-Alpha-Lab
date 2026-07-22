"""Offline, deterministic tests for Study 796 — Corporate-Bond-Low-Risk.

No network: everything runs on the deterministic synthetic control and on tiny synthetic
frames. Asserts the BAB (low-minus-high) machinery is faithful (recovers a planted low-risk
tilt, does NOT manufacture one from a null panel), that there is no look-ahead in the
execution lag, and that the pure inference helpers behave.

    pytest -q studies/796-corporate-bond-low-risk/tests
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from bond_low_risk import data, strategy as st


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
# The trailing-vol signal + book — mechanics and no look-ahead
# --------------------------------------------------------------------------- #
def test_trailing_vol_ranks_a_calm_vs_wild_asset():
    # A calm ramp vs a jumpy series: the calm one must show the lower trailing vol.
    idx = pd.bdate_range("2015-01-02", periods=300)
    rng = np.random.default_rng(0)
    calm = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.001, 300)), index=idx)
    wild = pd.Series(100 * np.cumprod(1 + rng.normal(0, 0.02, 300)), index=idx)
    prices = pd.concat({"CALM": calm, "WILD": wild}, axis=1)
    dret = prices.pct_change()
    vol = st.trailing_vol_at(dret, idx[-1], window=252)
    assert vol["CALM"] < vol["WILD"]


def test_bab_book_execution_lag_no_lookahead():
    # The book must never book a return on the very first month-end (one forward shift).
    prices, _ = data.synthetic_panel(lowrisk_strength=0.0, seed=1)
    bk = st.bab_book(prices)
    assert not bk.empty
    assert bk.index.min() > st.to_monthly(prices).index[0]
    assert set(["bab_gross", "low_ret", "high_ret", "k_low", "k_high", "turnover",
                "bab_net"]).issubset(bk.columns)


def test_bab_levers_the_low_leg_up():
    # By construction the low-vol leg is levered UP toward the risk target, the high-vol
    # leg levered DOWN — the BAB signature.
    prices, _ = data.synthetic_panel(lowrisk_strength=0.0, seed=1)
    bk = st.bab_book(prices)
    assert bk["k_low"].mean() > 1.0
    assert bk["k_high"].mean() < bk["k_low"].mean()


def test_costs_and_financing_charged_once_and_reduce_mean():
    prices, _ = data.synthetic_panel(lowrisk_strength=0.0, seed=1)
    g = st.bab_book(prices, cost_bps=0.0, fin_ann_bps=0.0)
    n = st.bab_book(prices, cost_bps=10.0, fin_ann_bps=120.0)
    # net strictly below gross; the gap ~ turnover*10bps + borrowed*120/12 bps
    assert n["bab_net"].mean() < g["bab_gross"].mean()
    borrowed = (g["k_low"] - 1.0).clip(lower=0.0) + g["k_high"]
    approx = (g["turnover"] * 10e-4 + borrowed * 120e-4 / 12).mean()
    assert abs((g["bab_gross"].mean() - n["bab_net"].mean()) - approx) < 1e-6


# --------------------------------------------------------------------------- #
# The synthetic positive control — faithful & powered
# --------------------------------------------------------------------------- #
def test_synthetic_null_does_not_fire():
    # No planted low-risk tilt (one Sharpe line) -> the BAB book must not clear the bar.
    prices, truth = data.synthetic_panel(lowrisk_strength=0.0, seed=314)
    assert not truth.has_lowrisk
    s = st.summary(st.bab_book(prices)["bab_gross"])
    assert abs(s["tstat"]) < 2.0


def test_synthetic_planted_lowrisk_lights_up():
    # A real planted low-risk premium must be recovered with a big HAC t.
    prices, truth = data.synthetic_panel(lowrisk_strength=1.0, seed=314)
    assert truth.has_lowrisk
    s = st.summary(st.bab_book(prices)["bab_gross"])
    assert s["tstat"] > 3.0
    assert s["mean"] > 0.03


def test_synthetic_null_robustness_over_many_seeds():
    nr = st.synthetic_null_robustness(n_seeds=10)
    assert nr["n_seeds"] == 10
    assert abs(nr["mean_null_t"]) < 1.0     # unbiased on average
    assert nr["frac_abs_t_gt2"] <= 0.2      # rarely fires by chance


def test_placebo_pvalue_large_on_null_panel():
    # On a no-anomaly panel the real BAB should not beat the vol-rank-shuffle null.
    prices, _ = data.synthetic_panel(lowrisk_strength=0.0, seed=314)
    pl = st.placebo_pvalue(prices, n_perm=300)
    assert pl["n_perm"] == 300
    assert pl["p_value"] > 0.10


def test_fingerprint_is_deterministic():
    prices, _ = data.synthetic_panel(lowrisk_strength=0.0, seed=314)
    assert data.fingerprint(prices) == data.fingerprint(prices)
