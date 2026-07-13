"""Fully-offline, deterministic tests for Study 747 — Founder-Led-Premium.

No network: every assertion runs on the synthetic control or on small in-memory frames.
The point is to pin the engine's *machinery* — that CAPM alpha separates alpha from beta,
that the HAC t is faithful, and that the positive control recovers a plant without faking
one — not the real-tape numbers (those live in docs/results.md).

    pytest -q studies/747-founder-led-premium/tests
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from founder_led_premium import data, strategy as st


def test_baskets_are_disjoint_and_sized():
    fset, pset = set(data.FOUNDER_TICKERS), set(data.PRO_TICKERS)
    assert fset.isdisjoint(pset)                       # no name is both founder and pro
    assert len(data.FOUNDER_TICKERS) == 13
    assert len(data.PRO_TICKERS) == 13
    assert all(f["founder"] for f in data.FOUNDER)
    assert all(not f["founder"] for f in data.PRO)


def test_fingerprint_is_deterministic():
    assert data.fingerprint(data.FIRMS) == data.fingerprint(data.FIRMS)
    assert len(data.fingerprint(data.FIRMS)) == 12


def test_long_short_is_founder_minus_pro():
    idx = pd.date_range("2020-01-01", periods=6, freq="MS")
    rets = pd.DataFrame({
        "A": [0.10, 0.02, -0.03, 0.05, 0.01, 0.04],   # founder leg
        "B": [0.01, 0.00, 0.02, -0.01, 0.03, 0.00],   # pro leg
        "SPY": [0.02, 0.01, 0.00, 0.01, 0.02, 0.01],
    }, index=idx)
    ls = st.long_short(rets, ["A"], ["B"])
    assert np.allclose(ls["ls"].to_numpy(), (rets["A"] - rets["B"]).to_numpy())


def test_missing_name_drops_from_equal_weight():
    # a delisted name is NaN in later months and must simply drop from that month's average
    idx = pd.date_range("2020-01-01", periods=4, freq="MS")
    rets = pd.DataFrame({
        "A": [0.10, 0.20, 0.30, 0.40],
        "B": [0.00, np.nan, np.nan, np.nan],   # "delists" after month 0
    }, index=idx)
    b = st.basket_returns(rets, ["A", "B"])
    assert np.isclose(b.iloc[0], (0.10 + 0.00) / 2)    # both present
    assert np.isclose(b.iloc[1], 0.20)                 # only A survives


def test_capm_separates_alpha_from_beta():
    # construct y = 0.9*mkt exactly (zero alpha) -> alpha ~ 0, beta ~ 0.9
    rng = np.random.default_rng(0)
    mkt = rng.normal(0.01, 0.04, 200)
    y = 0.9 * mkt
    c = st.capm_alpha(y, mkt)
    assert abs(c["alpha"]) < 1e-6
    assert abs(c["beta"] - 0.9) < 1e-6


def test_hac_mean_t_matches_naive_when_iid():
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 1.0, 500)                      # ~iid: HAC t ~ naive t
    hac = st.hac_mean_t(x)
    naive_t = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    assert abs(hac["t"] - naive_t) < 0.5


def test_synthetic_control_no_false_positive_at_zero():
    syn = data.synthetic_baskets(alpha_bps=0.0, seed=747)
    c = st.capm_alpha(syn["ls"], syn["mkt"])
    assert abs(c["t_alpha"]) < 2.0                     # must NOT fabricate significance


def test_synthetic_control_recovers_large_plant():
    syn = data.synthetic_baskets(alpha_bps=200.0, seed=747)
    c = st.capm_alpha(syn["ls"], syn["mkt"])
    assert c["t_alpha"] > 3.0                          # must light up on a real, large edge
    assert c["alpha_bps"] > 150.0


def test_placebo_pvalue_bounds():
    null = np.linspace(-100, 100, 1001)
    assert st.placebo_pvalue(0.0, null) > 0.9          # dead-centre obs -> large p
    assert st.placebo_pvalue(200.0, null) < 0.05       # far tail -> small p


def test_net_of_costs_reduces_gross():
    nc = st.net_of_costs(0.015, 11, 13)                # 150 bps/mo gross
    assert nc["net_bps"] < nc["gross_bps"]
    assert nc["monthly_drag_bps"] > 0
