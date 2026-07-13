"""Fully offline, deterministic tests for Study 745 — Corporate-Jet-Index.

No network: every test runs on the synthetic panel or pure functions. Confirms the
inference machinery is faithful (recovers a planted governance discount, stays quiet under
the null) and that the long/short construction has the right sign convention.

    pytest -q studies/745-corporate-jet-index/tests
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from corporate_jet_index import data, strategy as st


def test_table_shape():
    # 12 heavy / 12 low, disjoint, stable fingerprint.
    assert len(data.HEAVY) == 12 and len(data.LOW) == 12
    assert not (set(data.HEAVY) & set(data.LOW))
    assert data.fingerprint(data.JET_FIRMS) == "ad201b5ed46c"


def test_hac_tstat_matches_naive_on_iid():
    # On i.i.d. noise the HAC t and the naive t should nearly coincide.
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 0.02, 400)
    hac = st.hac_tstat(x)
    naive = x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))
    assert abs(hac["t"] - naive) < 0.25
    assert hac["n"] == 400 and hac["lags"] >= 1


def test_control_null_is_quiet():
    # With no planted discount, HAC t must stay inside |t| < 2 (no false positive).
    syn = data.synthetic_panel(alpha_bps_month=0.0, seed=745)
    assert abs(st.hac_tstat(syn["ls"])["t"]) < 2.0


def test_control_recovers_planted_discount():
    # A large planted heavy-basket discount must light up as a POSITIVE long/short t
    # (long low - short heavy), and clear t = 2.
    syn = data.synthetic_panel(alpha_bps_month=-80.0, seed=745)
    h = st.hac_tstat(syn["ls"])
    assert h["mean"] > 0 and h["t"] > 2.0


def test_market_model_alpha_strips_beta():
    # If LS = 1.5*mkt + small noise (pure beta, no alpha), the recovered alpha ~ 0
    # and beta ~ 1.5.
    rng = np.random.default_rng(7)
    mkt = rng.normal(0.006, 0.043, 300)
    ls = 1.5 * mkt + rng.normal(0.0, 0.005, 300)
    mm = st.market_model_alpha(ls, mkt)
    assert abs(mm["beta"] - 1.5) < 0.1
    # the intercept is economically ~0 (a few bp/mo), i.e. no real alpha planted
    assert abs(mm["alpha"]) < 0.002


def test_net_of_costs_reduces_and_charges_borrow():
    import pandas as pd
    s = pd.Series(np.full(120, 0.004))         # +40 bps/mo flat
    nc = st.net_of_costs(s)
    assert nc["net_month"] < nc["gross_month"]           # costs bite
    assert nc["borrow_ann"] > 0                          # short pays borrow
    # net drag ~ rebalance + borrow, both positive
    assert nc["gross_ann"] - nc["net_ann"] > 0


def test_annualize_mean_roundtrip():
    assert abs(st.annualize_mean(0.0) - 0.0) < 1e-12
    assert st.annualize_mean(0.01) > 0.12   # compounding > 12*monthly
