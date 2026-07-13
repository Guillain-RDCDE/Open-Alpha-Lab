"""Fully offline, deterministic tests for Study 753 — Reverse-Repo-Drain.

No network: everything runs on the deterministic synthetic control and on pure-function
identities. Verifies (1) the drain regime split is a proper, no-look-ahead partition,
(2) the first ``k`` months are undefined (dropped) not mislabelled, (3) the inference is a
faithful positive control — a zero planted edge stays below t=2, a large planted edge clears
it, and (4) Welch t is sign-consistent with the mean spread.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from reverse_repo_drain import data, strategy as st


def test_regime_partition_and_lag():
    syn = data.synthetic(n_months=120, edge=0.0, seed=753)
    drain, fill = st.regime_returns(syn, k=3, lag=1)
    # a proper partition of the usable months (first k undefined + last dropped by shift)
    assert len(drain) > 5 and len(fill) > 5
    assert len(drain) + len(fill) == 120 - 3 - 1  # drop first k (undefined) + last (no fwd)


def test_first_k_months_undefined():
    syn = data.synthetic(n_months=60, edge=0.0, seed=1)
    flag = st.draining(syn, k=6)
    # the first k months have no trailing change -> NaN (not silently 'fill')
    assert flag.iloc[:6].isna().all()
    assert flag.iloc[6:].notna().all()


def test_zero_edge_is_not_significant():
    syn = data.synthetic(n_months=240, edge=0.0, seed=753)
    s = st.summarize(syn, k=3)
    assert abs(s["t"]) < 2.0            # a noise drain cannot fake significance
    assert s["p_placebo"] > 0.05


def test_planted_edge_lights_up():
    syn = data.synthetic(n_months=240, edge=0.03, seed=753)
    s = st.summarize(syn, k=3)
    assert s["t"] > 2.0                 # a real planted drain edge is detected
    assert s["spread"] > 0
    assert s["p_placebo"] < 0.05


def test_welch_sign_matches_spread():
    syn = data.synthetic(n_months=180, edge=0.02, seed=7)
    s = st.summarize(syn, k=3)
    assert np.sign(s["t"]) == np.sign(s["spread"])


def test_timing_backtest_reports_gross_and_net():
    syn = data.synthetic(n_months=180, edge=0.0, seed=3)
    b = st.timing_backtest(syn, cost_bps=10.0)
    # net is never above gross once costs are charged, and buy-hold is reported
    assert b["net"]["ann_ret"] <= b["gross"]["ann_ret"] + 1e-9
    assert "sharpe" in b["buy_hold"]
    assert 0.0 <= b["exposure"] <= 1.0
