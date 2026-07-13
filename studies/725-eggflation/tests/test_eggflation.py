"""Fully offline, deterministic tests for Study 725 ("Eggflation").

No network: every test runs on the hardcoded egg series or the fixed-seed synthetic
world. We assert (1) the engine's HAC-OLS matches numpy's closed-form OLS on a known
input, (2) the synthetic positive control recovers a planted lead while the null world
stays flat, (3) the timer/cost accounting is internally consistent, and (4) the egg
proxy has the documented shape (three spikes, continuous monthly grid).

    pytest -q studies/725-eggflation/tests
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from eggflation import data, strategy as st  # noqa: E402


def test_egg_proxy_shape():
    egg = data.load_egg_index()
    # continuous monthly grid, no interior holes
    months = pd.PeriodIndex(egg.index, freq="M")
    expected = pd.period_range(months[0], months[-1], freq="M")
    assert len(months) == len(expected)
    assert not expected.difference(months).size
    # the three documented HPAI peaks are present and ordered
    peaks = data.egg_peaks()
    assert list(peaks["price"]) == sorted(peaks["price"])  # rising records
    assert peaks["price"].iloc[-1] > 6.0  # 2025 all-time high ~ $6.23


def test_hac_ols_matches_numpy_point_estimate():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(200)
    y = 1.5 + 0.7 * x + rng.standard_normal(200)
    r = st.hac_ols(y, [x], lags=6)
    # HAC changes the SEs, not the coefficients — those must match OLS exactly
    X = np.column_stack([np.ones(200), x])
    b_ols = np.linalg.lstsq(X, y, rcond=None)[0]
    assert np.allclose(r["coef"], b_ols, atol=1e-10)
    assert abs(r["coef"][1] - 0.7) < 0.15


def test_synthetic_control_recovers_lead_and_null_is_flat():
    w = data.synthetic_world(lead_beta=0.6, seed=725)
    pr = st.predictive(w, lag=1)
    assert pr["slope"] > 0.3 and pr["t"] > 3.0            # planted lead recovered strongly
    assert st.control_recovers(w, lag=1)["recovered"] == 1
    pl = st.circular_shift_placebo(w, lag=1, n_boot=500, seed=1)
    assert pl["p"] < 0.05                                  # beats its placebo

    w0 = data.synthetic_world(lead_beta=0.0, seed=725)
    assert abs(st.predictive(w0, lag=1)["t"]) < 2.0        # null world stays flat
    assert st.control_recovers(w0, lag=1)["recovered"] == 0


def test_timer_cost_and_sharpe_accounting():
    w = data.synthetic_world(lead_beta=0.6, seed=725)
    tm = st.egg_momentum_timer(w, window=3, rf_annual=0.02)
    # position is a clean 0/1
    assert set(np.unique(tm["signal"].dropna())) <= {0.0, 1.0}
    gross = st.summarize(tm["timer"])
    net = st.summarize(st.apply_costs(tm, cost_bps_one_way=10.0))
    # costs can only reduce the net CAGR (weakly), never increase it
    assert net["cagr"] <= gross["cagr"] + 1e-9
    # zero cost is a no-op
    zero = st.apply_costs(tm, cost_bps_one_way=0.0)
    assert np.allclose(zero.to_numpy(), tm["timer"].to_numpy(), atol=1e-12)


def test_predictive_is_deterministic():
    w = data.synthetic_world(lead_beta=0.4, seed=42)
    a = st.predictive(w, lag=1)["t"]
    b = st.predictive(w, lag=1)["t"]
    assert a == b  # pure function, no hidden randomness
