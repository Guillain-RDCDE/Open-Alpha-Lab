"""Fully offline, deterministic tests for Study 727.

No network: every test runs on the hardcoded maple series or synthetic data. We assert
pure-function behaviour and the two facts the verdict rests on — an administered price is
near-flat, and the seasonality engine can recover a *planted* season (positive control) yet
reports a null when there isn't one.

    pytest -q studies/727-maple-syrup-reserve/tests
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from maple_syrup_reserve import data, strategy as st  # noqa: E402


def test_maple_price_is_administered_and_flat():
    """The hardcoded PPAQ price is a near-flat, low-vol administered creep (the whole point)."""
    m = data.load_maple_price()
    s = st.summarize(m, periods_per_year=1.0)
    assert m.index[0].year == 2008 and m.index[-1].year == 2024
    assert 0.0 < s["cagr"] < 0.05          # a couple of percent a year, positive but tiny
    assert s["vol"] < 0.10                 # << a real soft commodity's 25-40%/yr
    assert s["mdd"] > -0.10                # barely any drawdown — a defended price


def test_annual_excess_sign_and_smallsample():
    """Maple underperforms a higher-return benchmark; the paired-excess t is finite."""
    m = data.load_maple_price()
    # a synthetic ~7%/yr benchmark on the same year-end grid
    idx = pd.to_datetime([f"{y}-12-31" for y in range(2008, 2025)])
    bench = pd.Series(100.0 * (1.07) ** np.arange(len(idx)), index=idx)
    ae = st.annual_excess_t(m, bench)
    assert ae["mean_excess"] < 0.0        # maple loses the race
    assert np.isfinite(ae["t"]) and ae["n"] >= 10


def test_synthetic_control_recovers_planted_season():
    """Positive control: a planted Feb-Apr premium is recovered with the right sign and |t|>2."""
    world, truth = data.synthetic_world()
    sea = st.season_tstat(world["ret"])
    cr = st.control_recovers(world["ret"], planted_sign=1)
    assert truth["spring_premium"] > 0
    assert cr["sign_ok"] == 1
    assert sea["spread"] > 0 and sea["tstat"] > 2.0


def test_synthetic_null_has_no_season():
    """With zero planted premium the same engine reports a null (no false positive)."""
    world, _ = data.synthetic_world(spring_premium=0.0, seed=42)
    sea = st.season_tstat(world["ret"])
    ci = st.season_bootstrap_ci(world["ret"], n_boot=1500, seed=42)
    assert abs(sea["tstat"]) < 2.0
    assert ci["lo"] < 0.0 < ci["hi"]      # CI straddles zero


def test_timer_flat_leg_earns_benchmark_no_lookahead():
    """Outside the season the timer earns the benchmark; in-season it earns the proxy."""
    idx = pd.date_range("2015-01-31", periods=24, freq="ME")
    proxy = pd.Series(0.02, index=idx)     # +2%/mo proxy
    bench = pd.Series(0.01, index=idx)     # +1%/mo benchmark
    timer = st.seasonal_timer(proxy, bench)
    for ts, v in timer.items():
        expected = 0.02 if ts.month in st.SUGARING_MONTHS else 0.01
        assert abs(v - expected) < 1e-12


def test_costs_reduce_return_by_budget():
    """apply_costs deducts exactly (legs*bps/1e4)/12 per month — one-way x NAV, documented."""
    idx = pd.date_range("2015-01-31", periods=12, freq="ME")
    r = pd.Series(0.0, index=idx)
    net = st.apply_costs(r, n_legs_per_year=2, cost_bps_one_way=15)
    assert np.allclose(net.values, -(2 * 15 / 1e4) / 12)
