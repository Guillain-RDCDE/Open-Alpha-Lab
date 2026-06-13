"""Strategy-engine tests for Study 100 (Melting-Ice), including the spine tests."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from melting_ice import data, strategy  # noqa: E402


# ---------------------------------------------------------------------------
# The constant-leverage identity
# ---------------------------------------------------------------------------
def test_letf_one_day_is_exactly_k_times_minus_fee():
    """Over a single day the levered return is k*r minus the daily fee — the identity."""
    close = pd.Series([100.0, 110.0])  # +10% underlying
    nav = strategy.simulate_letf(close, k=3.0, annual_fee=0.0)
    assert np.isclose(nav.iloc[-1] - 1.0, 3.0 * 0.10)


def test_fee_reduces_nav():
    close = pd.Series(np.linspace(100, 130, 300))
    free = strategy.simulate_letf(close, k=3.0, annual_fee=0.0)
    paid = strategy.simulate_letf(close, k=3.0, annual_fee=0.05)
    assert paid.iloc[-1] < free.iloc[-1]


def test_naive_curve_is_linear_in_cumulative_return():
    close = pd.Series([100.0, 100.0, 120.0])  # +20% over the window
    naive = strategy.naive_k_curve(close, k=3.0)
    assert np.isclose(naive.iloc[-1], 1.0 + 3.0 * 0.20)


def test_breakeven_identity_k3_is_g_equals_s2():
    """For k=3 the break-even variance equals the mean daily log-return: s2* = g."""
    close = pd.Series(100.0 * np.exp(np.cumsum(np.full(500, 0.0005))))  # constant drift
    d = strategy.drag_decomposition(close, k=3.0)
    # constant drift => zero variance => break-even variance ~ g (here g>0, s2~0).
    assert np.isclose(d["s2_star_daily"], d["g_daily"], rtol=1e-6)


def test_drag_term_scales_as_k_k_minus_1(uptrend_lowvol):
    """Doubling within the k(k-1)/2 family: 3x drag is 3x the 2x drag at same vol."""
    close = uptrend_lowvol[0]["close"]
    d2 = strategy.drag_decomposition(close, k=2.0)
    d3 = strategy.drag_decomposition(close, k=3.0)
    # k(k-1)/2: 2x -> 1*s2, 3x -> 3*s2, so the 3x drag is 3x the 2x drag.
    assert np.isclose(d3["drag_ann"] / d2["drag_ann"], 3.0, rtol=1e-9)


# ---------------------------------------------------------------------------
# Spine tests — the harness must report the SIGN of the gap correctly on BOTH paths
# ---------------------------------------------------------------------------
def test_spine_flat_highvol_decays(flat_highvol):
    """Zero-drift high-vol tape: the realised 3x NAV must end BELOW naive-3x (decay)."""
    close = flat_highvol[0]["close"]
    rvn = strategy.realized_vs_naive(close, k=3.0, annual_fee=0.0)
    d = strategy.drag_decomposition(close, k=3.0)
    assert rvn["compounding_won"] is False
    assert d["drag_dominates"] is True


def test_spine_uptrend_lowvol_compounds(uptrend_lowvol):
    """Steady low-vol uptrend: the realised 3x NAV must end ABOVE naive-3x (compounding)."""
    close = uptrend_lowvol[0]["close"]
    rvn = strategy.realized_vs_naive(close, k=3.0, annual_fee=0.0)
    d = strategy.drag_decomposition(close, k=3.0)
    assert rvn["compounding_won"] is True
    assert d["drag_dominates"] is False


# ---------------------------------------------------------------------------
# Real-tape reproduction — the simulator must match the actual fund within tolerance
# ---------------------------------------------------------------------------
def _have_real(name):
    try:
        data.load_pair(name, end="2026-06-12")
        return True
    except Exception:
        return False


@pytest.mark.skipif(not _have_real("TQQQ"),
                    reason="real TQQQ/QQQ tape unavailable (no cache, no network)")
def test_simulator_reproduces_real_tqqq():
    """The 3x simulator, fed QQQ, must reproduce the REAL TQQQ fund within tolerance."""
    under, letf, _ = data.load_pair("TQQQ", end="2026-06-12")
    rep = strategy.reproduction_error(under["close"], letf["close"], k=3.0)
    assert rep["corr"] > 0.99           # daily mechanics match almost exactly
    assert rep["rms_te_bps"] < 40.0     # small daily tracking error
    assert 0.8 < rep["final_ratio"] < 1.25  # final NAV within ~20% over 16 years


@pytest.mark.skipif(not _have_real("UPRO"),
                    reason="real UPRO/SPY tape unavailable (no cache, no network)")
def test_simulator_reproduces_real_upro():
    under, letf, _ = data.load_pair("UPRO", end="2026-06-12")
    rep = strategy.reproduction_error(under["close"], letf["close"], k=3.0)
    assert rep["corr"] > 0.99
    assert rep["rms_te_bps"] < 40.0
    assert 0.8 < rep["final_ratio"] < 1.25


# ---------------------------------------------------------------------------
# Stats & HAC sanity
# ---------------------------------------------------------------------------
def test_nav_stats_keys_and_drawdown_sign(uptrend_lowvol):
    nav = strategy.simulate_letf(uptrend_lowvol[0]["close"], k=3.0)
    s = strategy.nav_stats(nav)
    assert {"final", "cagr", "vol", "sharpe", "max_dd"} <= set(s)
    assert s["max_dd"] <= 0.0


def test_hac_tstat_finite_on_real_series():
    x = np.random.default_rng(0).normal(0.001, 0.01, 2000)
    t = strategy.hac_tstat(x)
    assert np.isfinite(t)
