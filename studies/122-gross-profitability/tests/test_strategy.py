"""Signal construction, the long-short engine, and the study's core spine:
the hedge beats the market only when a premium is actually planted."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gross_profitability import strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------
def test_gross_profitability_signal():
    """GP/A = GrossProfit / Assets, aligned on common tickers."""
    years = pd.Index([2010, 2011, 2012], name="year")
    tickers = ["A", "B", "C"]
    gp = pd.DataFrame([[100, 200, 150], [110, 210, 160], [120, 220, 170]],
                      index=years, columns=tickers, dtype=float)
    assets = pd.DataFrame([[1000, 500, 750], [1100, 510, 760], [1200, 520, 770]],
                          index=years, columns=tickers, dtype=float)
    gpa = st.gross_profitability(gp, assets)
    assert gpa.shape == (3, 3)
    assert np.allclose(gpa.to_numpy(), (gp / assets).to_numpy())


def test_gross_profitability_handles_missing_tickers():
    """Tickers present in only one frame are dropped gracefully."""
    years = pd.Index([2010], name="year")
    gp = pd.DataFrame([[100, 200]], index=years, columns=["A", "B"])
    assets = pd.DataFrame([[1000]], index=years, columns=["A"])  # B missing
    gpa = st.gross_profitability(gp, assets)
    assert "B" not in gpa.columns
    assert "A" in gpa.columns


# ---------------------------------------------------------------------------
# Quintile hedge
# ---------------------------------------------------------------------------
def _make_panel(n_firms=100, n_years=10, premium=0.0, seed=1):
    rng = np.random.default_rng(seed)
    years = list(range(2010, 2010 + n_years))
    firms = [f"F{i}" for i in range(n_firms)]
    raw = rng.lognormal(0, 0.5, size=(n_years, n_firms))
    z = (raw - raw.mean(1, keepdims=True)) / (raw.std(1, keepdims=True) + 1e-9)
    fwd = 0.1 + premium * z + 0.3 * rng.standard_normal((n_years, n_firms))
    sig = pd.DataFrame(raw, index=pd.Index(years, name="year"), columns=firms)
    ret = pd.DataFrame(fwd, index=pd.Index(years, name="year"), columns=firms)
    return sig, ret


def test_quintile_hedge_columns():
    sig, ret = _make_panel()
    h = st.quintile_hedge(sig, ret)
    assert set(h.columns) >= {"high", "low", "hedge", "market", "n"}


def test_quintile_hedge_index_is_years():
    sig, ret = _make_panel(n_years=8)
    h = st.quintile_hedge(sig, ret)
    assert h.index.name == "year" or h.index.name is None  # .T loses name
    assert len(h) <= 8


def test_hedge_is_high_minus_low():
    sig, ret = _make_panel()
    h = st.quintile_hedge(sig, ret)
    assert np.allclose(h["hedge"], h["high"] - h["low"])


def test_hedge_positive_with_premium():
    """With a planted premium, the high-GP/A quintile should beat the low."""
    sig, ret = _make_panel(n_firms=300, n_years=20, premium=0.10, seed=42)
    h = st.quintile_hedge(sig, ret)
    assert h["hedge"].mean() > 0.0


def test_hedge_near_zero_without_premium():
    """Without a premium the long-short is noise around zero."""
    sig, ret = _make_panel(n_firms=300, n_years=30, premium=0.0, seed=7)
    h = st.quintile_hedge(sig, ret)
    # Mean should be small relative to cross-sectional vol; |t| < 2 almost surely.
    n = len(h)
    mean = h["hedge"].mean()
    vol = h["hedge"].std(ddof=1)
    naive_t = abs(mean / vol * np.sqrt(n))
    assert naive_t < 3.0  # nearly always holds for noise with n≈30


def test_summary_smoke(null_panel):
    signal, fwd_ret, _ = null_panel
    h = st.quintile_hedge(signal, fwd_ret)
    s = st.summary(h["hedge"])
    assert "mean" in s and "tstat" in s and "n" in s
    assert s["n"] > 0
    assert np.isfinite(s["tstat"])


def test_summary_hit_rate_bounds(live_panel):
    signal, fwd_ret, _ = live_panel
    h = st.quintile_hedge(signal, fwd_ret)
    s = st.summary(h["hedge"])
    assert 0.0 <= s["hit_rate"] <= 1.0


def test_summary_sharpe_sign_matches_mean(live_panel):
    """Sharpe and mean have the same sign (they share the same numerator)."""
    signal, fwd_ret, _ = live_panel
    h = st.quintile_hedge(signal, fwd_ret)
    s = st.summary(h["hedge"])
    assert np.sign(s["sharpe"]) == np.sign(s["mean"])


# ---------------------------------------------------------------------------
# The study's spine — planted premium detection
# ---------------------------------------------------------------------------
def test_hedge_detects_premium_vs_null():
    """With a strong planted premium the hedge mean is clearly above the null case."""
    sig_null, ret_null = _make_panel(n_firms=300, n_years=20, premium=0.0, seed=10)
    sig_live, ret_live = _make_panel(n_firms=300, n_years=20, premium=0.10, seed=10)
    h_null = st.quintile_hedge(sig_null, ret_null)["hedge"].mean()
    h_live = st.quintile_hedge(sig_live, ret_live)["hedge"].mean()
    assert h_live > h_null + 0.01


def test_market_annual_returns_series(null_panel):
    _, fwd_ret, _ = null_panel
    m = st.market_annual(fwd_ret)
    assert isinstance(m, pd.Series)
    assert len(m) == len(fwd_ret)
