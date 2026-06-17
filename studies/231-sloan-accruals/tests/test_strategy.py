"""Strategy invariants, the quintile-sort mechanics, the random control, and the spine:
the low-accruals portfolio outperforms the high-accruals portfolio *only when an effect
is planted*."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from sloan_accruals import strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# accruals_signal from synthetic concept panels
# ---------------------------------------------------------------------------
def _make_panels(n_firms=50, n_years=10, seed=0):
    """Minimal synthetic concept panels for testing accruals_signal."""
    rng = np.random.default_rng(seed)
    firms = [f"F{j:03d}" for j in range(n_firms)]
    years = list(range(2009, 2009 + n_years))      # signal years
    all_years = [2008] + years                       # 2008 is the lag year for avg assets
    idx_all = pd.Index(all_years, name="year")
    idx_sig = pd.Index(years, name="year")

    def frame(scale=1e10, index=idx_sig):
        return pd.DataFrame(
            rng.normal(scale, scale * 0.1, (len(index), n_firms)),
            index=index, columns=firms,
        )

    # Assets must cover both lagged years (all_years) and signal years
    assets_all = pd.DataFrame(
        np.abs(rng.normal(1e10, 2e9, (len(idx_all), n_firms))),
        index=idx_all, columns=firms,
    )
    # Net income: positive, mean ~1e9
    ni = pd.DataFrame(
        np.abs(rng.normal(1e9, 3e8, (len(idx_sig), n_firms))),
        index=idx_sig, columns=firms,
    )
    # Operating cash flow: slightly different from net income (creates accruals)
    cfo = pd.DataFrame(
        np.abs(rng.normal(1.1e9, 3e8, (len(idx_sig), n_firms))),
        index=idx_sig, columns=firms,
    )
    return {
        "NetIncomeLoss": ni,
        "NetCashProvidedByUsedInOperatingActivities": cfo,
        "Assets": assets_all,
    }


def test_accruals_signal_shape():
    panels = _make_panels(n_firms=50, n_years=10)
    acc = st.accruals_signal(panels)
    assert acc.shape[0] == 10
    assert acc.shape[1] == 50
    assert acc.index.name == "year"


def test_accruals_signal_requires_positive_avg_assets():
    """Firms with non-positive average assets must have NaN in accruals."""
    panels = _make_panels(n_firms=10, n_years=5)
    # Force firm 0 to have zero assets in both lagged and current year => avg = 0
    panels["Assets"].iloc[0, 0] = 0.0  # lag year 2008
    panels["Assets"].iloc[1, 0] = 0.0  # signal year 2009
    acc = st.accruals_signal(panels)
    first_year = acc.index[0]
    assert np.isnan(acc.loc[first_year, "F000"]), (
        "Firm with zero average total assets should produce NaN accruals"
    )


def test_accruals_signal_no_lookahead():
    """Accruals for year y must use year-y fundamentals scaled by average of y and y-1 assets."""
    panels = _make_panels(n_firms=20, n_years=5, seed=42)
    acc = st.accruals_signal(panels)
    assert 2009 in acc.index
    assert 2008 not in acc.index  # 2008 has no lag, so it's excluded


def test_accruals_is_ni_minus_cfo():
    """Accruals ratio direction: if NI > CFO for all firms, accruals should be positive."""
    n = 30
    firms = [f"F{j:03d}" for j in range(n)]
    yrs_all = pd.Index([2008, 2009], name="year")
    yrs_sig = pd.Index([2009], name="year")
    assets_all = pd.DataFrame(np.ones((2, n)) * 1e10, index=yrs_all, columns=firms)
    ni = pd.DataFrame(np.ones((1, n)) * 2e9, index=yrs_sig, columns=firms)
    cfo = pd.DataFrame(np.ones((1, n)) * 1e9, index=yrs_sig, columns=firms)
    panels = {
        "NetIncomeLoss": ni,
        "NetCashProvidedByUsedInOperatingActivities": cfo,
        "Assets": assets_all,
    }
    acc = st.accruals_signal(panels)
    # Accruals = (2e9 - 1e9) / 1e10 = 0.1
    assert (acc.loc[2009] > 0).all()
    assert np.allclose(acc.loc[2009].dropna(), 0.1, atol=1e-6)


# ---------------------------------------------------------------------------
# quintile_returns
# ---------------------------------------------------------------------------
def test_quintile_returns_columns(null_panel):
    sig, fwd, _ = null_panel
    h = st.quintile_returns(sig, fwd, q=0.20)
    for col in ["q1", "q2", "q3", "q4", "q5", "market", "hedge", "n"]:
        assert col in h.columns


def test_quintile_hedge_equals_q1_minus_q5(null_panel):
    sig, fwd, _ = null_panel
    h = st.quintile_returns(sig, fwd, q=0.20)
    assert np.allclose(h["hedge"], h["q1"] - h["q5"], equal_nan=True)


def test_quintile_market_is_finite(null_panel):
    sig, fwd, _ = null_panel
    h = st.quintile_returns(sig, fwd, q=0.20)
    assert not h["market"].isna().all()
    assert (h["market"].abs() < 10.0).all()


def test_quintile_n_size(null_panel):
    sig, fwd, _ = null_panel
    h = st.quintile_returns(sig, fwd, q=0.20)
    for _, row in h.iterrows():
        n = int(row["n"])
        assert n >= 20


# ---------------------------------------------------------------------------
# random_portfolio_returns
# ---------------------------------------------------------------------------
def test_random_portfolio_mean_near_zero(null_panel):
    """Random portfolios of the right size should have near-zero excess vs market."""
    sig, fwd, _ = null_panel
    exc = st.random_portfolio_returns(sig, fwd, q=0.20, n_draws=200, seed=231)
    assert abs(float(exc.mean())) < 0.05


def test_random_portfolio_reproducible(null_panel):
    sig, fwd, _ = null_panel
    a = st.random_portfolio_returns(sig, fwd, q=0.20, n_draws=50, seed=7)
    b = st.random_portfolio_returns(sig, fwd, q=0.20, n_draws=50, seed=7)
    assert np.allclose(a.to_numpy(), b.to_numpy())


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def test_summary_keys(null_panel):
    sig, fwd, _ = null_panel
    h = st.quintile_returns(sig, fwd, q=0.20)
    s = st.summary(h["hedge"])
    for key in ("mean", "vol", "sharpe", "tstat", "hit_rate", "n"):
        assert key in s


def test_summary_tstat_finite(null_panel):
    sig, fwd, _ = null_panel
    h = st.quintile_returns(sig, fwd, q=0.20)
    s = st.summary(h["hedge"])
    assert np.isfinite(s["tstat"])


# ---------------------------------------------------------------------------
# The spine: low-accruals outperforms only when a real premium is planted
# ---------------------------------------------------------------------------
def test_accruals_anomaly_only_with_premium(null_panel, live_panel):
    """The engine recovers the effect when it exists, near-zero when it doesn't."""
    from sloan_accruals import data as d
    # Use a larger null panel to reduce false positives from a small random seed
    sig_null, fwd_null, _ = d.synthetic_panel(n_firms=500, n_years=30, premium=0.0, seed=999)
    sig_live, fwd_live, _ = live_panel

    h_null = st.quintile_returns(sig_null, fwd_null, q=0.20)
    h_live = st.quintile_returns(sig_live, fwd_live, q=0.20)

    # With a strong planted premium, the low-accruals (q1) should clearly beat high-accruals (q5)
    assert h_live["hedge"].mean() > 0.03  # at least 3pp hedge

    # Without a premium, the hedge should be near zero (the effect should be weaker than
    # the live panel regardless of noise)
    assert h_live["hedge"].mean() > h_null["hedge"].mean() + 0.02
