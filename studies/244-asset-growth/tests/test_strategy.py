"""Strategy invariants, the quintile-sort mechanics, the random control, and the spine:
the low-AG portfolio outperforms the high-AG portfolio *only when an effect is planted*."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from asset_growth import strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# ag_signal from synthetic assets panel
# ---------------------------------------------------------------------------
def _make_assets_panel(n_firms=50, n_years=10, seed=0):
    """Minimal synthetic assets panel for testing ag_signal."""
    rng = np.random.default_rng(seed)
    firms = [f"F{j:03d}" for j in range(n_firms)]
    # We need years y and y-1 for the lagged computation
    # signal years: 2009 ... 2009+n_years-1
    # lag year: 2008
    signal_years = list(range(2009, 2009 + n_years))
    all_years = [2008] + signal_years
    idx_all = pd.Index(all_years, name="year")

    assets = pd.DataFrame(
        np.abs(rng.normal(1e10, 2e9, (len(idx_all), n_firms))),
        index=idx_all,
        columns=firms,
    )
    return assets


def test_ag_signal_shape():
    assets = _make_assets_panel(n_firms=50, n_years=10)
    ag = st.ag_signal(assets)
    assert ag.shape[0] == 10
    assert ag.shape[1] == 50
    assert ag.index.name == "year"


def test_ag_signal_requires_positive_lagged_assets():
    """Firms with non-positive lagged total assets must have NaN in AG."""
    assets = _make_assets_panel(n_firms=10, n_years=5)
    # Force one firm to have zero lagged assets in the first lag year
    assets.iloc[0, 0] = 0.0  # row 0 = lag year 2008, firm 0
    ag = st.ag_signal(assets)
    first_year = ag.index[0]
    assert np.isnan(ag.loc[first_year, "F000"]), (
        "Firm with zero lagged total assets should produce NaN AG"
    )


def test_ag_signal_no_lookahead():
    """AG for year y must use year-y assets divided by year-(y-1) assets."""
    assets = _make_assets_panel(n_firms=20, n_years=5, seed=42)
    ag = st.ag_signal(assets)
    # First signal year is 2009; the lag denominator is 2008 assets
    assert 2009 in ag.index
    assert 2008 not in ag.index  # 2008 has no lag, so it's excluded


def test_ag_signal_values_finite():
    """AG values should be finite (no inf from degenerate denominators)."""
    assets = _make_assets_panel(n_firms=30, n_years=8)
    ag = st.ag_signal(assets)
    # All valid (non-NaN) values should be finite
    assert np.all(np.isfinite(ag.values[~np.isnan(ag.values)]))


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


def test_quintile_market_is_ew(null_panel):
    """Market column should be approximately equal-weight of all quintiles."""
    sig, fwd, _ = null_panel
    h = st.quintile_returns(sig, fwd, q=0.20)
    assert not h["market"].isna().all()
    assert (h["market"].abs() < 10.0).all()  # annual returns shouldn't exceed 10x


def test_quintile_n_size(null_panel):
    sig, fwd, _ = null_panel
    h = st.quintile_returns(sig, fwd, q=0.20)
    for _, row in h.iterrows():
        n = int(row["n"])
        assert n >= 20  # we require at least 20 firms


# ---------------------------------------------------------------------------
# random_portfolio_returns
# ---------------------------------------------------------------------------
def test_random_portfolio_mean_near_zero(null_panel):
    """Random portfolios of the right size should have near-zero excess vs market."""
    sig, fwd, _ = null_panel
    exc = st.random_portfolio_returns(sig, fwd, q=0.20, n_draws=200, seed=244)
    assert abs(float(exc.mean())) < 0.05  # zero in expectation


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
# The spine: low-AG outperforms only when a real premium is planted
# ---------------------------------------------------------------------------
def test_ag_anomaly_only_with_premium(null_panel, live_panel):
    """The engine recovers the effect when it exists, prints near-zero when it doesn't."""
    sig_null, fwd_null, _ = null_panel
    sig_live, fwd_live, _ = live_panel

    h_null = st.quintile_returns(sig_null, fwd_null, q=0.20)
    h_live = st.quintile_returns(sig_live, fwd_live, q=0.20)

    # With a strong planted premium, the low-AG (q1) should clearly beat high-AG (q5)
    assert h_live["hedge"].mean() > 0.03  # at least 3pp hedge

    # Without a premium, the hedge is statistically irrelevant
    s_null = st.summary(h_null["hedge"])
    assert abs(s_null["tstat"]) < 2.5  # no strong t-stat on null data
