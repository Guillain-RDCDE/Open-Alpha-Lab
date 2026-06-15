"""Strategy invariants, the quintile-sort mechanics, the random control, and the spine:
the low-growth portfolio outperforms the high-growth portfolio *only when an effect is planted*."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from sales_growth import strategy as st  # noqa: E402

EDGAR_REVENUES = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "_cache", "_edgar_Revenues.parquet"
)
EDGAR_YRRET = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "_cache", "_edgar_yrret.parquet"
)

requires_cache = pytest.mark.skipif(
    not os.path.exists(EDGAR_REVENUES) or not os.path.exists(EDGAR_YRRET),
    reason="EDGAR cache absent (offline CI); covered by synthetic tests",
)


# ---------------------------------------------------------------------------
# sales_growth_signal from synthetic concept panels
# ---------------------------------------------------------------------------
def _make_revenues(n_firms=50, n_years=10, seed=0):
    """Minimal synthetic Revenues DataFrame for testing sales_growth_signal."""
    rng = np.random.default_rng(seed)
    firms = [f"F{j:03d}" for j in range(n_firms)]
    # Signal years; lag denominator requires y-1, so add one prior year
    years = list(range(2007, 2007 + n_years + 1))  # includes lag year 2007
    idx = pd.Index(years, name="year")
    return pd.DataFrame(
        np.abs(rng.normal(1e10, 2e9, (len(years), n_firms))),
        index=idx,
        columns=firms,
    )


def test_sales_growth_signal_shape():
    rev = _make_revenues(n_firms=50, n_years=10)
    sg = st.sales_growth_signal(rev)
    # Should have n_years rows (one per signal year, skipping the lag year)
    assert sg.shape[0] == 10
    assert sg.shape[1] == 50
    assert sg.index.name == "year"


def test_sales_growth_signal_no_lookahead():
    """Growth for year y must use year-(y-1) as the denominator."""
    rev = _make_revenues(n_firms=20, n_years=5, seed=42)
    sg = st.sales_growth_signal(rev)
    # First signal year = 2008 (needs 2007 as lag, which is present)
    assert 2008 in sg.index
    # 2007 should not be in sg (no lag year 2006 in our frame)
    assert 2007 not in sg.index


def test_sales_growth_requires_positive_prior_revenues():
    """Firms with non-positive prior revenues must produce NaN growth."""
    rev = _make_revenues(n_firms=10, n_years=5)
    # Force one firm to have zero revenues in the lag year (2007 -> row 0)
    rev.iloc[0, 0] = 0.0
    sg = st.sales_growth_signal(rev)
    first_signal_year = sg.index[0]
    assert np.isnan(sg.loc[first_signal_year, "F000"]), (
        "Firm with zero prior revenues should produce NaN sales growth"
    )


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


def test_quintile_market_is_reasonable(null_panel):
    """Market column should be at a plausible annual-return level."""
    sig, fwd, _ = null_panel
    h = st.quintile_returns(sig, fwd, q=0.20)
    assert not h["market"].isna().all()
    assert (h["market"].abs() < 10.0).all()  # annual returns not exceeding 10x


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
    exc = st.random_portfolio_returns(sig, fwd, q=0.20, n_draws=200, seed=199)
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
# The spine: low-growth outperforms only when a real premium is planted
# ---------------------------------------------------------------------------
def test_growth_anomaly_only_with_premium(null_panel, live_panel):
    """The engine recovers the effect when it exists, prints near-zero when it doesn't."""
    sig_null, fwd_null, _ = null_panel
    sig_live, fwd_live, _ = live_panel

    h_null = st.quintile_returns(sig_null, fwd_null, q=0.20)
    h_live = st.quintile_returns(sig_live, fwd_live, q=0.20)

    # With a strong planted premium, the low-growth (q1) should clearly beat high-growth (q5)
    assert h_live["hedge"].mean() > 0.03  # at least 3pp hedge

    # Without a premium, the hedge is not strongly one-directional
    s_null = st.summary(h_null["hedge"])
    assert abs(s_null["tstat"]) < 3.0  # no strong t-stat on null data


# ---------------------------------------------------------------------------
# Real-tape tests (cache-gated)
# ---------------------------------------------------------------------------
@requires_cache
def test_real_quintile_returns_non_empty():
    from sales_growth import data
    sig, fwd = data.fetch_panel()
    h = st.quintile_returns(sig, fwd, q=0.20)
    assert not h.empty
    assert len(h) >= 10  # at least 10 years of data


@requires_cache
def test_real_summary_has_finite_tstat():
    from sales_growth import data
    sig, fwd = data.fetch_panel()
    h = st.quintile_returns(sig, fwd, q=0.20)
    s = st.summary(h["hedge"])
    assert np.isfinite(s["tstat"])
