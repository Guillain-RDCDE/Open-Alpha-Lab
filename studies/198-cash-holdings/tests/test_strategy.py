"""Strategy invariants, the quintile-sort mechanics, the random control, and the spine:
the high-cash portfolio outperforms the low-cash portfolio *only when an effect is planted*."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from cash_holdings import strategy as st  # noqa: E402

# ---------------------------------------------------------------------------
# Cache-gate for real-data tests
# ---------------------------------------------------------------------------
_CACHE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..", "_cache"
)
_EDGAR_ASSETS = os.path.join(_CACHE_DIR, "_edgar_Assets.parquet")
_EDGAR_CASH = os.path.join(_CACHE_DIR, "_edgar_CashAndCashEquivalentsAtCarryingValue.parquet")
_EDGAR_YRRET = os.path.join(_CACHE_DIR, "_edgar_yrret.parquet")

requires_cache = pytest.mark.skipif(
    not (
        os.path.exists(_EDGAR_ASSETS)
        and os.path.exists(_EDGAR_CASH)
        and os.path.exists(_EDGAR_YRRET)
    ),
    reason="EDGAR cache absent (offline CI); covered by synthetic tests",
)


# ---------------------------------------------------------------------------
# cash_ratio_signal — unit tests with minimal synthetic concept panels
# ---------------------------------------------------------------------------
def _make_concept_panels(n_firms=50, n_years=10, seed=0):
    """Minimal synthetic concept panels for testing cash_ratio_signal."""
    rng = np.random.default_rng(seed)
    firms = [f"F{j:03d}" for j in range(n_firms)]
    years = list(range(2009, 2009 + n_years))
    idx = pd.Index(years, name="year")

    assets = pd.DataFrame(
        np.abs(rng.normal(1e10, 2e9, (n_years, n_firms))),
        index=idx, columns=firms,
    )
    cash = pd.DataFrame(
        np.abs(rng.normal(5e8, 1e8, (n_years, n_firms))),
        index=idx, columns=firms,
    )
    return assets, cash


def test_cash_ratio_signal_shape():
    assets, cash = _make_concept_panels(n_firms=50, n_years=10)
    ratio = st.cash_ratio_signal(assets, cash)
    assert ratio.shape[0] == 10
    assert ratio.shape[1] == 50
    assert ratio.index.name == "year"


def test_cash_ratio_signal_bounded():
    """All valid ratios must be in [0, 1]."""
    assets, cash = _make_concept_panels(n_firms=100, n_years=15)
    ratio = st.cash_ratio_signal(assets, cash)
    vals = ratio.to_numpy().ravel()
    vals = vals[np.isfinite(vals)]
    assert (vals >= 0).all()
    assert (vals <= 1.0).all()


def test_cash_ratio_signal_zero_assets_gives_nan():
    """Firms with zero or negative total assets must have NaN ratio."""
    assets, cash = _make_concept_panels(n_firms=10, n_years=5, seed=42)
    assets.iloc[0, 0] = 0.0  # force one firm to have zero assets in first year
    ratio = st.cash_ratio_signal(assets, cash)
    first_year = ratio.index[0]
    assert np.isnan(ratio.loc[first_year, "F000"])


def test_cash_ratio_signal_gt1_clamped_to_nan():
    """Ratios > 1 must be clamped to NaN (data errors)."""
    assets, cash = _make_concept_panels(n_firms=10, n_years=5, seed=99)
    # Make cash > assets for first firm in first year
    first_yr = assets.index[0]
    assets.loc[first_yr, "F000"] = 1.0
    cash.loc[first_yr, "F000"] = 2.0
    ratio = st.cash_ratio_signal(assets, cash)
    assert np.isnan(ratio.loc[first_yr, "F000"])


def test_cash_ratio_signal_definition():
    """spot-check ratio = cash / assets for a known firm-year."""
    assets, cash = _make_concept_panels(n_firms=10, n_years=5, seed=7)
    ratio = st.cash_ratio_signal(assets, cash)
    # Pick a firm and year where assets > cash (ratio should be in (0,1))
    yr = ratio.index[2]
    ticker = ratio.columns[3]
    expected = float(cash.loc[yr, ticker]) / float(assets.loc[yr, ticker])
    if 0 <= expected <= 1:
        assert abs(float(ratio.loc[yr, ticker]) - expected) < 1e-10


# ---------------------------------------------------------------------------
# quintile_returns
# ---------------------------------------------------------------------------
def test_quintile_returns_columns(null_panel):
    sig, fwd, _ = null_panel
    h = st.quintile_returns(sig, fwd, q=0.20)
    for col in ["q1", "q2", "q3", "q4", "q5", "market", "hedge", "n"]:
        assert col in h.columns


def test_quintile_hedge_equals_q5_minus_q1(null_panel):
    sig, fwd, _ = null_panel
    h = st.quintile_returns(sig, fwd, q=0.20)
    # Palazzo hedge: long HIGH cash (q5), short LOW cash (q1)
    assert np.allclose(h["hedge"], h["q5"] - h["q1"], equal_nan=True)


def test_quintile_market_reasonable(null_panel):
    """Market column should be a realistic annual return level."""
    sig, fwd, _ = null_panel
    h = st.quintile_returns(sig, fwd, q=0.20)
    assert not h["market"].isna().all()
    assert (h["market"].abs() < 10.0).all()


def test_quintile_n_size(null_panel):
    """Each year's quintile row must have at least 20 firms."""
    sig, fwd, _ = null_panel
    h = st.quintile_returns(sig, fwd, q=0.20)
    for _, row in h.iterrows():
        n = int(row["n"])
        assert n >= 20


def test_quintile_requires_min_firms():
    """Years with fewer than 20 firms should be excluded from results."""
    sig, fwd, _ = __import__("cash_holdings").data.synthetic_panel(
        n_firms=10, n_years=5, premium=0.0, seed=0
    )
    h = st.quintile_returns(sig, fwd, q=0.20)
    assert len(h) == 0  # n=10 < 20 minimum


# ---------------------------------------------------------------------------
# random_portfolio_returns
# ---------------------------------------------------------------------------
def test_random_portfolio_mean_near_zero(null_panel):
    """Random portfolios of the right size should have near-zero excess vs market."""
    sig, fwd, _ = null_panel
    exc = st.random_portfolio_returns(sig, fwd, q=0.20, n_draws=200, seed=198)
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


def test_summary_short_series():
    """Single-element series should return NaN for most stats."""
    s = st.summary(pd.Series([0.05]))
    assert np.isnan(s["tstat"])


# ---------------------------------------------------------------------------
# The spine: high-cash outperforms only when a real premium is planted
# ---------------------------------------------------------------------------
def test_cash_anomaly_only_with_premium(null_panel, live_panel):
    """The engine recovers the effect when it exists, prints near-zero when it doesn't."""
    sig_null, fwd_null, _ = null_panel
    sig_live, fwd_live, _ = live_panel

    h_null = st.quintile_returns(sig_null, fwd_null, q=0.20)
    h_live = st.quintile_returns(sig_live, fwd_live, q=0.20)

    # With a strong planted premium, the high-cash (q5) should clearly beat low-cash (q1)
    assert h_live["hedge"].mean() > 0.03  # at least 3pp hedge

    # Without a premium, the hedge is statistically irrelevant
    s_null = st.summary(h_null["hedge"])
    assert abs(s_null["tstat"]) < 2.5  # no strong t-stat on null data


# ---------------------------------------------------------------------------
# Real-data spine test (cache-gated)
# ---------------------------------------------------------------------------
@requires_cache
def test_real_panel_loads_and_quintiles_run():
    """Real panel loads and produces a non-empty quintile-return table."""
    from cash_holdings import data
    sig, fwd = data.fetch_panel()
    assert not sig.empty
    h = st.quintile_returns(sig, fwd, q=0.20)
    assert not h.empty
    assert "hedge" in h.columns


@requires_cache
def test_real_panel_summary_is_finite():
    """Hedge summary statistics must all be finite on the real tape."""
    from cash_holdings import data
    sig, fwd = data.fetch_panel()
    h = st.quintile_returns(sig, fwd, q=0.20)
    s = st.summary(h["hedge"])
    assert np.isfinite(s["mean"])
    assert np.isfinite(s["tstat"])
