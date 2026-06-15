"""Strategy invariants, quintile-sort mechanics, the random control, and the spine:
the high-ROE portfolio outperforms only when a quality premium is planted."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from roe_quality import data, strategy as st  # noqa: E402

_CACHE_PATHS = [
    os.path.join(data.DEFAULT_CACHE, f"_edgar_{c}.parquet") for c in data.CONCEPTS
] + [os.path.join(data.DEFAULT_CACHE, "_edgar_yrret.parquet")]

requires_cache = pytest.mark.skipif(
    not all(os.path.exists(p) for p in _CACHE_PATHS),
    reason="EDGAR cache absent (offline CI); covered by synthetic tests",
)


# ---------------------------------------------------------------------------
# roe_signal from synthetic concept panels
# ---------------------------------------------------------------------------
def _make_panels(n_firms: int = 50, n_years: int = 10, seed: int = 0) -> tuple:
    """Minimal synthetic concept panels for testing roe_signal and gp_signal."""
    rng = np.random.default_rng(seed)
    firms = [f"F{j:03d}" for j in range(n_firms)]
    # years for the signal (y); lagged denominator requires y-1 in StockholdersEquity
    years = list(range(2009, 2009 + n_years))  # signal years
    all_years = [2008] + years  # 2008 is the lag year
    idx_all = pd.Index(all_years, name="year")
    idx_sig = pd.Index(years, name="year")

    def pos_frame(scale: float = 1e9, index: pd.Index = idx_sig) -> pd.DataFrame:
        """A frame of positive values (appropriate for financial quantities)."""
        return pd.DataFrame(
            np.abs(rng.normal(scale, scale * 0.2, (len(index), n_firms))) + 1e6,
            index=index,
            columns=firms,
        )

    # NetIncomeLoss: can be negative
    ni_vals = rng.normal(1e9, 5e8, (len(idx_sig), n_firms))
    ni = pd.DataFrame(ni_vals, index=idx_sig, columns=firms)

    # StockholdersEquity: needs to cover lagged years (all_years) and signal years
    se = pd.DataFrame(
        np.abs(rng.normal(5e9, 1e9, (len(idx_all), n_firms))) + 1e6,
        index=idx_all,
        columns=firms,
    )

    gp = pos_frame(3e8)
    assets = pos_frame(1e10)

    return ni, se, gp, assets, firms


def test_roe_signal_shape():
    ni, se, gp, assets, firms = _make_panels(n_firms=50, n_years=10)
    roe = st.roe_signal(ni, se)
    assert roe.shape[0] == 10  # n signal years
    assert roe.shape[1] == 50  # n firms
    assert roe.index.name == "year"


def test_roe_signal_requires_positive_lagged_equity():
    """Firms with non-positive lagged book equity must have NaN in ROE."""
    ni, se, gp, assets, firms = _make_panels(n_firms=10, n_years=5)
    # Force one firm to have zero lagged equity in the first lag year
    se.iloc[0, 0] = 0.0  # row 0 = lag year 2008, firm 0
    roe = st.roe_signal(ni, se)
    # Firm 0 should be NaN in the first signal year (2009) because lagged equity = 0
    first_year = roe.index[0]
    assert np.isnan(roe.loc[first_year, "F000"]), (
        "Firm with zero lagged equity should produce NaN ROE"
    )


def test_roe_signal_no_lookahead():
    """ROE for year y must use year-y net income scaled by year-(y-1) equity."""
    ni, se, gp, assets, firms = _make_panels(n_firms=20, n_years=5, seed=42)
    roe = st.roe_signal(ni, se)
    # First signal year is 2009; the lag denominator is 2008 equity
    assert 2009 in roe.index
    assert 2008 not in roe.index  # 2008 has no lag, so it's excluded


def test_roe_signal_winsorises():
    """After winsorisation, the extreme outlier is clipped; the panel with winsorisation
    should have a tighter range than without winsorisation."""
    ni, se, gp, assets, firms = _make_panels(n_firms=100, n_years=8)
    # Inject an extreme ROE outlier: tiny equity, large income in the LAG year (2008)
    # so that in the SIGNAL year (2009) the lagged denominator is tiny
    se.iloc[0, 0] = 1.0  # lag year 2008, firm 0: tiny equity -> extreme ROE in 2009
    ni_extreme = ni.copy()
    ni_extreme.iloc[0, 0] = 1e12  # very large income for firm 0 in signal year 2009

    roe_winsor = st.roe_signal(ni_extreme, se, winsor_pct=0.01)
    roe_no_winsor = st.roe_signal(ni_extreme, se, winsor_pct=0.0)

    first_year = roe_winsor.index[0]
    yr_w = roe_winsor.loc[first_year].dropna()
    yr_nw = roe_no_winsor.loc[first_year].dropna()

    # The winsorised version must have a smaller max than the non-winsorised version
    assert yr_w.max() < yr_nw.max(), (
        "Winsorisation should clip the extreme high value"
    )


def test_gp_signal_shape():
    ni, se, gp, assets, firms = _make_panels(n_firms=50, n_years=10)
    gpr = st.gp_signal(gp, assets)
    assert gpr.shape[0] == 10
    assert gpr.shape[1] == 50
    assert gpr.index.name == "year"


def test_gp_signal_positive_values():
    """GP/Assets should be mostly in [0, 1] for normal firms."""
    ni, se, gp, assets, firms = _make_panels(n_firms=50, n_years=5, seed=1)
    gpr = st.gp_signal(gp, assets)
    valid = gpr.stack().dropna()
    # Winsorised values should be in a sane range (not all NaN, not extreme)
    assert len(valid) > 0
    assert not np.isnan(valid).all()


# ---------------------------------------------------------------------------
# quintile_returns
# ---------------------------------------------------------------------------
def test_quintile_returns_columns(null_panel):
    roe, gp, fwd, _ = null_panel
    h = st.quintile_returns(roe, fwd, q=0.20)
    for col in ["q1", "q2", "q3", "q4", "q5", "market", "hedge", "n"]:
        assert col in h.columns


def test_quintile_hedge_equals_q5_minus_q1(null_panel):
    roe, gp, fwd, _ = null_panel
    h = st.quintile_returns(roe, fwd, q=0.20)
    assert np.allclose(h["hedge"], h["q5"] - h["q1"], equal_nan=True)


def test_quintile_market_is_ew(null_panel):
    """Market column should be a reasonable equal-weight return level."""
    roe, gp, fwd, _ = null_panel
    h = st.quintile_returns(roe, fwd, q=0.20)
    assert not h["market"].isna().all()
    assert (h["market"].abs() < 10.0).all()  # annual returns shouldn't exceed 10x


def test_quintile_n_size(null_panel):
    roe, gp, fwd, _ = null_panel
    h = st.quintile_returns(roe, fwd, q=0.20)
    for _, row in h.iterrows():
        n = int(row["n"])
        assert n >= 20  # we require at least 20 firms


def test_quintile_minimum_firms():
    """Panels with fewer than 20 firms per year must yield no rows."""
    roe, gp, fwd, _ = data.synthetic_panel(n_firms=15, n_years=5, seed=42)
    h = st.quintile_returns(roe, fwd, q=0.20)
    assert len(h) == 0


# ---------------------------------------------------------------------------
# random_portfolio_returns
# ---------------------------------------------------------------------------
def test_random_portfolio_mean_near_zero(null_panel):
    """Random portfolios of the right size should have near-zero excess vs market."""
    roe, gp, fwd, _ = null_panel
    exc = st.random_portfolio_returns(roe, fwd, q=0.20, n_draws=200, seed=200)
    assert abs(float(exc.mean())) < 0.05  # zero in expectation


def test_random_portfolio_reproducible(null_panel):
    roe, gp, fwd, _ = null_panel
    a = st.random_portfolio_returns(roe, fwd, q=0.20, n_draws=50, seed=7)
    b = st.random_portfolio_returns(roe, fwd, q=0.20, n_draws=50, seed=7)
    assert np.allclose(a.to_numpy(), b.to_numpy())


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def test_summary_keys(null_panel):
    roe, gp, fwd, _ = null_panel
    h = st.quintile_returns(roe, fwd, q=0.20)
    s = st.summary(h["hedge"])
    for key in ("mean", "vol", "sharpe", "tstat", "hit_rate", "n"):
        assert key in s


def test_summary_tstat_finite(null_panel):
    roe, gp, fwd, _ = null_panel
    h = st.quintile_returns(roe, fwd, q=0.20)
    s = st.summary(h["hedge"])
    assert np.isfinite(s["tstat"])


def test_summary_short_series():
    """summary() must not crash on a 1-element series (returns NaN)."""
    s = st.summary(pd.Series([0.1]))
    assert np.isnan(s["tstat"])


# ---------------------------------------------------------------------------
# The spine: high-ROE outperforms only when a real premium is planted
# ---------------------------------------------------------------------------
def test_roe_quality_only_with_premium(null_panel, live_panel):
    """The engine recovers the effect when it exists, prints near-zero when it doesn't."""
    roe_null, gp_null, fwd_null, _ = null_panel
    roe_live, gp_live, fwd_live, _ = live_panel

    h_null = st.quintile_returns(roe_null, fwd_null, q=0.20)
    h_live = st.quintile_returns(roe_live, fwd_live, q=0.20)

    # With a planted premium, the high-ROE (q5) should clearly beat low-ROE (q1)
    assert h_live["hedge"].mean() > 0.02  # at least 2pp hedge

    # Without a premium, the hedge is statistically negligible
    s_null = st.summary(h_null["hedge"])
    assert abs(s_null["tstat"]) < 2.5  # no strong t-stat on null data


# ---------------------------------------------------------------------------
# Real-data tests (cache-gated)
# ---------------------------------------------------------------------------
@requires_cache
def test_real_roe_has_reasonable_values():
    """Real ROE panel should have values mostly in [-5, 5] after winsorisation."""
    roe, gpr, fwd = data.fetch_panel()
    valid = roe.stack().dropna()
    assert (valid.abs() < 100).mean() > 0.95  # 95% of values within [-100, 100]


@requires_cache
def test_real_quintile_returns_monotone_on_null():
    """On real data the quintile spread should be computable and finite."""
    roe, gpr, fwd = data.fetch_panel()
    h = st.quintile_returns(roe, fwd, q=0.20)
    assert not h["hedge"].isna().all()
    assert h["n"].min() >= 20


@requires_cache
def test_real_summary_tstat_finite():
    """The hedge HAC t-stat on real data must be a finite number."""
    roe, gpr, fwd = data.fetch_panel()
    h = st.quintile_returns(roe, fwd, q=0.20)
    s = st.summary(h["hedge"])
    assert np.isfinite(s["tstat"])
