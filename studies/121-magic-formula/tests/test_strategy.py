"""The strategy invariants, the hedge mechanics, the random control, and the study's spine:
the Magic Formula top decile beats a random draw *only when a real premium is planted*."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from magic_formula import strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# magic_formula_rank from synthetic concept panels
# ---------------------------------------------------------------------------
def _make_panels(n_firms=50, n_years=10, seed=0):
    """Create minimal synthetic concept panels for testing magic_formula_rank."""
    rng = np.random.default_rng(seed)
    firms = [f"F{j:03d}" for j in range(n_firms)]
    years = list(range(2010, 2010 + n_years))
    idx = pd.Index(years, name="year")

    def frame(scale=1.0, offset=0.0):
        return pd.DataFrame(
            np.abs(rng.normal(scale, scale * 0.2, (n_years, n_firms))) + offset,
            index=idx, columns=firms,
        )

    return {
        "OperatingIncomeLoss": frame(1e9),
        "Assets": frame(1e10, offset=1e9),
        "LiabilitiesCurrent": frame(2e9),
        "LongTermDebtNoncurrent": frame(1e9),
        "CashAndCashEquivalentsAtCarryingValue": frame(5e8),
        "StockholdersEquity": frame(3e9, offset=1e9),
    }


def test_magic_formula_rank_shape():
    panels = _make_panels(n_firms=50, n_years=10)
    rank = st.magic_formula_rank(panels)
    assert rank.shape[0] == 10
    assert rank.shape[1] == 50
    assert rank.index.name == "year"


def test_magic_formula_rank_monotone():
    """Firms with higher ROC and higher EY should have higher combined rank."""
    rng = np.random.default_rng(7)
    n, m = 5, 20
    firms = [f"F{j}" for j in range(m)]
    years = list(range(2010, 2010 + n))
    idx = pd.Index(years, name="year")

    assets_val = np.ones((n, m)) * 1e10
    liab_val = np.ones((n, m)) * 2e9
    cash_val = np.ones((n, m)) * 5e8
    eq_val = np.ones((n, m)) * 3e9
    ltd_val = np.ones((n, m)) * 1e9
    # EBIT increases monotonically across firms — firm m-1 should always rank highest
    ebit_val = np.outer(np.ones(n), np.linspace(0.5e9, 3e9, m))

    panels = {
        "OperatingIncomeLoss": pd.DataFrame(ebit_val, index=idx, columns=firms),
        "Assets": pd.DataFrame(assets_val, index=idx, columns=firms),
        "LiabilitiesCurrent": pd.DataFrame(liab_val, index=idx, columns=firms),
        "LongTermDebtNoncurrent": pd.DataFrame(ltd_val, index=idx, columns=firms),
        "CashAndCashEquivalentsAtCarryingValue": pd.DataFrame(cash_val, index=idx, columns=firms),
        "StockholdersEquity": pd.DataFrame(eq_val, index=idx, columns=firms),
    }
    rank = st.magic_formula_rank(panels)
    # Every year, the last firm should have the highest rank
    for y in years:
        row = rank.loc[y].dropna()
        assert row.idxmax() == firms[-1], f"year {y}: {row.idxmax()} != {firms[-1]}"


def test_magic_formula_rank_no_negative_ic():
    """Firms with negative invested capital must be excluded (rank NaN)."""
    panels = _make_panels(n_firms=10, n_years=5)
    # Force one firm to have very large liabilities → negative IC
    panels["LiabilitiesCurrent"].iloc[:, 0] = 1e12
    rank = st.magic_formula_rank(panels)
    # Firm 0 should be NaN in every year
    assert rank.iloc[:, 0].isna().all(), "Firm with negative IC should have NaN rank"


# ---------------------------------------------------------------------------
# quantile_hedge
# ---------------------------------------------------------------------------
def test_quantile_hedge_columns(null_panel):
    sig, fwd, _ = null_panel
    h = st.quantile_hedge(sig, fwd, q=0.10)
    assert set(["top", "bottom", "market", "hedge", "n", "n_top", "n_bot"]).issubset(h.columns)


def test_quantile_hedge_n_top_size(null_panel):
    sig, fwd, _ = null_panel
    h = st.quantile_hedge(sig, fwd, q=0.10)
    # Top-decile count should be approximately 10% of n
    for _, row in h.iterrows():
        n, n_top = int(row["n"]), int(row["n_top"])
        assert n_top >= max(1, int(0.08 * n))
        assert n_top <= int(0.15 * n) + 2


def test_quantile_hedge_hedge_equals_top_minus_market(null_panel):
    sig, fwd, _ = null_panel
    h = st.quantile_hedge(sig, fwd, q=0.10)
    assert np.allclose(h["hedge"], h["top"] - h["market"], equal_nan=True)


# ---------------------------------------------------------------------------
# random_portfolio_returns
# ---------------------------------------------------------------------------
def test_random_portfolio_mean_near_zero(null_panel):
    """Random portfolios of the right size should have near-zero excess vs market."""
    sig, fwd, _ = null_panel
    exc = st.random_portfolio_returns(sig, fwd, q=0.10, n_draws=200, seed=121)
    assert abs(float(exc.mean())) < 0.05  # zero in expectation


def test_random_portfolio_reproducible(null_panel):
    sig, fwd, _ = null_panel
    a = st.random_portfolio_returns(sig, fwd, q=0.10, n_draws=50, seed=7)
    b = st.random_portfolio_returns(sig, fwd, q=0.10, n_draws=50, seed=7)
    assert np.allclose(a.to_numpy(), b.to_numpy())


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------
def test_summary_keys(null_panel):
    sig, fwd, _ = null_panel
    h = st.quantile_hedge(sig, fwd, q=0.10)
    s = st.summary(h["hedge"])
    for key in ("mean", "vol", "sharpe", "tstat", "hit_rate", "n"):
        assert key in s


def test_summary_tstat_finite(null_panel):
    sig, fwd, _ = null_panel
    h = st.quantile_hedge(sig, fwd, q=0.10)
    s = st.summary(h["hedge"])
    assert np.isfinite(s["tstat"])


# ---------------------------------------------------------------------------
# The spine: top decile only outperforms when a real premium is planted
# ---------------------------------------------------------------------------
def test_mf_beats_random_only_with_premium(null_panel, live_panel):
    """The engine recovers the effect when it exists, and prints near-zero when it doesn't."""
    sig_null, fwd_null, _ = null_panel
    sig_live, fwd_live, _ = live_panel

    h_null = st.quantile_hedge(sig_null, fwd_null, q=0.10)
    h_live = st.quantile_hedge(sig_live, fwd_live, q=0.10)

    # With a strong planted premium, top decile clearly beats market
    assert h_live["hedge"].mean() > 0.02  # at least 2pp excess

    # Without a premium, the excess is tiny and statistically meaningless
    s_null = st.summary(h_null["hedge"])
    assert abs(s_null["tstat"]) < 2.5  # no strong t-stat on null data
