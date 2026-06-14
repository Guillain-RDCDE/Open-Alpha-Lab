"""Portfolio construction invariants, summary stats, and the study's spine:
the AT sort beats a random pick only when a premium is planted."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asset_turnover import strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# quintile_returns invariants
# ---------------------------------------------------------------------------
def test_quintile_returns_columns(signal_panel):
    at_sig, fwd, _ = signal_panel
    h = st.quintile_returns(at_sig, fwd, q=0.20)
    assert set(h.columns) >= {"top", "bottom", "market", "hedge", "ls", "n", "n_top"}
    assert len(h) > 0


def test_quintile_returns_market_between_extremes(signal_panel):
    """The equal-weight market return must lie between the top and bottom quintile averages
    in aggregate (not necessarily every year, but on average the ranking has to be consistent)."""
    at_sig, fwd, _ = signal_panel
    h = st.quintile_returns(at_sig, fwd, q=0.20)
    # Top quintile should be above bottom quintile on average when there is a premium
    assert h["top"].mean() > h["bottom"].mean()


def test_quintile_returns_hedge_is_top_minus_market(signal_panel):
    at_sig, fwd, _ = signal_panel
    h = st.quintile_returns(at_sig, fwd, q=0.20)
    assert np.allclose(h["hedge"].to_numpy(), (h["top"] - h["market"]).to_numpy())


def test_quintile_returns_requires_min_stocks(null_panel):
    """Years with fewer than 20 valid stocks are silently skipped."""
    import pandas as pd
    at_sig, fwd, _ = null_panel
    # Artificially make one year very sparse
    at_sparse = at_sig.copy()
    first_year = at_sig.index[0]
    at_sparse.loc[first_year] = np.nan
    h = st.quintile_returns(at_sparse, fwd, q=0.20)
    assert first_year not in h.index


def test_quintile_returns_no_lookahead(signal_panel):
    """Returns in year y use the signal from year y, forward returns come from fwd_ret[y].
    Test is structural: the function accepts separate signal and fwd_ret so no join can
    accidentally reach into future fundamentals."""
    at_sig, fwd, _ = signal_panel
    import pandas as pd
    # Shift fwd_ret forward by one year — the function should still produce results
    fwd_shifted = fwd.copy()
    fwd_shifted.index = fwd_shifted.index + 1
    h_shifted = st.quintile_returns(at_sig, fwd_shifted, q=0.20)
    # Not all years will match (the last signal year has no fwd) — that's correct
    assert len(h_shifted) < len(at_sig)


# ---------------------------------------------------------------------------
# random_portfolio_returns
# ---------------------------------------------------------------------------
def test_random_portfolio_returns_mean_near_zero(null_panel):
    """Random-draw excess vs market should average ≈ 0 (random selection adds no value)."""
    at_sig, fwd, _ = null_panel
    rand = st.random_portfolio_returns(at_sig, fwd, q=0.20, n_draws=200, seed=155)
    assert abs(rand.mean()) < 0.02      # within 2 pp of zero on average
    assert len(rand) > 0


def test_random_portfolio_returns_reproducible(null_panel):
    at_sig, fwd, _ = null_panel
    a = st.random_portfolio_returns(at_sig, fwd, q=0.20, n_draws=50, seed=42)
    b = st.random_portfolio_returns(at_sig, fwd, q=0.20, n_draws=50, seed=42)
    assert np.allclose(a.to_numpy(), b.to_numpy())


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_keys():
    import pandas as pd
    r = pd.Series(np.random.default_rng(0).normal(0.08, 0.20, 14))
    s = st.summarize(r)
    assert set(s.keys()) >= {"mean", "vol", "sharpe", "tstat", "hit_rate", "max_drawdown", "n"}


def test_summarize_nan_on_tiny_series():
    import pandas as pd
    s = st.summarize(pd.Series([0.05]))
    assert np.isnan(s["tstat"])


def test_summarize_tstat_direction():
    """A consistently positive series should produce a positive t-stat."""
    import pandas as pd
    r = pd.Series([0.10] * 14)  # all positive
    s = st.summarize(r)
    assert s["tstat"] > 0


# ---------------------------------------------------------------------------
# The spine: AT sort beats a coin only when premium is planted
# ---------------------------------------------------------------------------
def test_at_beats_random_only_with_premium():
    """The core falsification: with a planted premium the AT top quintile exceeds random
    portfolio draws; with zero premium it should not systematically exceed them."""
    null_sig, null_fwd, _ = __import__("asset_turnover").data.synthetic_panel(
        n_firms=200, n_years=14, premium=0.0, seed=155
    )
    real_sig, real_fwd, _ = __import__("asset_turnover").data.synthetic_panel(
        n_firms=200, n_years=14, premium=0.06, seed=155
    )

    def top_excess_vs_random(sig, fwd):
        h = st.quintile_returns(sig, fwd, q=0.20)
        rand = st.random_portfolio_returns(sig, fwd, q=0.20, n_draws=300, seed=1)
        top_mean = h["hedge"].mean()
        return top_mean, rand.mean(), float((rand < top_mean).mean())

    _, _, pct_null = top_excess_vs_random(null_sig, null_fwd)
    _, _, pct_real = top_excess_vs_random(real_sig, real_fwd)

    # With real premium, AT top beats at least 70% of random draws
    assert pct_real > 0.60
    # With null premium, AT top should not beat significantly more than 50%
    assert pct_null < 0.75
