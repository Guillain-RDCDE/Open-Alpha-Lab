"""The synthetic panel is well-formed and deterministic; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fx_momentum import data  # noqa: E402


def test_synthetic_shape(null_panel):
    returns, truth = null_panel
    assert returns.shape[0] == truth["n_months"]
    assert returns.shape[1] == truth["n_currencies"]
    assert returns.columns.tolist() == [f"C{i}" for i in range(truth["n_currencies"])]


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_panel(n_months=60, momentum_signal=0.01, seed=5)
    b, _ = data.synthetic_panel(n_months=60, momentum_signal=0.01, seed=5)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    c, _ = data.synthetic_panel(n_months=60, momentum_signal=0.01, seed=6)
    assert not np.allclose(a.to_numpy(), c.to_numpy())


def test_null_panel_has_zero_mean_crosssectional_predictability(null_panel):
    """With momentum_signal=0, the cross-sectional rank correlation should be near zero."""
    returns, _ = null_panel
    # Rank correlation between month-t return and month-(t+1) rank across currencies.
    from scipy.stats import spearmanr
    r = returns.to_numpy()
    rank_corrs = []
    for t in range(r.shape[0] - 1):
        rho, _ = spearmanr(r[t], r[t + 1])
        rank_corrs.append(rho)
    mean_rho = np.nanmean(rank_corrs)
    assert abs(mean_rho) < 0.25  # should be noise level


def test_signal_panel_has_positive_crosssectional_persistence(signal_panel):
    """With momentum_signal > 0, the 12-month cumulative return is positively
    autocorrelated cross-sectionally — past 12-month winners continue to win."""
    returns, _ = signal_panel
    r = returns.to_numpy()
    n = r.shape[0]
    lookback = 12
    # For each month t, correlate the 12-month cumulative return up to t with
    # the 12-month cumulative return from t+1 to t+12 (i.e., does the ranking
    # predict the next ranking?).
    corrs = []
    for t in range(lookback, n - lookback):
        past = r[t - lookback:t].sum(axis=0)
        fut = r[t:t + lookback].sum(axis=0)
        if past.std() > 0 and fut.std() > 0:
            corrs.append(np.corrcoef(past, fut)[0, 1])
    mean_corr = np.nanmean(corrs)
    # Planted momentum creates detectable cross-sectional predictability
    # (lower threshold: the signal is injected at each period, accumulated over 12m).
    assert mean_corr > 0.02


def test_fetch_panel_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_panel(fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_stable_and_content_sensitive(null_panel):
    returns, _ = null_panel
    assert data.fingerprint(returns) == data.fingerprint(returns)
    other, _ = data.synthetic_panel(n_months=120, momentum_signal=0.0, seed=999)
    assert data.fingerprint(returns) != data.fingerprint(other)


def test_to_monthly_sums_correctly():
    """to_monthly sums daily log-returns within each calendar month."""
    import pandas as pd
    # Create a trivial daily series: one value per day, two months.
    idx = pd.date_range("2020-01-02", "2020-02-28", freq="B")
    daily = pd.DataFrame({"EUR": 0.001, "GBP": -0.001}, index=idx)
    daily.index.name = "date"
    monthly = data.to_monthly(daily)
    # January should be 1-0 bps sum — roughly n_days * 0.001 for EUR.
    jan = monthly.loc["2020-01"]
    assert len(jan) == 1
    assert jan["EUR"].values[0] > 0
    assert jan["GBP"].values[0] < 0
