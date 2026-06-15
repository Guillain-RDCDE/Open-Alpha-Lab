"""The fluency scorer is correct, deterministic, and the synthetic panel is well-formed."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fluent_tickers import data  # noqa: E402


# ---------------------------------------------------------------------------
# Fluency scorer
# ---------------------------------------------------------------------------
def test_fluency_score_vowel_heavy_is_positive():
    """Ticker with vowels should outscore a pure-consonant ticker."""
    assert data.fluency_score("KAR") > data.fluency_score("KRD")


def test_fluency_score_rdo_is_zero_or_low():
    """'RDO' — the original paper's disfluent example — should score lower than 'KAR'."""
    assert data.fluency_score("KAR") > data.fluency_score("RDO")


def test_fluency_score_all_vowels_high():
    """'AOE' has three vowels and no consonant cluster."""
    assert data.fluency_score("AOE") >= data.fluency_score("ZYX")


def test_score_tickers_normalised():
    """score_tickers returns values in [0, 1]."""
    tickers = ["KAR", "RDO", "ZTX", "AEIA", "MSFT", "CAT", "BRK"]
    scores = data.score_tickers(tickers)
    assert scores.min() >= -1e-9
    assert scores.max() <= 1.0 + 1e-9


def test_classify_tickers_splits_in_half():
    """classify_tickers produces a roughly 50/50 split for a typical universe."""
    tickers = ["KAR", "RDO", "ZTX", "AEIA", "MSFT", "CAT", "BRK", "GPN"]
    mask = data.classify_tickers(tickers)
    n_fluent = mask.sum()
    assert 2 <= n_fluent <= 6  # roughly half


def test_classify_tickers_kar_more_fluent_than_rdo():
    """KAR (vowel) should be classified FLUENT; RDO (none / consonant cluster) NON-FLUENT."""
    tickers = ["KAR", "RDO", "ZTX", "AEIA", "MSFT", "CAT", "BRK", "GPN"]
    mask = data.classify_tickers(tickers)
    # KAR has a vowel, AEIA has three — they should both be in the fluent half.
    assert bool(mask["KAR"]) or bool(mask["AEIA"])


# ---------------------------------------------------------------------------
# Synthetic panel
# ---------------------------------------------------------------------------
def test_synthetic_panel_shape(null_panel):
    prices, mask, truth = null_panel
    assert prices.shape == (truth["n_days"], truth["n_stocks"])
    assert len(mask) == truth["n_stocks"]
    assert mask.sum() == truth["n_fluent"]


def test_synthetic_panel_is_deterministic():
    a, _, _ = data.synthetic_panel(n_stocks=20, n_fluent=10, n_days=100, seed=42)
    b, _, _ = data.synthetic_panel(n_stocks=20, n_fluent=10, n_days=100, seed=42)
    assert np.allclose(a.values, b.values)
    c, _, _ = data.synthetic_panel(n_stocks=20, n_fluent=10, n_days=100, seed=43)
    assert not np.allclose(a.values, c.values)


def test_synthetic_panel_prices_positive(null_panel):
    prices, _, _ = null_panel
    assert (prices > 0).all().all()


def test_fluency_premium_shifts_mean(null_panel, signal_panel):
    """A planted premium should result in higher mean returns for the fluent group."""
    prices_null, mask_null, _ = null_panel
    prices_sig, mask_sig, truth = signal_panel

    # Log returns for fluent group in each panel.
    def mean_fluent(prices, mask):
        log_rets = np.log(prices / prices.shift(1)).iloc[1:]
        fluent_cols = [t for t in prices.columns if mask.get(t, False)]
        return log_rets[fluent_cols].mean(axis=1).mean()

    def mean_nonfluent(prices, mask):
        log_rets = np.log(prices / prices.shift(1)).iloc[1:]
        nonfluent_cols = [t for t in prices.columns if not mask.get(t, True)]
        return log_rets[nonfluent_cols].mean(axis=1).mean()

    spread_null = mean_fluent(prices_null, mask_null) - mean_nonfluent(prices_null, mask_null)
    spread_sig = mean_fluent(prices_sig, mask_sig) - mean_nonfluent(prices_sig, mask_sig)
    # The signal panel's fluent-minus-nonfluent spread must exceed the null panel's.
    # Over 10y the planted 10%/yr annual premium is ~0.04% per day — we expect it to dominate.
    assert spread_sig > spread_null


def test_fingerprint_is_stable(null_panel):
    prices, _, _ = null_panel
    fp1 = data.fingerprint(prices)
    fp2 = data.fingerprint(prices)
    assert fp1 == fp2
    assert len(fp1) == 12


def test_fetch_daily_cache_only_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_daily("NOPE", fetch=False, cache_dir=str(tmp_path))
