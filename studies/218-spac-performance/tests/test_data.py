"""Synthetic tape is well-formed and deterministic; real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spac_performance import data  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY_CACHE = os.path.abspath(os.path.join(HERE, "..", "_cache"))
SPAK_CACHE = os.path.join(STUDY_CACHE, "spak_daily.parquet")
SPY_CACHE = os.path.join(STUDY_CACHE, "spy_daily.parquet")


def test_synthetic_shape_and_columns(null_tape):
    prices, truth = null_tape
    assert len(prices) == truth["n_days"]
    assert list(prices.columns) == ["spac", "spy"]
    assert (prices > 0).all().all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_days=300, alpha_ann=0.0, seed=5)
    b, _ = data.synthetic_daily(n_days=300, alpha_ann=0.0, seed=5)
    assert np.allclose(a["spy"].to_numpy(), b["spy"].to_numpy())
    c, _ = data.synthetic_daily(n_days=300, alpha_ann=0.0, seed=6)
    assert not np.allclose(a["spy"].to_numpy(), c["spy"].to_numpy())


def test_synthetic_alpha_knob_shifts_return():
    """A higher planted alpha must produce a higher fund CAGR (market held fixed, seed fixed)."""
    p_low, _ = data.synthetic_daily(n_days=500, alpha_ann=-0.30, seed=7)
    p_high, _ = data.synthetic_daily(n_days=500, alpha_ann=0.05, seed=7)
    cagr_low = float((p_low["spac"].iloc[-1] / p_low["spac"].iloc[0]) ** (252 / 500) - 1)
    cagr_high = float((p_high["spac"].iloc[-1] / p_high["spac"].iloc[0]) ** (252 / 500) - 1)
    assert cagr_high > cagr_low


def test_synthetic_index_is_business_days(null_tape):
    prices, _ = null_tape
    diffs = np.diff(prices.index.values).astype("timedelta64[D]").astype(int)
    assert diffs.min() >= 1
    assert diffs.max() <= 3


def test_fetch_daily_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_daily("NOPE", fetch=False, cache_dir=str(tmp_path))


def test_load_spak_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_spak_vs_spy(fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(null_tape):
    prices, _ = null_tape
    assert data.fingerprint(prices["spy"]) == data.fingerprint(prices["spy"])
    other, _ = data.synthetic_daily(n_days=600, alpha_ann=0.0, seed=99)
    assert data.fingerprint(prices["spy"]) != data.fingerprint(other["spy"])


@pytest.mark.skipif(not os.path.exists(SPAK_CACHE), reason="real-tape cache absent offline/CI")
def test_load_spak_vs_spy_shape():
    """Real tape: SPAK and SPY are aligned and have the expected columns."""
    prices = data.load_spak_vs_spy(fetch=False, cache_dir=STUDY_CACHE)
    assert "SPAK" in prices.columns
    assert "SPY" in prices.columns
    assert len(prices) > 400  # roughly 484 days
    assert (prices > 0).all().all()


@pytest.mark.skipif(not os.path.exists(SPY_CACHE), reason="real-tape cache absent offline/CI")
def test_load_basket_vs_spy_shape():
    """Real tape: basket and SPY are aligned and rebased to 100."""
    prices = data.load_basket_vs_spy(fetch=False, cache_dir=STUDY_CACHE)
    assert "BASKET" in prices.columns
    assert "SPY" in prices.columns
    assert len(prices) > 1000
    assert (prices > 0).all().all()
