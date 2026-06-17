"""Tests for the data layer — synthetic generator and real-data loader (cache-gated)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cannabis_stocks import data  # noqa: E402

# ---------------------------------------------------------------------------
# Cache-gate: any test that reads real Yahoo data must be marked with this.
# ---------------------------------------------------------------------------
_CACHE_DIR = data.DEFAULT_CACHE
_SPY_CACHE = data._cache_path("SPY", _CACHE_DIR)
_MSOS_CACHE = data._cache_path("MSOS", _CACHE_DIR)

requires_cache = pytest.mark.skipif(
    not (os.path.exists(_SPY_CACHE) and os.path.exists(_MSOS_CACHE)),
    reason="real-data cache absent (offline CI); covered by synthetic tests",
)


# ---------------------------------------------------------------------------
# Synthetic generator tests — always run, no network needed
# ---------------------------------------------------------------------------
def test_synthetic_shape_and_columns():
    prices, truth = data.synthetic_daily(n_years=5, seed=212)
    assert list(prices.columns) == ["spy", "cannabis"]
    assert len(prices) == 5 * data.TRADING_DAYS_PER_YEAR
    assert prices.index.name == "date"


def test_synthetic_is_deterministic():
    p1, t1 = data.synthetic_daily(n_years=5, seed=42)
    p2, t2 = data.synthetic_daily(n_years=5, seed=42)
    assert (p1["spy"] == p2["spy"]).all()
    assert (p1["cannabis"] == p2["cannabis"]).all()
    assert t1 == t2


def test_synthetic_prices_positive():
    prices, _ = data.synthetic_daily(n_years=10, seed=212)
    assert (prices > 0).all().all()


def test_synthetic_negative_alpha_erodes_value():
    """Strongly negative alpha should cause cannabis to underperform SPY over 10 years."""
    prices, _ = data.synthetic_daily(n_years=10, beta=1.0, alpha_ann=-0.30,
                                      idio_vol_ann=0.10, seed=212)
    total_spy = prices["spy"].iloc[-1] / prices["spy"].iloc[0]
    total_can = prices["cannabis"].iloc[-1] / prices["cannabis"].iloc[0]
    assert total_can < total_spy, (
        f"Expected cannabis total return < SPY with -30%/yr alpha; "
        f"got spy={total_spy:.2f}x, cannabis={total_can:.2f}x"
    )


def test_synthetic_null_cannabis_uncorrelated():
    """At beta=0, cannabis should have near-zero correlation with SPY."""
    prices, _ = data.synthetic_daily(n_years=20, beta=0.0, alpha_ann=0.0, seed=0)
    r_spy = np.log(prices["spy"]).diff().dropna()
    r_can = np.log(prices["cannabis"]).diff().dropna()
    corr = float(np.corrcoef(r_spy, r_can)[0, 1])
    assert abs(corr) < 0.15, f"Expected near-zero corr at beta=0, got {corr:.3f}"


def test_fingerprint_changes_with_prices():
    p1, _ = data.synthetic_daily(n_years=5, seed=1)
    p2, _ = data.synthetic_daily(n_years=5, seed=2)
    assert data.fingerprint(p1) != data.fingerprint(p2)


def test_truth_dict_keys():
    _, truth = data.synthetic_daily(n_years=5)
    for key in ("beta", "alpha_ann", "n_years", "n_days", "seed",
                "planted_beta", "planted_alpha_ann"):
        assert key in truth, f"missing key: {key}"


# ---------------------------------------------------------------------------
# Real-data tests — only run when the cache is present
# ---------------------------------------------------------------------------
@requires_cache
def test_real_load_pair_returns_aligned_frame():
    pair = data.load_pair("MSOS", fetch=False, cache_dir=_CACHE_DIR)
    assert list(pair.columns) == ["spy", "cannabis"]
    assert len(pair) > 500, "Expected >500 common trading days"
    assert pair.notna().all().all()


@requires_cache
def test_real_prices_start_after_msos_ipo():
    pair = data.load_pair("MSOS", fetch=False, cache_dir=_CACHE_DIR)
    assert str(pair.index[0].date()) >= data.MSOS_START


@requires_cache
def test_real_fingerprints_stable():
    """Fingerprints should be stable for the same cached parquet."""
    pair = data.load_pair("MSOS", fetch=False, cache_dir=_CACHE_DIR)
    fp1 = data.fingerprint(pair, "spy")
    fp2 = data.fingerprint(pair, "spy")
    assert fp1 == fp2
    assert len(fp1) == 12
