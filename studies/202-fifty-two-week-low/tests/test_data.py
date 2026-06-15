"""The synthetic panel is well-formed and deterministic; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fifty_two_week_low import data  # noqa: E402

# ---------------------------------------------------------------------------
# Cache guard for real-data tests
# ---------------------------------------------------------------------------
_CACHE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "_cache")
)
_SAMPLE_CACHE = data._cache_path("AAPL", _CACHE_DIR)

requires_cache = pytest.mark.skipif(
    not os.path.exists(_SAMPLE_CACHE),
    reason="cache absent (offline CI); covered by synthetic tests",
)


# ---------------------------------------------------------------------------
# Synthetic panel tests
# ---------------------------------------------------------------------------
def test_synthetic_shape(coin_panel):
    panel, truth = coin_panel
    assert panel.shape == (truth["n_days"], truth["n_stocks"])
    assert list(panel.columns) == [f"S{i:02d}" for i in range(truth["n_stocks"])]
    assert panel.index.name == "date"


def test_synthetic_positive_prices(coin_panel):
    panel, _ = coin_panel
    assert (panel > 0).all().all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_panel(n_stocks=5, n_days=300, reversion=0.1, seed=42)
    b, _ = data.synthetic_panel(n_stocks=5, n_days=300, reversion=0.1, seed=42)
    assert np.allclose(a.to_numpy(), b.to_numpy())
    c, _ = data.synthetic_panel(n_stocks=5, n_days=300, reversion=0.1, seed=99)
    assert not np.allclose(a.to_numpy(), c.to_numpy())


def test_synthetic_business_day_index(coin_panel):
    panel, _ = coin_panel
    # All dates must be weekdays (Mon=0 ... Fri=4, no Sat=5/Sun=6)
    assert (panel.index.dayofweek < 5).all()


def test_reversion_induces_negative_autocorrelation():
    """The reversion knob plants negative autocorrelation in returns (as intended)."""
    flat, _ = data.synthetic_panel(n_stocks=5, n_days=800, reversion=0.0, seed=202)
    rev, _ = data.synthetic_panel(n_stocks=5, n_days=800, reversion=0.30, seed=202)

    def mean_lag1_ac(panel):
        log_ret = np.log(panel).diff().dropna()
        acs = []
        for col in log_ret.columns:
            r = log_ret[col].to_numpy()
            acs.append(np.corrcoef(r[:-1], r[1:])[0, 1])
        return np.mean(acs)

    ac_flat = mean_lag1_ac(flat)
    ac_rev = mean_lag1_ac(rev)
    # Flat should have near-zero AC; reverting should have negative AC
    assert abs(ac_flat) < 0.08
    assert ac_rev < -0.10


def test_fetch_daily_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_daily("NOPE", fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(coin_panel):
    panel, _ = coin_panel
    assert data.fingerprint(panel) == data.fingerprint(panel)
    other, _ = data.synthetic_panel(n_stocks=10, n_days=600, reversion=0.0, seed=99)
    assert data.fingerprint(panel) != data.fingerprint(other)


# ---------------------------------------------------------------------------
# Real-data test (cache-gated)
# ---------------------------------------------------------------------------
@requires_cache
def test_real_fetch_aapl_shape():
    """Real AAPL daily bar loads from cache and has expected columns."""
    bars = data.fetch_daily("AAPL", fetch=False, cache_dir=_CACHE_DIR)
    assert set(["open", "high", "low", "close", "volume"]).issubset(bars.columns)
    assert len(bars) > 100
    assert (bars["close"] > 0).all()


@requires_cache
def test_real_panel_aligned():
    """Tickers available in cache align to a common date grid."""
    avail = [
        t for t in data.SP500_BASKET
        if os.path.exists(data._cache_path(t, _CACHE_DIR))
    ]
    if len(avail) < 2:
        pytest.skip("need at least 2 cached tickers")
    panel = data.load_panel(avail, cache_dir=_CACHE_DIR)
    assert panel.shape[1] == len(avail)
    assert not panel.isna().any().any()
