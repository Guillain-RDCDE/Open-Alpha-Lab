"""Data-layer invariants for Study 251 (Crypto-Reversal)."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from crypto_reversal import data  # noqa: E402

_CACHE_PATH = data._cache_path()
requires_cache = pytest.mark.skipif(
    not os.path.exists(_CACHE_PATH),
    reason="crypto panel cache absent (offline CI); covered by synthetic tests",
)


def test_synthetic_panel_shape_and_index(planted):
    prices, truth = planted
    assert prices.shape[1] == truth["n_tokens"] == 20
    assert prices.shape[0] == truth["n_days"]
    assert prices.index.is_monotonic_increasing
    assert not prices.isna().any().any()
    # 365-day calendar: consecutive daily index, no weekday gaps.
    deltas = prices.index.to_series().diff().dropna().dt.days.unique()
    assert set(deltas.tolist()) == {1}


def test_synthetic_panel_deterministic():
    a, _ = data.synthetic_panel(seed=213)
    b, _ = data.synthetic_panel(seed=213)
    assert np.allclose(a.to_numpy(), b.to_numpy())


def test_synthetic_seed_changes_panel():
    a, _ = data.synthetic_panel(seed=213)
    b, _ = data.synthetic_panel(seed=999)
    assert not np.allclose(a.to_numpy(), b.to_numpy())


def test_fingerprint_stable_and_sensitive(planted):
    prices, _ = planted
    fp1 = data.fingerprint(prices)
    assert len(fp1) == 12
    bumped = prices.copy()
    bumped.iloc[0, 0] *= 1.01
    assert data.fingerprint(bumped) != fp1


def test_null_has_lower_revert_kappa(null_panel, planted):
    assert null_panel[1]["kappa"] == 0.0
    assert planted[1]["kappa"] > 0.0


@requires_cache
def test_real_panel_loads_and_is_wide():
    panel = data.fetch_panel(fetch=False)
    assert panel.shape[1] >= 10          # at least 10 surviving liquid tokens
    assert panel.shape[0] > 1500         # several years of daily data
    assert "BTC-USD" in panel.columns
