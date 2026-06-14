"""The synthetic panel is well-formed and deterministic; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from leverage_anomaly import data  # noqa: E402


def test_synthetic_shape(null_panel):
    lev, fwd, truth = null_panel
    assert lev.shape == fwd.shape
    assert lev.shape[0] == 12    # n_years
    assert lev.shape[1] == 150   # n_firms
    assert lev.index.name == "year"
    assert fwd.index.name == "year"


def test_synthetic_is_deterministic():
    a, b, _ = data.synthetic_panel(n_firms=100, n_years=8, premium=0.05, seed=10)
    c, d, _ = data.synthetic_panel(n_firms=100, n_years=8, premium=0.05, seed=10)
    assert np.allclose(a.to_numpy(), c.to_numpy())
    assert np.allclose(b.to_numpy(), d.to_numpy())
    e, _, _ = data.synthetic_panel(n_firms=100, n_years=8, premium=0.05, seed=11)
    assert not np.allclose(a.to_numpy(), e.to_numpy())


def test_truth_dataclass(null_panel, anomaly_panel):
    _, _, null_truth = null_panel
    _, _, anom_truth = anomaly_panel
    assert null_truth.premium == 0.0
    assert not null_truth.has_premium
    assert anom_truth.premium == 0.08
    assert anom_truth.has_premium


def test_premium_affects_spread():
    """A planted premium lifts the low-minus-high spread in expectation."""
    lev0, fwd0, _ = data.synthetic_panel(n_firms=300, n_years=20, premium=0.0, seed=99)
    lev1, fwd1, _ = data.synthetic_panel(n_firms=300, n_years=20, premium=0.10, seed=99)

    def avg_spread(lev, fwd):
        spreads = []
        for y in lev.index:
            s = lev.loc[y]
            if y not in fwd.index:
                continue
            r = fwd.loc[y]
            lo = r[s <= s.quantile(0.20)].mean()
            hi = r[s >= s.quantile(0.80)].mean()
            spreads.append(lo - hi)
        return float(np.nanmean(spreads))

    s0 = avg_spread(lev0, fwd0)
    s1 = avg_spread(lev1, fwd1)
    # With premium=0.10, low-minus-high should be meaningfully positive
    assert s1 > s0
    assert s1 > 0.03   # at least 3 pct/yr planted premium is recovered on average


def test_fingerprint_stable_and_content_sensitive(null_panel, anomaly_panel):
    lev0, fwd0, _ = null_panel
    lev1, fwd1, _ = anomaly_panel
    # Leverage signals are identical (same seed, only premium differs in fwd_ret)
    assert data.fingerprint(lev0) == data.fingerprint(lev0)
    # Forward returns differ between null and anomaly panels
    assert data.fingerprint(fwd0) != data.fingerprint(fwd1)
    # Different seeds produce different leverage panels
    lev_other, _, _ = data.synthetic_panel(n_firms=150, n_years=12, premium=0.0, seed=99)
    assert data.fingerprint(lev0) != data.fingerprint(lev_other)


def test_fetch_panel_returns_empty_when_cache_absent(tmp_path):
    """Absent cache → three empty DataFrames (not an exception)."""
    l1, l2, fr = data.fetch_panel(cache_dir=str(tmp_path))
    assert l1.empty
    assert l2.empty
    assert fr.empty
