"""The synthetic panel is well-formed and deterministic; real-fetch path is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from asset_turnover import data  # noqa: E402


def test_synthetic_panel_shape(null_panel):
    at_sig, fwd, truth = null_panel
    assert at_sig.shape == (truth["n_years"], truth["n_firms"])
    assert fwd.shape == (truth["n_years"], truth["n_firms"])
    assert at_sig.index.name == "year"
    assert fwd.index.name == "year"


def test_synthetic_panel_at_values_positive(null_panel):
    """Asset turnover ratios are generated as exp(normal), hence always strictly positive."""
    at_sig, _, _ = null_panel
    assert (at_sig > 0).all().all()


def test_synthetic_panel_is_deterministic():
    a_sig, a_fwd, _ = data.synthetic_panel(n_firms=80, n_years=10, premium=0.03, seed=7)
    b_sig, b_fwd, _ = data.synthetic_panel(n_firms=80, n_years=10, premium=0.03, seed=7)
    assert np.allclose(a_sig.to_numpy(), b_sig.to_numpy())
    assert np.allclose(a_fwd.to_numpy(), b_fwd.to_numpy())

    # Different seed → different values
    c_sig, _, _ = data.synthetic_panel(n_firms=80, n_years=10, premium=0.03, seed=8)
    assert not np.allclose(a_sig.to_numpy(), c_sig.to_numpy())


def test_synthetic_panel_premium_knob():
    """Premium=0 → cross-sectional correlation between AT rank and fwd return ≈ 0;
    premium=0.06 → positive cross-sectional correlation."""
    import pandas as pd

    null_sig, null_fwd, _ = data.synthetic_panel(n_firms=300, n_years=20, premium=0.0, seed=42)
    real_sig, real_fwd, _ = data.synthetic_panel(n_firms=300, n_years=20, premium=0.06, seed=42)

    def mean_cs_corr(sig, fwd):
        corrs = []
        for y in sig.index:
            s = sig.loc[y].dropna()
            f = fwd.loc[y].dropna()
            s, f = s.align(f, join="inner")
            if len(s) > 10:
                corrs.append(float(pd.Series(s.values).corr(pd.Series(f.values))))
        return float(np.mean(corrs))

    corr_null = mean_cs_corr(null_sig, null_fwd)
    corr_real = mean_cs_corr(real_sig, real_fwd)
    assert abs(corr_null) < 0.15      # null: near-zero cross-sectional correlation
    assert corr_real > corr_null      # real: AT rank correlates with returns when premium > 0


def test_fetch_panel_offline_returns_empty(tmp_path):
    """When the EDGAR caches are absent, fetch_panel returns empty DataFrames gracefully."""
    at_sig, roa_sig, fwd = data.fetch_panel(cache_dir=str(tmp_path))
    assert at_sig.empty
    assert roa_sig.empty
    assert fwd.empty


def test_fingerprint_stable_and_sensitive(null_panel):
    at_sig, _, _ = null_panel
    assert data.fingerprint(at_sig) == data.fingerprint(at_sig)
    other_sig, _, _ = data.synthetic_panel(n_firms=150, n_years=14, premium=0.0, seed=99)
    assert data.fingerprint(at_sig) != data.fingerprint(other_sig)
