"""The synthetic panel is well-formed and deterministic; the real fetch is cache-safe."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from home_bias import data  # noqa: E402


def test_synthetic_shape(equal_mu_low_corr):
    ret, truth = equal_mu_low_corr
    assert len(ret) == truth["n_days"]
    assert list(ret.columns) == ["SPY", "EFA", "EEM"]
    assert ret.notna().all().all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_panel(n_days=500, seed=7)
    b, _ = data.synthetic_panel(n_days=500, seed=7)
    assert np.allclose(a["SPY"].to_numpy(), b["SPY"].to_numpy())
    c, _ = data.synthetic_panel(n_days=500, seed=8)
    assert not np.allclose(a["SPY"].to_numpy(), c["SPY"].to_numpy())


def test_corr_knob_affects_realized_corr():
    """The corr_intl parameter actually shifts the realized SPY-EFA correlation."""
    lo, _ = data.synthetic_panel(n_days=5000, corr_intl=0.20, seed=145)
    hi, _ = data.synthetic_panel(n_days=5000, corr_intl=0.90, seed=145)
    assert lo["SPY"].corr(lo["EFA"]) < hi["SPY"].corr(hi["EFA"])
    assert hi["SPY"].corr(hi["EFA"]) > 0.7


def test_mu_spread_plants_us_outperformance():
    """When mu_us > mu_intl, SPY mean daily return > EFA mean daily return."""
    ret, truth = data.synthetic_panel(n_days=10000, mu_us=0.12 / 252, mu_intl=0.06 / 252,
                                       corr_intl=0.5, seed=145)
    assert ret["SPY"].mean() > ret["EFA"].mean()
    assert truth["mu_spread_ann"] > 0.05


def test_fetch_panel_cache_missing_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_panel(cache_dir=str(tmp_path), fetch=False)


def test_fingerprint_stable_and_content_sensitive(equal_mu_low_corr):
    ret, _ = equal_mu_low_corr
    assert data.fingerprint(ret) == data.fingerprint(ret)
    other, _ = data.synthetic_panel(n_days=3000, seed=99)
    assert data.fingerprint(ret) != data.fingerprint(other)


def test_log_returns_finite(equal_mu_low_corr):
    ret, _ = equal_mu_low_corr
    assert np.isfinite(ret.to_numpy()).all()
