"""The synthetic tape is well-formed and deterministic; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cold_open import data  # noqa: E402


def test_synthetic_shape_and_columns(null_tape):
    df, truth = null_tape
    assert len(df) == truth["n_years"]
    assert list(df.columns) == ["jan_ret", "fbd_ret"]
    assert df.index.name == "year"


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_annual(n_years=40, signal_strength=0.1, seed=7)
    b, _ = data.synthetic_annual(n_years=40, signal_strength=0.1, seed=7)
    assert np.allclose(a["jan_ret"].to_numpy(), b["jan_ret"].to_numpy())
    c, _ = data.synthetic_annual(n_years=40, signal_strength=0.1, seed=8)
    assert not np.allclose(a["jan_ret"].to_numpy(), c["jan_ret"].to_numpy())


def test_synthetic_returns_finite(null_tape):
    df, _ = null_tape
    assert df["jan_ret"].isna().sum() == 0
    assert df["fbd_ret"].isna().sum() == 0
    assert np.isfinite(df["jan_ret"].to_numpy()).all()
    assert np.isfinite(df["fbd_ret"].to_numpy()).all()


def test_synthetic_year_index(null_tape):
    df, truth = null_tape
    assert df.index[0] == truth["start_year"]
    assert df.index[-1] == truth["start_year"] + truth["n_years"] - 1


def test_signal_creates_correlation():
    """The signal knob really induces Jan-sign vs Feb-Dec correlation (and 0 induces ~none)."""
    null, _ = data.synthetic_annual(n_years=500, signal_strength=0.0, seed=80)
    strong, _ = data.synthetic_annual(n_years=500, signal_strength=0.60, seed=80)
    import pandas as pd
    corr_null = np.corrcoef(
        np.sign(null["jan_ret"].to_numpy()), null["fbd_ret"].to_numpy()
    )[0, 1]
    corr_strong = np.corrcoef(
        np.sign(strong["jan_ret"].to_numpy()), strong["fbd_ret"].to_numpy()
    )[0, 1]
    assert abs(corr_null) < 0.10
    assert corr_strong > 0.20


def test_fetch_gspc_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_gspc(fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_stable_and_sensitive(null_tape):
    df, _ = null_tape
    assert data.fingerprint(df) == data.fingerprint(df)
    other, _ = data.synthetic_annual(n_years=80, signal_strength=0.0, seed=99)
    assert data.fingerprint(df) != data.fingerprint(other)
