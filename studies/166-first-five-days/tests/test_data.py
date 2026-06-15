"""The synthetic tape is well-formed and deterministic; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from first_five_days import data  # noqa: E402


def test_synthetic_shape(null_tape):
    df, truth = null_tape
    assert len(df) == truth["n_years"]
    assert list(df.columns) == ["ret_5d", "ret_year"]


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_annual(n_years=50, signal_strength=0.2, seed=7)
    b, _ = data.synthetic_annual(n_years=50, signal_strength=0.2, seed=7)
    assert np.allclose(a["ret_5d"].to_numpy(), b["ret_5d"].to_numpy())
    c, _ = data.synthetic_annual(n_years=50, signal_strength=0.2, seed=8)
    assert not np.allclose(a["ret_5d"].to_numpy(), c["ret_5d"].to_numpy())


def test_synthetic_null_has_equity_drift(null_tape):
    """Even the null tape should have a positive average return (planted drift)."""
    df, _ = null_tape
    assert df["ret_year"].mean() > 0


def test_signal_tape_has_higher_contrast_than_null(null_tape, signal_tape):
    """Planting a signal_strength > 0 should raise the mean-contrast."""
    from first_five_days import strategy as st

    null_c = st.mean_contrast(null_tape[0])["contrast_pct"]
    sig_c = st.mean_contrast(signal_tape[0])["contrast_pct"]
    assert sig_c > null_c


def test_fetch_gspc_cache_only_raises_without_cache(tmp_path):
    with pytest.raises((FileNotFoundError, Exception)):
        data.fetch_gspc(fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(null_tape):
    df, _ = null_tape
    assert data.fingerprint(df) == data.fingerprint(df)
    other, _ = data.synthetic_annual(n_years=100, signal_strength=0.0, seed=99)
    assert data.fingerprint(df) != data.fingerprint(other)
