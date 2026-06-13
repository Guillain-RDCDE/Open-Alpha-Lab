"""The synthetic tape is well-formed and deterministic; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vix_term_structure import data  # noqa: E402


def test_synthetic_shape_and_columns(null_tape):
    df, truth = null_tape
    assert len(df) == truth["n_days"]
    assert set(["VIX", "VIX3M", "slope", "SPY_close", "SPY_ret"]).issubset(df.columns)
    assert (df["VIX"] > 0).all()
    assert (df["VIX3M"] > 0).all()
    assert (df["SPY_close"] > 0).all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_days=500, contango_signal=0.0, seed=111)
    b, _ = data.synthetic_daily(n_days=500, contango_signal=0.0, seed=111)
    assert np.allclose(a["SPY_close"].to_numpy(), b["SPY_close"].to_numpy())
    c, _ = data.synthetic_daily(n_days=500, contango_signal=0.0, seed=999)
    assert not np.allclose(a["SPY_close"].to_numpy(), c["SPY_close"].to_numpy())


def test_slope_definition(null_tape):
    """slope = log(VIX3M / VIX) should match the computed column."""
    df, _ = null_tape
    expected = np.log(df["VIX3M"] / df["VIX"])
    assert np.allclose(df["slope"].to_numpy(), expected.to_numpy(), atol=1e-9)


def test_contango_is_majority_state(null_tape):
    """Natural VIX dynamics produce more contango than backwardation."""
    df, _ = null_tape
    frac_contango = (df["slope"] > 0).mean()
    assert frac_contango > 0.55, f"Expected >55% contango, got {frac_contango:.2%}"


def test_spy_ret_is_log_diff(null_tape):
    """SPY_ret should equal the log-difference of SPY_close."""
    df, _ = null_tape
    computed = np.log(df["SPY_close"]).diff()
    assert np.allclose(df["SPY_ret"].iloc[1:].to_numpy(), computed.iloc[1:].to_numpy(), atol=1e-9)


def test_signal_knob_has_effect():
    """A planted signal should produce positive slope-rank / return correlation."""
    null, _ = data.synthetic_daily(n_days=2000, contango_signal=0.0, seed=111)
    sig, _ = data.synthetic_daily(n_days=2000, contango_signal=0.05, seed=111)
    # Correlation between lagged slope rank and forward return should be higher with signal.
    slope_rank_null = null["slope"].rolling(252, min_periods=63).rank(pct=True).shift(1)
    slope_rank_sig = sig["slope"].rolling(252, min_periods=63).rank(pct=True).shift(1)
    fwd_null = np.log(null["SPY_close"]).diff().shift(-1)
    fwd_sig = np.log(sig["SPY_close"]).diff().shift(-1)
    valid_null = slope_rank_null.notna() & fwd_null.notna()
    valid_sig = slope_rank_sig.notna() & fwd_sig.notna()
    corr_null = np.corrcoef(slope_rank_null[valid_null], fwd_null[valid_null])[0, 1]
    corr_sig = np.corrcoef(slope_rank_sig[valid_sig], fwd_sig[valid_sig])[0, 1]
    assert corr_sig > corr_null + 0.02, (
        f"Signal tape correlation ({corr_sig:.4f}) should clearly exceed null ({corr_null:.4f})"
    )


def test_fetch_daily_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_daily(fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(null_tape):
    df, _ = null_tape
    assert data.fingerprint(df) == data.fingerprint(df)
    other, _ = data.synthetic_daily(n_days=2000, contango_signal=0.0, seed=42)
    assert data.fingerprint(df) != data.fingerprint(other)
