"""The synthetic tape is well-formed and deterministic; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from move_index import data  # noqa: E402


def test_synthetic_shape_and_columns(null_tape):
    df, truth = null_tape
    assert len(df) == truth["n_days"]
    assert list(df.columns) == ["MOVE", "VIX", "SPY_close", "SPY_ret"]


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_days=200, move_signal=0.0, seed=5)
    b, _ = data.synthetic_daily(n_days=200, move_signal=0.0, seed=5)
    assert np.allclose(a["SPY_close"].to_numpy(), b["SPY_close"].to_numpy())
    c, _ = data.synthetic_daily(n_days=200, move_signal=0.0, seed=6)
    assert not np.allclose(a["SPY_close"].to_numpy(), c["SPY_close"].to_numpy())


def test_move_in_plausible_range(null_tape):
    df, _ = null_tape
    assert df["MOVE"].min() > 30
    assert df["MOVE"].max() < 260
    assert df["VIX"].min() > 5
    assert df["VIX"].max() < 100


def test_spy_close_positive(null_tape):
    df, _ = null_tape
    assert (df["SPY_close"] > 0).all()


def test_spy_ret_aligned(null_tape):
    """SPY_ret[i] should match log(SPY_close[i]/SPY_close[i-1]) for i >= 1."""
    df, _ = null_tape
    log_ret = np.log(df["SPY_close"]).diff().dropna().to_numpy()
    spy_ret_tail = df["SPY_ret"].iloc[1:].to_numpy()
    assert np.allclose(log_ret, spy_ret_tail, atol=1e-10)


def test_signal_knob_creates_predictability():
    """The move_signal knob must induce a real negative correlation between MOVE rank
    and next-day SPY returns (and zero correlation when signal=0)."""
    null, _ = data.synthetic_daily(n_days=2000, move_signal=0.0, seed=112)
    sig, _ = data.synthetic_daily(n_days=2000, move_signal=0.05, seed=112)

    # Compute rolling MOVE percentile and lag it by 1
    for df, label, expected_neg in [(null, "null", False), (sig, "signal", True)]:
        rank = df["MOVE"].rolling(252, min_periods=63).rank(pct=True).shift(1)
        fwd = df["SPY_ret"]
        combined = rank.to_frame("rank").join(fwd.rename("fwd")).dropna()
        corr = combined["rank"].corr(combined["fwd"])
        if expected_neg:
            assert corr < -0.02, f"{label}: expected neg corr, got {corr:.4f}"
        else:
            assert abs(corr) < 0.10, f"{label}: expected near-zero corr, got {corr:.4f}"


def test_fetch_daily_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_daily(fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(null_tape):
    df, _ = null_tape
    assert data.fingerprint(df) == data.fingerprint(df)
    other, _ = data.synthetic_daily(n_days=2000, move_signal=0.0, seed=99)
    assert data.fingerprint(df) != data.fingerprint(other)
