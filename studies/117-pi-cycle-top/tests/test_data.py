"""The synthetic tape is well-formed and deterministic; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pi_cycle_top import data  # noqa: E402


def test_synthetic_shape_and_ohlc(null_tape):
    bars, truth = null_tape
    assert len(bars) == truth["n_days"]
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    # OHLC sanity: the bar's range brackets its open and close.
    assert (bars["high"] >= bars[["open", "close"]].max(axis=1) - 1e-9).all()
    assert (bars["low"] <= bars[["open", "close"]].min(axis=1) + 1e-9).all()
    assert (bars["high"] >= bars["low"]).all()
    assert (bars["close"] > 0).all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_years=5, top_sharpness=0.3, seed=7)
    b, _ = data.synthetic_daily(n_years=5, top_sharpness=0.3, seed=7)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
    c, _ = data.synthetic_daily(n_years=5, top_sharpness=0.3, seed=8)
    assert not np.allclose(a["close"].to_numpy(), c["close"].to_numpy())


def test_top_sharpness_creates_drawdown():
    """The top_sharpness knob really creates local drawdowns at the planted tops."""
    flat, _ = data.synthetic_daily(n_years=10, top_sharpness=0.0, seed=117)
    peaked, truth = data.synthetic_daily(n_years=10, top_sharpness=0.80, seed=117)
    # Around each planted signal the peaked tape should have lower returns
    for sig_date in truth["planted_signals"]:
        after_flat = flat.loc[flat.index > sig_date].iloc[:30]["close"]
        after_peak = peaked.loc[peaked.index > sig_date].iloc[:30]["close"]
        at_sig_flat = flat["close"].asof(sig_date)
        at_sig_peak = peaked["close"].asof(sig_date)
        if len(after_flat) > 5 and len(after_peak) > 5:
            ret_flat = np.log(after_flat.iloc[-1] / at_sig_flat) if at_sig_flat > 0 else 0
            ret_peak = np.log(after_peak.iloc[-1] / at_sig_peak) if at_sig_peak > 0 else 0
            assert ret_peak < ret_flat, (
                f"Expected planted top to create drawdown at {sig_date}, "
                f"got ret_peak={ret_peak:.3f} ret_flat={ret_flat:.3f}"
            )


def test_fetch_daily_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_daily(fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(null_tape):
    bars, _ = null_tape
    assert data.fingerprint(bars) == data.fingerprint(bars)
    other, _ = data.synthetic_daily(n_years=10, top_sharpness=0.0, seed=99)
    assert data.fingerprint(bars) != data.fingerprint(other)


def test_truth_records_correct_n_signals(signal_tape):
    _, truth = signal_tape
    assert truth["n_signals"] == 3
    assert len(truth["planted_signals"]) == 3
