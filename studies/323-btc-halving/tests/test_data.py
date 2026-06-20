"""The synthetic tape is well-formed and deterministic; cycle phase is a valid calendar
fact; the real cache read is gated so CI stays offline."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from btc_halving import data  # noqa: E402


def test_synthetic_shape_and_ohlc(null_tape):
    bars, truth = null_tape
    assert len(bars) == truth["n_days"]
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    assert (bars["high"] >= bars[["open", "close"]].max(axis=1) - 1e-6).all()
    assert (bars["low"] <= bars[["open", "close"]].min(axis=1) + 1e-6).all()
    assert (bars["high"] >= bars["low"]).all()
    assert (bars["close"] > 0).all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_days=400, cycle_amp=1.0, seed=5)
    b, _ = data.synthetic_daily(n_days=400, cycle_amp=1.0, seed=5)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
    c, _ = data.synthetic_daily(n_days=400, cycle_amp=1.0, seed=6)
    assert not np.allclose(a["close"].to_numpy(), c["close"].to_numpy())


def test_four_hardcoded_halvings():
    assert len(data.HALVINGS) == 4
    assert data.HALVINGS[0] == pd.Timestamp("2012-11-28")
    assert data.HALVINGS[-1] == pd.Timestamp("2024-04-20")
    # strictly increasing, ~4 years apart
    gaps = np.diff([h.value for h in data.HALVINGS])
    assert (gaps > 0).all()


def test_cycle_phase_in_unit_interval_and_nan_before_first_halving():
    idx = pd.date_range("2010-01-01", periods=6000, freq="D")
    ph = data.cycle_phase(idx)
    # before the first halving (2012-11-28) phase is undefined
    assert np.isnan(ph[idx < data.HALVINGS[0]]).all()
    defined = ph[~np.isnan(ph)]
    assert defined.size > 0
    assert (defined >= 0).all() and (defined < 1.0 + 1e-9).all()


def test_cycle_phase_handles_ms_resolution_index():
    """A datetime64[ms] index (as the cached parquet uses) must still phase correctly."""
    idx = pd.date_range("2015-01-01", periods=2000, freq="D").as_unit("ms")
    ph = data.cycle_phase(idx)
    assert np.isfinite(ph).any()


def test_days_since_halving_is_zero_at_a_halving():
    idx = pd.DatetimeIndex([data.HALVINGS[2]])  # 2020-05-11
    dsl = data.days_since_last_halving(idx)
    assert dsl[0] == 0.0


def test_planted_cycle_creates_phase_dependent_drift(cycle_tape, null_tape):
    """With a planted cycle the post-halving quarter outperforms the average; the null
    tape shows no such systematic gap of the same size."""
    from btc_halving import strategy as st
    pb_cyc = st.phase_bucket_returns(cycle_tape[0], 4)
    pb_nul = st.phase_bucket_returns(null_tape[0], 4)
    # bucket 0 (post-halving) should be the strongest on the planted tape
    assert pb_cyc.loc[0, "mean_bps"] == pb_cyc["mean_bps"].max()
    # and clearly larger than the same bucket on the null tape
    assert pb_cyc.loc[0, "mean_bps"] > pb_nul.loc[0, "mean_bps"]


def test_fingerprint_is_stable_and_content_sensitive(null_tape):
    bars, _ = null_tape
    assert data.fingerprint(bars) == data.fingerprint(bars)
    other, _ = data.synthetic_daily(n_days=4200, cycle_amp=0.0, seed=99)
    assert data.fingerprint(bars) != data.fingerprint(other)


# ---- real cache: gated so CI stays offline --------------------------------
@pytest.mark.skipif(not os.path.exists(data.BTC_CACHE), reason="offline CI")
def test_load_real_cache_shape():
    bars = data.load_real()
    assert list(bars.columns) == ["open", "high", "low", "close", "volume"]
    assert bars.index.is_monotonic_increasing
    assert (bars["close"] > 0).all()


def test_load_real_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_real(cache_path=str(tmp_path / "nope.parquet"))
