"""The synthetic tape is well-formed and deterministic; the regime table is sane; the
real loader is cache-gated (skips cleanly offline)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fed_balance_sheet import data  # noqa: E402


def test_synthetic_shape(null):
    frame, truth = null
    assert len(frame) == truth["n_days"]
    assert list(frame.columns) == ["close", "regime"]
    assert (frame["close"] > 0).all()
    assert set(frame["regime"].unique()) <= {"QE", "QT", "FLAT"}


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_days=400, qe_edge=0.0005, seed=5)
    b, _ = data.synthetic_daily(n_days=400, qe_edge=0.0005, seed=5)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
    c, _ = data.synthetic_daily(n_days=400, qe_edge=0.0005, seed=6)
    assert not np.allclose(a["close"].to_numpy(), c["close"].to_numpy())


def test_regime_table_is_contiguous_and_covers_history():
    """The hard-coded windows must abut with no gaps or overlaps."""
    w = data.regime_windows()
    for i in range(len(w) - 1):
        assert w.iloc[i]["end"] == w.iloc[i + 1]["start"]
    assert set(w["regime"].unique()) == {"QE", "QT", "FLAT"}
    # A known QE day (COVID purchases) and a known QT day (2022 runoff).
    lab = data.label_regime(pd.DatetimeIndex(["2020-06-01", "2023-01-01", "1995-01-01"]))
    assert lab.iloc[0] == "QE"
    assert lab.iloc[1] == "QT"
    assert lab.iloc[2] == "FLAT"


def test_control_plants_a_qe_qt_gap_and_null_does_not():
    """The qe_edge knob really lifts QE-day mean above QT-day mean; 0 leaves them equal."""
    flat, _ = data.synthetic_daily(n_days=6000, qe_edge=0.0, seed=317)
    ctrl, _ = data.synthetic_daily(n_days=6000, qe_edge=0.001, seed=317)

    def gap(frame):
        ret = frame["close"].pct_change()
        qe = ret[frame["regime"] == "QE"].mean()
        qt = ret[frame["regime"] == "QT"].mean()
        return qe - qt

    assert abs(gap(flat)) < 5e-4         # null: QE and QT drift the same
    assert gap(ctrl) > 1e-3              # control: QE clearly above QT


def test_fingerprint_is_stable_and_content_sensitive(null):
    frame, _ = null
    assert data.fingerprint(frame) == data.fingerprint(frame)
    other, _ = data.synthetic_daily(n_days=6000, qe_edge=0.0, seed=99)
    assert data.fingerprint(frame) != data.fingerprint(other)


# ---- real loader: cache-gated, never required offline ----------------------
def _shared_cache_path():
    """First existing shared-cache candidate for SPY, or None (drives the skipif)."""
    return next((p for p in data._shared_cache_candidates("SPY") if os.path.exists(p)), None)


SHARED = _shared_cache_path()


def test_load_real_raises_without_any_cache(tmp_path, monkeypatch):
    """With no shared cache and an empty study cache, load_real raises (callers fall back)."""
    # Force every shared-cache candidate to a path that does not exist.
    monkeypatch.setattr(data, "_shared_cache_candidates",
                        lambda ticker, mode="split_only": [str(tmp_path / "nope.parquet")])
    with pytest.raises(FileNotFoundError):
        data.load_real("SPY", fetch=False, cache_dir=str(tmp_path))


@pytest.mark.skipif(SHARED is None or not os.path.exists(SHARED),
                    reason="offline CI: shared SPY cache absent")
def test_load_real_from_shared_cache_is_labelled():
    bars = data.load_real("SPY", fetch=False)
    assert list(bars.columns) == ["close", "regime"]
    assert (bars["close"] > 0).all()
    assert set(bars["regime"].unique()) <= {"QE", "QT", "FLAT"}
    assert bars.index.is_monotonic_increasing
