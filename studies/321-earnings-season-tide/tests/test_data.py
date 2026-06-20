"""The synthetic tape and the earnings-window calendar mask are well-formed and
deterministic; the real loader is cache-safe (skips offline)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from earnings_season_tide import data  # noqa: E402


def test_synthetic_shape(null_tape):
    bars, truth = null_tape
    assert len(bars) == truth["n_days"]
    assert list(bars.columns) == ["close"]
    assert (bars["close"] > 0).all()
    assert bars.index.tz is None
    assert bars.index.is_monotonic_increasing


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_years=10, tide_bps=5.0, seed=7)
    b, _ = data.synthetic_daily(n_years=10, tide_bps=5.0, seed=7)
    assert np.allclose(a["close"].to_numpy(), b["close"].to_numpy())
    c, _ = data.synthetic_daily(n_years=10, tide_bps=5.0, seed=8)
    assert not np.allclose(a["close"].to_numpy(), c["close"].to_numpy())


def test_earnings_window_mask_only_in_four_months(null_tape):
    bars, _ = null_tape
    mask = data.in_earnings_window(bars.index)
    months_in = set(bars.index[mask.to_numpy()].month.unique().tolist())
    assert months_in <= {1, 4, 7, 10}
    # And every flagged day's day-of-month is inside one of the declared ranges.
    for ts in bars.index[mask.to_numpy()][:200]:
        ok = any(ts.month == mon and lo <= ts.day <= hi
                 for (mon, lo, hi) in data.PEAK_WINDOWS)
        assert ok


def test_earnings_window_mask_excludes_out_of_window_dates():
    idx = pd.DatetimeIndex(["2020-03-15", "2020-06-01", "2020-01-05", "2020-10-20"])
    mask = data.in_earnings_window(idx)
    # Mar 15 and Jun 1 are out; Jan 5 is before the Jan window; Oct 20 is in.
    assert mask.tolist() == [False, False, False, True]


def test_tide_creates_higher_in_window_mean():
    """The tide knob really lifts the in-window mean (and 0 leaves it ~flat)."""
    flat, _ = data.synthetic_daily(n_years=33, tide_bps=0.0, seed=321)
    rev, _ = data.synthetic_daily(n_years=33, tide_bps=20.0, seed=321)
    r_flat = np.log(flat["close"]).diff().dropna()
    r_rev = np.log(rev["close"]).diff().dropna()
    m_flat = data.in_earnings_window(r_flat.index).reindex(r_flat.index).to_numpy()
    m_rev = data.in_earnings_window(r_rev.index).reindex(r_rev.index).to_numpy()
    gap_flat = r_flat.to_numpy()[m_flat].mean() - r_flat.to_numpy()[~m_flat].mean()
    gap_rev = r_rev.to_numpy()[m_rev].mean() - r_rev.to_numpy()[~m_rev].mean()
    assert abs(gap_flat) < 5e-4
    assert gap_rev > 1e-3   # a clear positive in-window tide


def test_fingerprint_is_stable_and_content_sensitive(null_tape):
    bars, _ = null_tape
    assert data.fingerprint(bars) == data.fingerprint(bars)
    other, _ = data.synthetic_daily(n_years=33, tide_bps=0.0, seed=99)
    assert data.fingerprint(bars) != data.fingerprint(other)


def test_load_real_offline_raises_without_cache(tmp_path, monkeypatch):
    """With no study/shared cache and no quantlab data, the loader refuses to fetch."""
    monkeypatch.setattr(data, "STUDY_CACHE", str(tmp_path))
    monkeypatch.setattr(data, "REPO_CACHE", str(tmp_path / "nope"))
    with pytest.raises((FileNotFoundError, RuntimeError)):
        data.load_real("ZZZ_NO_SUCH", mode="total_return", fetch=False)


# --- the one cache-reading test: gated so it skips in offline CI ---
SHARED = data.shared_cache_path("SPY", "total_return")


@pytest.mark.skipif(not os.path.exists(SHARED), reason="offline CI: no shared SPY cache")
def test_load_real_shared_cache_shape():
    bars = data.load_real("SPY", "total_return", fetch=False)
    assert list(bars.columns) == ["close"]
    assert (bars["close"] > 0).all()
    assert bars.index.tz is None
