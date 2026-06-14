"""The synthetic tape is well-formed, deterministic, and the lunar engine is correct."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lunar_effect import data  # noqa: E402


# ---------------------------------------------------------------------------
# Synthetic tape basics
# ---------------------------------------------------------------------------
def test_synthetic_shape_and_columns(null_frame):
    frame, truth = null_frame
    assert len(frame) == truth["n_days"]
    assert list(frame.columns) == ["ret", "lunar_phase", "lunar_label"]
    assert frame["ret"].notna().all()
    assert frame["lunar_phase"].between(0.0, 1.0, inclusive="left").all()
    assert set(frame["lunar_label"].unique()) <= {"NEW", "FULL", "OTHER"}


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_lunar(n_days=500, seed=7)
    b, _ = data.synthetic_lunar(n_days=500, seed=7)
    assert np.allclose(a["ret"].to_numpy(), b["ret"].to_numpy())
    c, _ = data.synthetic_lunar(n_days=500, seed=8)
    assert not np.allclose(a["ret"].to_numpy(), c["ret"].to_numpy())


def test_null_has_no_planted_effect(null_frame):
    """On the null tape, the planted discount/premium knobs are zero."""
    _, truth = null_frame
    assert truth["full_discount"] == 0.0
    assert truth["new_premium"] == 0.0


def test_label_counts_plausible(null_frame):
    """NEW + FULL together cover roughly 2 * window_days / 29.53 of trading days.

    With window_days=7 the fraction is ~47% each side, so together they cover
    up to ~95% of trading days — almost all days fall in one window or the other,
    leaving only a thin OTHER band.  This is expected behaviour: two overlapping
    14-day spans in a 29.53-day cycle.
    """
    frame, truth = null_frame
    frac_new = (frame["lunar_label"] == "NEW").mean()
    frac_full = (frame["lunar_label"] == "FULL").mean()
    # Each window is 14/29.53 ≈ 47% → together up to 95%, minimum ~40% each
    assert 0.30 < frac_new < 0.65
    assert 0.30 < frac_full < 0.65


def test_fetch_panel_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_panel(fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_is_stable_and_content_sensitive(null_frame):
    frame, _ = null_frame
    assert data.fingerprint(frame) == data.fingerprint(frame)
    other, _ = data.synthetic_lunar(n_days=3000, seed=99)
    assert data.fingerprint(frame) != data.fingerprint(other)


# ---------------------------------------------------------------------------
# Lunar phase engine
# ---------------------------------------------------------------------------
def test_lunar_phase_range():
    """Phase is always in [0, 1)."""
    import pandas as pd
    idx = pd.bdate_range("2000-01-01", periods=1000, name="date")
    phase = data.lunar_phase(idx)
    assert (phase >= 0.0).all() and (phase < 1.0).all()


def test_known_new_moon():
    """2000-01-06 is very close to the epoch new moon — phase should be near 0 or 1.

    The epoch new moon is at JD 2451551.26 (6 Jan 2000 18:14 UTC); we compute
    the phase at noon UTC (JD 2451550.0), which is ~1.26 days *before* the epoch,
    so phase ≈ 1 - 1.26/29.53 ≈ 0.957 — correctly very close to a new moon.
    """
    import pandas as pd
    idx = pd.DatetimeIndex(["2000-01-06"], name="date")
    phase = data.lunar_phase(idx)
    p = float(phase.iloc[0])
    # Near 0 (just after new moon) OR near 1 (just before new moon) — both are "new moon"
    assert p < 0.10 or p > 0.90


def test_full_moon_roughly_half_cycle_later():
    """~15 days after a new moon the phase should be near 0.5."""
    import pandas as pd
    # 2000-01-06 = new moon epoch; +15 calendar days = 2000-01-21
    idx = pd.DatetimeIndex(["2000-01-21"], name="date")
    phase = data.lunar_phase(idx)
    assert 0.40 < float(phase.iloc[0]) < 0.60


def test_lunar_label_has_all_three_values():
    """A long history should produce all three labels."""
    import pandas as pd
    idx = pd.bdate_range("1990-01-01", periods=2000, name="date")
    labels = data.lunar_label(idx)
    assert set(labels.unique()) == {"NEW", "FULL", "OTHER"}


def test_planted_effect_shifts_mean(signal_frame, null_frame):
    """Planting full_discount and new_premium shifts means in the right direction."""
    sig_frame, _ = signal_frame
    null, _ = null_frame
    # On the signal tape, full-moon days should have lower mean than on the null
    sig_full = sig_frame.loc[sig_frame["lunar_label"] == "FULL", "ret"].mean()
    null_full = null.loc[null["lunar_label"] == "FULL", "ret"].mean()
    assert sig_full < null_full
    # And new-moon days should have higher mean
    sig_new = sig_frame.loc[sig_frame["lunar_label"] == "NEW", "ret"].mean()
    null_new = null.loc[null["lunar_label"] == "NEW", "ret"].mean()
    assert sig_new > null_new
