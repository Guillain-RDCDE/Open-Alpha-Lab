"""Data-layer tests — synthetic determinism (offline) + a skipif real-cache smoke test."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from front_end_trend import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic determinism & shape (offline)
# --------------------------------------------------------------------------- #
def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(seed=925)
    b, _ = data.synthetic_daily(seed=925)
    for col in ("rate", "long", "mid", "cash"):
        assert np.allclose(a[col].to_numpy(), b[col].to_numpy())


def test_synthetic_shape_and_columns():
    frame, truth = data.synthetic_daily(n_years=12, seed=925)
    assert {"rate", "long", "mid", "cash"}.issubset(frame.columns)
    assert isinstance(frame.index, pd.DatetimeIndex)
    assert len(frame) == truth["n_days"] == 12 * data.TRADING_DAYS_PER_YEAR
    # OOB-safe: the synthetic index must stay inside pandas' ns horizon.
    assert frame.index[-1] < pd.Timestamp("2262-01-01")


def test_synthetic_rate_stays_in_a_plausible_box():
    frame, _ = data.synthetic_daily(seed=925)
    assert frame["rate"].min() >= 0.0
    assert frame["rate"].max() <= 8.0


def test_synthetic_cash_accrues_and_never_falls():
    frame, _ = data.synthetic_daily(seed=925)
    assert (frame["cash"].diff().dropna() >= 0).all()
    assert frame["cash"].iloc[-1] > frame["cash"].iloc[0]


def test_signal_strength_controls_rate_increment_persistence():
    _, hi = data.synthetic_daily(signal_strength=1.0, seed=925)
    _, lo = data.synthetic_daily(signal_strength=0.0, seed=925)
    assert hi["phi"] > 0.8 and lo["phi"] == 0.0
    assert hi["dy_autocorr1"] > 0.5
    assert abs(lo["dy_autocorr1"]) < 0.1


def test_unconditional_rate_vol_is_held_fixed_across_the_knob():
    """The knob must change predictability, not risk — otherwise the control is unfair."""
    hi, _ = data.synthetic_daily(signal_strength=1.0, seed=925)
    lo, _ = data.synthetic_daily(signal_strength=0.0, seed=925)
    sd_hi = float(hi["rate"].diff().dropna().std(ddof=1))
    sd_lo = float(lo["rate"].diff().dropna().std(ddof=1))
    assert 0.6 < sd_hi / sd_lo < 1.7


def test_duration_leg_is_more_volatile_than_the_intermediate_leg():
    frame, truth = data.synthetic_daily(seed=925)
    v_long = frame["long"].pct_change().std(ddof=1)
    v_mid = frame["mid"].pct_change().std(ddof=1)
    assert v_long > v_mid
    assert truth["dur_long"] > truth["dur_mid"]


def test_synthetic_panel_is_distinct_per_seed():
    panel = data.synthetic_panel(n_seeds=3, signal_strength=0.0)
    assert len(panel) == 3
    fps = {data.fingerprint(f) for f, _ in panel}
    assert len(fps) == 3


def test_fingerprint_stable_and_sensitive():
    a, _ = data.synthetic_daily(seed=925)
    b, _ = data.synthetic_daily(seed=926)
    fp = data.fingerprint(a)
    assert fp == data.fingerprint(a) and len(fp) == 12
    assert fp != data.fingerprint(b)


def test_load_prices_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_prices(cache_dir=str(tmp_path))


def test_cache_path_strips_the_caret():
    """^IRX must land on prices_IRX_1d.parquet — the shared-cache naming convention."""
    p = data._cache_path("^IRX", "/tmp/x")
    assert os.path.basename(p) == "prices_IRX_1d.parquet"


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped entirely when the cache is absent (CI safe)
# --------------------------------------------------------------------------- #
CACHE = data.DEFAULT_CACHE


@pytest.mark.skipif(not data.have_real(cache_dir=CACHE),
                    reason="no real _cache present (offline / CI) — synthetic tests cover the logic")
def test_real_cache_race_runs():
    px = data.load_prices().dropna()
    assert px.index[-1] <= pd.Timestamp(data.AS_OF)
    cmp = st.compare_rate_rule(px["^IRX"], px["TLT"], px["IEF"], px["BIL"])
    for k in ("excess_sharpe_adv", "t_diff_vs_mid", "dd_switch_abs", "dd_mid_abs"):
        assert np.isfinite(cmp[k])
    # TLT carries far more duration risk than IEF — a basic tape sanity check.
    assert cmp["static_long"]["vol_ann"] > cmp["static_mid"]["vol_ann"]
