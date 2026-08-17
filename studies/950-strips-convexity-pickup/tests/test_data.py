"""Data-layer tests — synthetic determinism (offline) + a skipif real-cache smoke test.

Nothing here touches the network, and nothing here needs ``studies/_cache`` to exist: the
one test that reads the real tape is skipped outright when the cache is absent, which is
the state of a fresh CI checkout.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from zero_convexity import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic determinism & shape (offline)
# --------------------------------------------------------------------------- #
def test_synthetic_is_deterministic():
    a, _ = data.synthetic_panel(seed=950)
    b, _ = data.synthetic_panel(seed=950)
    for col in ("zero", "coupon", "cash", "yield_pp"):
        assert np.allclose(a[col].to_numpy(), b[col].to_numpy())


def test_synthetic_seed_changes_the_tape():
    a, _ = data.synthetic_panel(seed=950)
    b, _ = data.synthetic_panel(seed=951)
    assert not np.allclose(a["zero"].to_numpy(), b["zero"].to_numpy())


def test_synthetic_shape_and_columns():
    panel, truth = data.synthetic_panel(n_years=12, seed=950)
    assert {"zero", "coupon", "cash", "yield_pp"}.issubset(panel.columns)
    assert isinstance(panel.index, pd.DatetimeIndex)
    assert len(panel) == truth["n_days"] == 12 * data.TRADING_DAYS_PER_YEAR
    # OOB-safe: the synthetic index must stay well inside pandas' ns horizon.
    assert panel.index[-1] < pd.Timestamp("2262-01-01")


def test_synthetic_cash_is_monotone_growing():
    panel, _ = data.synthetic_panel(seed=950)
    assert (panel["cash"].diff().dropna() > 0).all()
    assert panel["cash"].iloc[-1] > panel["cash"].iloc[0]


def test_synthetic_prices_are_positive_and_yield_is_plausible():
    panel, _ = data.synthetic_panel(seed=950)
    assert (panel[["zero", "coupon"]] > 0).all().all()
    # A 30-year yield that wanders but stays in a recognisable band.
    assert panel["yield_pp"].between(-2.0, 12.0).mean() > 0.95


def test_zero_leg_is_more_volatile_than_the_coupon_leg():
    """The planted duration ordering must show up in the tape (24y vs 16.5y)."""
    panel, truth = data.synthetic_panel(seed=950)
    v_zero = panel["zero"].pct_change().std()
    v_coupon = panel["coupon"].pct_change().std()
    assert v_zero / v_coupon > 1.2
    assert truth["duration_ratio"] == pytest.approx(24.0 / 16.5)


def test_signal_strength_controls_the_convexity_gap_only():
    _, t1 = data.synthetic_panel(signal_strength=1.0, seed=950)
    _, t0 = data.synthetic_panel(signal_strength=0.0, seed=950)
    # At ss=0 convexity-per-unit-duration is identical for both legs (the null) ...
    assert t0["convexity_per_dur_long"] == pytest.approx(t0["convexity_per_dur_short"])
    # ... and at ss=1 the zero leg carries a genuine pickup, paid for with carry.
    assert t1["convexity_per_dur_long"] > t1["convexity_per_dur_short"]
    assert t1["carry_giveup_bp_mo"] > 0.0
    assert t0["carry_giveup_bp_mo"] == 0.0
    # Duration is untouched by the knob — only convexity and its price move.
    assert t0["dur_long"] == t1["dur_long"] and t0["dur_short"] == t1["dur_short"]


def test_synthetic_daily_alias_matches_panel():
    a, _ = data.synthetic_panel(n_years=6, seed=950)
    b, _ = data.synthetic_daily(n_years=6, seed=950)
    assert np.allclose(a["zero"].to_numpy(), b["zero"].to_numpy())


def test_fingerprint_stable_and_sensitive():
    a, _ = data.synthetic_panel(seed=950)
    b, _ = data.synthetic_panel(seed=951)
    fp = data.fingerprint(a)
    assert fp == data.fingerprint(a) and len(fp) == 12
    assert fp != data.fingerprint(b)


def test_column_name_strips_the_caret():
    assert data.column_name("^TYX") == "TYX"
    assert data.column_name("EDV") == "EDV"


def test_load_prices_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_prices(cache_dir=str(tmp_path))


def test_have_real_is_false_on_an_empty_dir(tmp_path):
    assert data.have_real(cache_dir=str(tmp_path)) is False


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped entirely when the shared cache is absent (CI safe)
# --------------------------------------------------------------------------- #
CACHE = data.DEFAULT_CACHE


@pytest.mark.skipif(not data.have_real(cache_dir=CACHE),
                    reason="no real _cache present (offline / CI) — synthetic tests cover the logic")
def test_real_cache_race_runs():
    px = data.load_prices()
    assert px.index[-1] <= pd.Timestamp(data.AS_OF)
    race = st.run_race(px["EDV"].dropna(), px["TLT"].dropna(),
                       px["BIL"].dropna(), px["TYX"].dropna())
    assert len(race) > 1000
    # The duration match is the whole premise: the two arms must have near-equal vol.
    ratio = race["e_zero"].std() / race["e_mix"].std()
    assert 0.9 < ratio < 1.1
    m = st.to_monthly(race, px["TYX"].dropna())
    reg = st.convexity_regression(m)
    for k in ("a_bp_mo", "b1", "b2", "b2_t"):
        assert np.isfinite(reg[k])
