"""Data-layer tests — synthetic determinism (offline) + a skipif real-cache smoke test."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from t_plus_one import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic determinism & shape (offline)
# --------------------------------------------------------------------------- #
def test_synthetic_is_deterministic():
    a, _ = data.synthetic_panel(seed=926)
    b, _ = data.synthetic_panel(seed=926)
    for key in ("treated", "control", "cash"):
        assert np.allclose(a[key].to_numpy(), b[key].to_numpy())


def test_synthetic_seed_changes_the_world():
    a, _ = data.synthetic_panel(seed=926)
    b, _ = data.synthetic_panel(seed=927)
    assert not np.allclose(a["treated"]["close"].to_numpy(), b["treated"]["close"].to_numpy())


def test_synthetic_shape_and_columns():
    panel, truth = data.synthetic_panel(n_days=800, seed=926)
    assert set(panel) == {"treated", "control", "cash"}
    for key in ("treated", "control"):
        assert {"open", "high", "low", "close"}.issubset(panel[key].columns)
        assert isinstance(panel[key].index, pd.DatetimeIndex)
        assert len(panel[key]) == truth["n_days"] == 800
        # OOB-safe: the synthetic index must stay inside pandas' nanosecond horizon.
        assert panel[key].index[-1] < pd.Timestamp("2262-01-01")


def test_synthetic_high_low_bracket_open_close():
    panel, _ = data.synthetic_panel(n_days=600, seed=926)
    for key in ("treated", "control"):
        df = panel[key]
        assert (df["high"] >= df[["open", "close"]].max(axis=1) - 1e-9).all()
        assert (df["low"] <= df[["open", "close"]].min(axis=1) + 1e-9).all()


def test_synthetic_cash_is_monotone_growing():
    panel, _ = data.synthetic_panel(seed=926)
    assert (panel["cash"]["close"].diff().dropna() > 0).all()


def test_signal_strength_zero_plants_nothing():
    _, t1 = data.synthetic_panel(signal_strength=1.0, seed=926)
    _, t0 = data.synthetic_panel(signal_strength=0.0, seed=926)
    assert t0["planted_on_drift_bps"] == 0.0
    assert t0["planted_tom_bps"] == 0.0
    assert t1["planted_on_drift_bps"] > 0.0


def test_null_control_leg_is_identical_across_signal_strength():
    """The control asset must not move when the planted break is switched on."""
    a, _ = data.synthetic_panel(signal_strength=1.0, seed=926)
    b, _ = data.synthetic_panel(signal_strength=0.0, seed=926)
    assert np.allclose(a["control"]["close"].to_numpy(), b["control"]["close"].to_numpy())


def test_synthetic_daily_is_the_treated_leg():
    ohlc, truth = data.synthetic_daily(n_days=500, seed=926)
    panel, _ = data.synthetic_panel(n_days=500, seed=926)
    assert np.allclose(ohlc.to_numpy(), panel["treated"].to_numpy())
    assert truth["n_days"] == 500


def test_fingerprint_stable_and_sensitive():
    a, _ = data.synthetic_panel(seed=926)
    fp1 = data.fingerprint(a["treated"])
    fp2 = data.fingerprint(a["treated"])
    b, _ = data.synthetic_panel(seed=927)
    assert fp1 == fp2 and len(fp1) == 12
    assert fp1 != data.fingerprint(b["treated"])


def test_loaders_raise_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_ohlc(cache_dir=str(tmp_path))
    with pytest.raises(FileNotFoundError):
        data.load_prices(cache_dir=str(tmp_path))
    assert not data.have_real(cache_dir=str(tmp_path))


def test_constants_are_the_documented_event():
    assert data.SWITCH_DATE == "2024-05-28"
    assert data.AS_OF == "2026-06-30"
    assert data.CONTROL == "SPY" and set(data.TREATED) == {"EFA", "EEM"}


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped entirely when the shared cache is absent (CI safe)
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not data.have_real(),
                    reason="no shared _cache present (offline / CI) — synthetic tests cover the logic")
def test_real_cache_did_runs():
    panel = data.load_ohlc()
    for tk, df in panel.items():
        assert df.index[-1] <= pd.Timestamp(data.AS_OF), tk
    tab = st.did_table(panel, "EFA", data.CONTROL, data.SWITCH_DATE, data.AS_OF)
    for col in ("r_on", "abs_on", "on_var_share", "abs_cc"):
        assert np.isfinite(tab[col]["did"])
        assert np.isfinite(tab[col]["did_t"])
    assert tab["_windows"]["n_each"] > 200
