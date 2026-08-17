"""Data-layer tests — synthetic determinism (offline) + a skipif real-cache smoke test."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from reset_freq import data, strategy as st  # noqa: E402


# --------------------------------------------------------------------------- #
# Synthetic determinism & shape (offline)
# --------------------------------------------------------------------------- #
def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(seed=943)
    b, _ = data.synthetic_daily(seed=943)
    assert np.allclose(a["asset"].to_numpy(), b["asset"].to_numpy())
    assert np.allclose(a["cash"].to_numpy(), b["cash"].to_numpy())


def test_synthetic_shape_and_columns():
    prices, truth = data.synthetic_daily(n_years=10, seed=943)
    assert {"asset", "cash"}.issubset(prices.columns)
    assert isinstance(prices.index, pd.DatetimeIndex)
    assert len(prices) == truth["n_days"] == 10 * data.TRADING_DAYS_PER_YEAR
    # OOB-safe: the synthetic index must stay inside pandas' ns Timestamp horizon.
    assert prices.index[-1] < pd.Timestamp("2262-01-01")


def test_synthetic_cash_is_monotone_growing():
    prices, _ = data.synthetic_daily(n_years=10, seed=943)
    assert (prices["cash"].diff().dropna() > 0).all()
    assert prices["cash"].iloc[-1] > prices["cash"].iloc[0]


def test_choppiness_knob_signs_the_autocorrelation():
    _, chop = data.synthetic_daily(phi=-0.15, n_years=10, seed=943)
    _, trend = data.synthetic_daily(phi=+0.15, n_years=10, seed=943)
    assert chop["realised_ac1"] < -0.05
    assert trend["realised_ac1"] > +0.05


def test_signal_strength_zero_is_the_iid_null():
    _, t0 = data.synthetic_daily(phi=-0.15, signal_strength=0.0, n_years=10, seed=943)
    assert t0["phi_eff"] == 0.0
    assert abs(t0["realised_ac1"]) < 0.05


def test_choppiness_knob_holds_variance_fixed():
    """Turning the knob must change path shape, not total volatility."""
    vols = [data.synthetic_daily(phi=p, n_years=10, seed=943)[1]["realised_vol_ann"]
            for p in (-0.15, 0.0, 0.15)]
    assert max(vols) - min(vols) < 0.01     # within 1 vol point of each other


def test_synthetic_panel_keys_and_determinism():
    panel = data.synthetic_panel(phis=(-0.1, 0.1), n_seeds=2, n_years=5)
    assert set(panel.keys()) == {(-0.1, 943), (-0.1, 944), (0.1, 943), (0.1, 944)}
    again = data.synthetic_panel(phis=(-0.1, 0.1), n_seeds=2, n_years=5)
    for k in panel:
        assert np.allclose(panel[k][0]["asset"].to_numpy(), again[k][0]["asset"].to_numpy())


def test_fingerprint_stable_and_sensitive():
    a, _ = data.synthetic_daily(n_years=5, seed=943)
    b, _ = data.synthetic_daily(n_years=5, seed=944)
    fp1 = data.fingerprint(a)
    assert fp1 == data.fingerprint(a) and len(fp1) == 12
    assert fp1 != data.fingerprint(b)


def test_safe_name_strips_yahoo_punctuation():
    assert data.safe_name("^IRX") == "IRX"
    assert data.safe_name("SPY") == "SPY"


def test_load_prices_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.load_prices(cache_dir=str(tmp_path))


# --------------------------------------------------------------------------- #
# Real-cache smoke test — skipped entirely when the shared cache is absent (CI safe)
# --------------------------------------------------------------------------- #
CACHE = data.DEFAULT_CACHE


@pytest.mark.skipif(not data.have_real(cache_dir=CACHE),
                    reason="no shared _cache present (offline / CI) — synthetic tests cover the logic")
def test_real_cache_race_runs():
    px = data.load_prices()
    assert px.index[-1] <= pd.Timestamp(data.AS_OF)
    frame = st.build_frame(px, "SSO")
    r = st.race(frame, 2.0)
    for k in ("sharpe_adv", "t_diff", "fee_leg_ann_pct", "w_max"):
        assert np.isfinite(r[k])
    # A 2x sleeve must be far more volatile than the 1x index it levers.
    assert r["abs_daily"]["vol_ann"] > 1.5 * r["abs_asset"]["vol_ann"]
