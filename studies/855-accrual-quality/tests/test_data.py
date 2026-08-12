"""Data-layer invariants: the Dechow-Dichev residual-vol construction (clean accruals → low
vol, noisy accruals → high vol), point-in-time stamping, and the synthetic panel's shape —
all offline, fixed seeds."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from accrual_quality import data  # noqa: E402


def _make_frame(cfo_scaled, accr_scaled, assets=1000.0, start="2010-03-31"):
    """Build an aligned quarterly frame with consecutive ~quarterly ends where the scaled CFO
    and scaled accrual are prescribed. NI is back-solved from accr = (NI-CFO)/avgTA (avgTA is
    ``assets`` since assets are held constant)."""
    n = len(cfo_scaled)
    ends = pd.date_range(start, periods=n, freq="QE")
    filed = ends + pd.Timedelta(days=35)
    cfo = np.asarray(cfo_scaled, dtype=float) * assets
    accr = np.asarray(accr_scaled, dtype=float) * assets    # NI - CFO, dollars
    ni = cfo + accr
    return pd.DataFrame({"end": ends, "filed": filed, "ni": ni, "cfo": cfo,
                         "assets": float(assets), "ar": np.nan, "inv": np.nan})


def test_clean_accruals_have_low_residual_vol():
    # accruals a perfectly linear function of CFO -> DD residual ~ 0 -> low aq_vol
    rng = np.random.default_rng(0)
    x = rng.normal(0.0, 0.03, 24)
    accr = 0.5 * x                                   # exact linear map, no noise
    fr = _make_frame(x, accr)
    sig = data.build_signal(fr, window=12, min_obs=8)
    assert len(sig) > 0
    assert np.nanmedian(sig["aq_vol"].to_numpy()) < 1e-6


def test_noisy_accruals_have_higher_residual_vol():
    rng = np.random.default_rng(1)
    x = rng.normal(0.0, 0.03, 24)
    clean = _make_frame(x, 0.5 * x)
    noisy = _make_frame(x, 0.5 * x + rng.normal(0.0, 0.02, 24))
    v_clean = np.nanmedian(data.build_signal(clean)["aq_vol"].to_numpy())
    v_noisy = np.nanmedian(data.build_signal(noisy)["aq_vol"].to_numpy())
    assert v_noisy > v_clean


def test_quality_is_negative_residual_vol():
    rng = np.random.default_rng(2)
    x = rng.normal(0.0, 0.03, 24)
    fr = _make_frame(x, 0.5 * x + rng.normal(0.0, 0.02, 24))
    sig = data.build_signal(fr)
    assert np.allclose(sig["quality"].to_numpy(), -sig["aq_vol"].to_numpy(), equal_nan=True)


def test_signal_is_point_in_time_filed_after_end():
    rng = np.random.default_rng(3)
    x = rng.normal(0.0, 0.03, 24)
    fr = _make_frame(x, 0.5 * x + rng.normal(0.0, 0.02, 24))
    sig = data.build_signal(fr)
    assert (sig["filed"] > sig["end"]).all()


def test_build_signal_thin_frame_returns_empty():
    rng = np.random.default_rng(4)
    x = rng.normal(0.0, 0.03, 6)
    fr = _make_frame(x, 0.5 * x)
    sig = data.build_signal(fr, window=12, min_obs=8)
    assert len(sig) == 0


def test_synthetic_panel_shapes():
    prices, ev = data.synthetic_panel(n_names=12, n_quarters=24, edge=0.0, seed=1)
    assert prices.shape[1] == 12
    assert {"ticker", "end", "filed", "quality", "aq_vol"}.issubset(ev.columns)
    assert (ev["filed"] > ev["end"]).all()
    assert np.allclose(ev["aq_vol"].to_numpy(), -ev["quality"].to_numpy())
    assert len(ev) > 50


def test_synthetic_panel_deterministic():
    p1, e1 = data.synthetic_panel(n_names=8, n_quarters=20, edge=0.1, seed=42)
    p2, e2 = data.synthetic_panel(n_names=8, n_quarters=20, edge=0.1, seed=42)
    assert np.allclose(p1.to_numpy(), p2.to_numpy())
    assert np.allclose(e1["quality"].to_numpy(), e2["quality"].to_numpy())


def test_align_quarters_matches_on_period_end():
    ends = pd.date_range("2015-03-31", periods=6, freq="QE")
    ni = pd.DataFrame({"end": ends, "filed": ends + pd.Timedelta(days=30),
                       "val": np.arange(1.0, 7.0)})
    cfo = pd.DataFrame({"end": ends, "filed": ends + pd.Timedelta(days=30),
                        "val": np.arange(1.0, 7.0) * 2})
    assets = pd.DataFrame({"end": ends, "filed": ends + pd.Timedelta(days=30),
                           "val": np.full(6, 1000.0)})
    empty = pd.DataFrame(columns=["end", "filed", "val"])
    fr = data.align_quarters(ni, cfo, assets, empty, empty)
    assert len(fr) == 6
    assert (fr["cfo"] == fr["ni"] * 2).all()
