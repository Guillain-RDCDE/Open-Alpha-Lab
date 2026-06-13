"""The synthetic tape is well-formed and deterministic; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dollar_smile import data  # noqa: E402


def test_synthetic_shape_and_columns(null_tape):
    daily, truth = null_tape
    assert len(daily) == truth["n_years"] * 252
    assert list(daily.columns) == ["dollar", "spy", "eem"]
    assert (daily["dollar"] > 0).all()
    assert (daily["spy"] > 0).all()
    assert (daily["eem"] > 0).all()


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_years=10, pred_r=0.0, seed=5)
    b, _ = data.synthetic_daily(n_years=10, pred_r=0.0, seed=5)
    assert np.allclose(a["spy"].to_numpy(), b["spy"].to_numpy())
    c, _ = data.synthetic_daily(n_years=10, pred_r=0.0, seed=6)
    assert not np.allclose(a["spy"].to_numpy(), c["spy"].to_numpy())


def test_pred_r_creates_cross_correlation():
    """The pred_r knob really induces predictive cross-correlation between dollar and equity."""
    flat, _ = data.synthetic_daily(n_years=20, pred_r=0.0, contemp_r=0.0, seed=114)
    linked, _ = data.synthetic_daily(n_years=20, pred_r=-0.8, contemp_r=-0.8, seed=114)

    # Compute weekly log returns and lag-1 cross-correlation
    import pandas as pd

    def weekly_xcorr(df):
        w = df.resample("W").last()
        dr = np.log(w["dollar"]).diff().dropna()
        er = np.log(w["spy"]).diff().dropna()
        # align
        idx = dr.index.intersection(er.index)
        x = dr.loc[idx].to_numpy()
        y = er.loc[idx].to_numpy()
        # lag-1: x[:-1] predicts y[1:]
        return np.corrcoef(x[:-1], y[1:])[0, 1]

    xcorr_flat = abs(weekly_xcorr(flat))
    xcorr_linked = abs(weekly_xcorr(linked))
    # The linked tape should have stronger predictive correlation
    assert xcorr_linked > xcorr_flat


def test_contemp_r_creates_contemporaneous_correlation():
    """The contemp_r knob induces same-period dollar-equity co-movement."""
    flat, _ = data.synthetic_daily(n_years=20, pred_r=0.0, contemp_r=0.0, seed=114)
    contemp, _ = data.synthetic_daily(n_years=20, pred_r=0.0, contemp_r=-0.8, seed=114)

    import pandas as pd

    def weekly_contemp_corr(df):
        w = df.resample("W").last()
        dr = np.log(w["dollar"]).diff().dropna()
        er = np.log(w["spy"]).diff().dropna()
        idx = dr.index.intersection(er.index)
        return np.corrcoef(dr.loc[idx].to_numpy(), er.loc[idx].to_numpy())[0, 1]

    corr_flat = abs(weekly_contemp_corr(flat))
    corr_contemp = abs(weekly_contemp_corr(contemp))
    assert corr_contemp > corr_flat


def test_fetch_cache_only_raises_without_cache(tmp_path):
    with pytest.raises(FileNotFoundError):
        data.fetch_daily("NOPE", fetch=False, cache_dir=str(tmp_path))


def test_fingerprint_stable_and_content_sensitive(null_tape):
    daily, _ = null_tape
    assert data.fingerprint(daily) == data.fingerprint(daily)
    other, _ = data.synthetic_daily(n_years=20, pred_r=0.0, seed=99)
    assert data.fingerprint(daily) != data.fingerprint(other)
