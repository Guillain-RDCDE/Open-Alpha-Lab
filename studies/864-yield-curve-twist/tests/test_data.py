"""The synthetic tape is well-formed and deterministic; the real fetch is cache-safe."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from curve_twist import data  # noqa: E402


def test_synthetic_shape_and_columns(null_tape):
    df, truth = null_tape
    assert len(df) == truth["n_days"]
    need = {"FVX", "TNX", "TYX", "IEF_close", "TLT_close", "SPY_close",
            "fly", "dfly", "slope", "level", "IEF_ret", "TLT_ret", "SPY_ret"}
    assert need.issubset(set(df.columns))


def test_synthetic_is_deterministic():
    a, _ = data.synthetic_daily(n_days=600, fly_signal=0.0, seed=7)
    b, _ = data.synthetic_daily(n_days=600, fly_signal=0.0, seed=7)
    assert np.allclose(a["IEF_close"].to_numpy(), b["IEF_close"].to_numpy())
    assert np.allclose(a["fly"].to_numpy(), b["fly"].to_numpy())
    c, _ = data.synthetic_daily(n_days=600, fly_signal=0.0, seed=8)
    assert not np.allclose(a["IEF_close"].to_numpy(), c["IEF_close"].to_numpy())


def test_butterfly_identity(null_tape):
    df, _ = null_tape
    fly = 2.0 * df["TNX"] - df["FVX"] - df["TYX"]
    assert np.allclose(df["fly"].to_numpy(), fly.to_numpy())
    # slope control is the 5s10s spread
    assert np.allclose(df["slope"].to_numpy(), (df["TNX"] - df["FVX"]).to_numpy())
    # dfly is the day-over-day change (twist), first entry NaN
    assert np.isnan(df["dfly"].iloc[0])
    assert np.allclose(df["dfly"].to_numpy()[1:], np.diff(df["fly"].to_numpy()))


def test_synthetic_prices_positive_and_yields_sane(null_tape):
    df, _ = null_tape
    for c in ("IEF_close", "TLT_close", "SPY_close"):
        assert (df[c] > 0).all()
    for c in ("FVX", "TNX", "TYX"):
        assert (df[c] > 0).all()
        assert (df[c] < 10.0).all()


def test_synthetic_index_is_business_days(null_tape):
    df, _ = null_tape
    assert (df.index.dayofweek < 5).all()


def test_butterfly_wanders(null_tape):
    """The reconstructed butterfly must have real day-to-day variation for the sort/
    regression to have something to bite on."""
    df, _ = null_tape
    assert df["fly"].std() > 0.02


def test_signal_knob_shifts_forward_returns():
    """A planted fly_signal > 0 makes high-butterfly days precede higher IEF returns."""
    null, _ = data.synthetic_daily(n_days=4000, fly_signal=0.0, seed=42)
    sig, _ = data.synthetic_daily(n_days=4000, fly_signal=0.02, seed=42)
    # the correlation of lagged fly rank with next-day IEF return rises with the knob
    import pandas as pd
    def corr(df):
        rank = df["fly"].rolling(252, min_periods=63).rank(pct=True).shift(1)
        return df["IEF_ret"].corr(rank)
    assert corr(sig) > corr(null)


@pytest.mark.skipif(not os.path.exists(data.cache_path()),
                    reason="real cache absent offline CI")
def test_real_cache_loads_and_has_butterfly():
    df = data.load_daily()
    assert len(df) > 1000
    assert {"fly", "dfly", "slope", "IEF_ret"}.issubset(set(df.columns))
    assert df.index.max().year <= 2026
    # the butterfly identity holds on the real tape too
    assert np.allclose(df["fly"].to_numpy(),
                       (2 * df["TNX"] - df["FVX"] - df["TYX"]).to_numpy())
