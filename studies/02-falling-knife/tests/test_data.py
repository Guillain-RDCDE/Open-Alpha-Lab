"""Return-decomposition identities and OHLC sanity."""

import numpy as np

from falling_knife import data


def test_intraday_overnight_identity(synth_ohlc):
    """(1+overnight)(1+intraday) must equal (1+close_to_close) exactly."""
    ret = data.daily_returns(synth_ohlc)
    lhs = (1 + ret["r_co"]) * (1 + ret["r_oc"])
    rhs = 1 + ret["r_cc"]
    valid = ret["r_cc"].notna()
    assert np.allclose(lhs[valid], rhs[valid], atol=1e-12)


def test_first_return_is_nan(synth_ohlc):
    ret = data.daily_returns(synth_ohlc)
    assert np.isnan(ret["r_cc"].iloc[0])  # no prior close on day 0


def test_intraday_low_non_positive(synth_ohlc):
    """Open-to-low return can never be positive (low <= open by construction)."""
    ret = data.daily_returns(synth_ohlc)
    assert (ret["r_intraday_low"] <= 1e-12).all()


def test_ohlc_geometry(synth_ohlc):
    o, h, l, c = (synth_ohlc[k] for k in ("Open", "High", "Low", "Close"))
    assert (h >= o - 1e-9).all() and (h >= c - 1e-9).all()
    assert (l <= o + 1e-9).all() and (l <= c + 1e-9).all()
