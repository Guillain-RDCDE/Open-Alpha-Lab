"""The synthetic bars are deterministic and internally consistent, and the IBS signal recovers the
baked reversal on one tape and finds nothing on the null."""

import numpy as np

from rubber_band import data, ibs


def test_deterministic_and_consistent_bars(reversal):
    ohlc, truth = reversal
    ohlc2, _ = data.synthetic_ohlc(kappa=0.0035, seed=19)
    assert np.allclose(ohlc.to_numpy(), ohlc2.to_numpy())
    # OHLC sanity: Low <= Close <= High and Low <= Open <= High
    assert (ohlc["Low"] <= ohlc["Close"] + 1e-9).all()
    assert (ohlc["Close"] <= ohlc["High"] + 1e-9).all()
    assert (ohlc["Low"] <= ohlc["Open"] + 1e-9).all() and (ohlc["Open"] <= ohlc["High"] + 1e-9).all()
    assert truth.has_reversal


def test_ibs_in_unit_interval(reversal_ohlc):
    v = ibs.ibs(reversal_ohlc)
    assert v.min() >= 0.0 and v.max() <= 1.0
    # matches the definition exactly on a clean bar
    o = reversal_ohlc.iloc[10]
    manual = (o["Close"] - o["Low"]) / (o["High"] - o["Low"])
    assert abs(v.iloc[10] - manual) < 1e-9


def test_reversal_recovered(reversal_ohlc):
    rs = ibs.reversal_strength(reversal_ohlc)
    assert rs["ibs_slope"] < 0                       # low IBS -> higher next-day return
    assert rs["low_minus_high_bps"] > 5              # a clear low-minus-high next-day spread
    assert rs["reversal_present"]


def test_reversal_absent_on_null(null_ohlc):
    rs = ibs.reversal_strength(null_ohlc)
    assert abs(rs["low_minus_high_bps"]) < 5         # random walk: IBS predicts nothing
    assert abs(rs["ibs_slope"]) < 1e-3
