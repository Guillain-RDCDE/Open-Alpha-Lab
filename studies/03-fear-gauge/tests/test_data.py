"""Data plumbing that doesn't touch the network: returns + vix_chg identities."""

import numpy as np
import pandas as pd

from fear_gauge import data


def test_daily_returns_identities(synth_market):
    out = data.daily_returns(synth_market)
    prev = synth_market["Close"].shift(1)
    assert np.allclose(out["r_cc"].dropna(), (synth_market["Close"] / prev - 1).dropna())
    assert np.allclose(out["r_oc"], synth_market["Close"] / synth_market["Open"] - 1)


def test_vix_chg_identity():
    idx = pd.bdate_range("2020-01-01", periods=4)
    close = pd.Series([20.0, 26.0, 26.0, 13.0], index=idx)
    df = pd.DataFrame({"Close": close})
    df["vix_prev"] = df["Close"].shift(1)
    df["vix_chg"] = df["Close"] / df["vix_prev"] - 1.0
    assert np.isclose(df["vix_chg"].iloc[1], 0.30)   # 20 -> 26 = +30%
    assert np.isclose(df["vix_chg"].iloc[3], -0.5)   # 26 -> 13 = -50%
