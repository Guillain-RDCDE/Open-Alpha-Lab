"""The synthetic panel is deterministic; the long-low/short-high accruals hedge pays only when accruals
really predict; the null shows nothing; the hedge is long-minus-short; the accruals ratio is computed
correctly. All offline on the seeded synthetic world."""

import numpy as np
import pandas as pd

from smoke_screen import data, strategy as st


def test_world_deterministic(accruals_world):
    sig, fwd, truth = accruals_world
    sig2, fwd2, _ = data.synthetic_panel(accruals_premium=0.06, seed=52)
    assert np.allclose(sig.to_numpy(), sig2.to_numpy())
    assert np.allclose(fwd.to_numpy(), fwd2.to_numpy())
    assert truth.has_premium


def test_low_accruals_hedge_pays(accruals_world):
    sig, fwd, _ = accruals_world
    s = st.summary(st.quantile_hedge(sig, fwd, long_high=False)["hedge"])   # long low accruals
    assert s["mean"] > 0.0
    assert s["sharpe"] > 0.3


def test_null_world_no_hedge(null_world):
    sig, fwd, _ = null_world
    assert abs(st.summary(st.quantile_hedge(sig, fwd, long_high=False)["hedge"])["mean"]) < 0.03


def test_hedge_is_long_minus_short(accruals_world):
    sig, fwd, _ = accruals_world
    h = st.quantile_hedge(sig, fwd, long_high=False)
    assert np.allclose(h["hedge"], h["low"] - h["high"])   # long low, short high


def test_accruals_formula():
    years = [2010]
    ni = pd.DataFrame({"A": [10.0]}, index=pd.Index(years, name="year"))
    cfo = pd.DataFrame({"A": [7.0]}, index=pd.Index(years, name="year"))
    assets = pd.DataFrame({"A": [100.0]}, index=pd.Index(years, name="year"))
    acc = st.accruals(ni, cfo, assets)
    assert np.isclose(acc.loc[2010, "A"], 0.03)   # (10-7)/100
