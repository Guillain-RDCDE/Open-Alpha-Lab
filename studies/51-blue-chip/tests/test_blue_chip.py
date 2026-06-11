"""The synthetic panel is deterministic; the long-high/short-low quality hedge pays only when quality
really predicts; the null shows nothing; the hedge is long-minus-short; gross profitability divides
correctly. All offline on the seeded synthetic world."""

import numpy as np
import pandas as pd

from blue_chip import data, strategy as st


def test_world_deterministic(quality_world):
    sig, fwd, truth = quality_world
    sig2, fwd2, _ = data.synthetic_panel(quality_premium=0.06, seed=51)
    assert np.allclose(sig.to_numpy(), sig2.to_numpy())
    assert np.allclose(fwd.to_numpy(), fwd2.to_numpy())
    assert truth.has_premium


def test_quality_world_hedge_pays(quality_world):
    sig, fwd, _ = quality_world
    s = st.summary(st.quantile_hedge(sig, fwd, long_high=True)["hedge"])
    assert s["mean"] > 0.0
    assert s["sharpe"] > 0.3


def test_null_world_no_hedge(null_world):
    sig, fwd, _ = null_world
    assert abs(st.summary(st.quantile_hedge(sig, fwd, long_high=True)["hedge"])["mean"]) < 0.03


def test_hedge_is_long_minus_short(quality_world):
    sig, fwd, _ = quality_world
    h = st.quantile_hedge(sig, fwd, long_high=True)
    assert np.allclose(h["hedge"], h["high"] - h["low"])


def test_gross_profitability_divides():
    years = [2010, 2011]
    gp = pd.DataFrame({"A": [10.0, 12.0], "B": [5.0, 6.0]}, index=pd.Index(years, name="year"))
    assets = pd.DataFrame({"A": [100.0, 100.0], "B": [50.0, 50.0]}, index=pd.Index(years, name="year"))
    gpa = st.gross_profitability(gp, assets)
    assert np.isclose(gpa.loc[2010, "A"], 0.10)
    assert np.isclose(gpa.loc[2011, "B"], 0.12)
