"""The synthetic panel is deterministic; the long-high/short-low F-score hedge pays only when high-F
firms outperform; the null shows nothing; the F-score is bounded 0–9 and computed from the nine
components; the hedge is long minus short."""
import numpy as np, pandas as pd
from scorecard import data, strategy as st


def test_world_deterministic(fscore_world):
    sig, fwd, truth = fscore_world
    sig2, fwd2, _ = data.synthetic_panel(fscore_premium=0.05, seed=65)
    assert np.allclose(sig.to_numpy(), sig2.to_numpy())
    assert np.allclose(fwd.to_numpy(), fwd2.to_numpy())
    assert truth.has_premium


def test_fscore_world_hedge_pays(fscore_world):
    sig, fwd, _ = fscore_world
    s = st.summary(st.quantile_hedge(sig, fwd, long_high=True)["hedge"])   # long high F
    assert s["mean"] > 0.0
    assert s["sharpe"] > 0.3


def test_null_world_no_hedge(null_world):
    sig, fwd, _ = null_world
    assert abs(st.summary(st.quantile_hedge(sig, fwd, long_high=True)["hedge"])["mean"]) < 0.03


def test_hedge_is_long_minus_short(fscore_world):
    sig, fwd, _ = fscore_world
    h = st.quantile_hedge(sig, fwd, long_high=True)
    assert np.allclose(h["hedge"], h["high"] - h["low"])


def test_fscore_bounded_and_assembled():
    yrs = pd.Index([2010, 2011], name="year")
    one = lambda v: pd.DataFrame({"A": [v, v]}, index=yrs)
    p = {c: one(1.0) for c in st.CONCEPTS}
    p["Assets"] = one(100.0); p["Revenues"] = one(50.0); p["LiabilitiesCurrent"] = one(10.0)
    f = st.piotroski_fscore(p)
    assert f.loc[2011, "A"] >= 0 and f.loc[2011, "A"] <= 9
