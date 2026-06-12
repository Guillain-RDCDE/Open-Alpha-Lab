"""The synthetic panel is deterministic; the long-buyback/short-issuer hedge pays only when high issuers
underperform; the null shows nothing; net_issuance is a pct change; the hedge is long minus short."""
import numpy as np, pandas as pd
from share_shuffle import data, strategy as st


def test_world_deterministic(issuance_world):
    sig, fwd, truth = issuance_world
    sig2, fwd2, _ = data.synthetic_panel(issuance_premium=0.06, seed=64)
    assert np.allclose(sig.to_numpy(), sig2.to_numpy())
    assert np.allclose(fwd.to_numpy(), fwd2.to_numpy())
    assert truth.has_premium


def test_issuance_world_hedge_pays(issuance_world):
    sig, fwd, _ = issuance_world
    s = st.summary(st.quantile_hedge(sig, fwd, long_high=False)["hedge"])   # long low issuance
    assert s["mean"] > 0.0
    assert s["sharpe"] > 0.3


def test_null_world_no_hedge(null_world):
    sig, fwd, _ = null_world
    assert abs(st.summary(st.quantile_hedge(sig, fwd, long_high=False)["hedge"])["mean"]) < 0.03


def test_hedge_is_long_minus_short(issuance_world):
    sig, fwd, _ = issuance_world
    h = st.quantile_hedge(sig, fwd, long_high=False)
    assert np.allclose(h["hedge"], h["low"] - h["high"])


def test_net_issuance_is_pct_change():
    sh = pd.DataFrame({"A": [100.0, 110.0, 99.0]}, index=pd.Index([2010, 2011, 2012], name="year"))
    iss = st.net_issuance(sh)
    assert np.isclose(iss.loc[2011, "A"], 0.10)
    assert np.isclose(iss.loc[2012, "A"], -0.10)
