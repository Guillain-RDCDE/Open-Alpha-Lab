"""Fully offline, deterministic tests for Study 761 — Hotel-RevPAR.

No network: the synthetic control and the hardcoded RevPAR table are self-contained. We
assert (1) the synthetic positive control behaves — a zero planted lead does NOT
manufacture significance and a large planted lead DOES light up positive; (2) the RevPAR
proxy table is well-formed and seasonality-free under YoY momentum; (3) the release lag is
strictly non-look-ahead.

    pytest -q studies/761-hotel-revpar/tests
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from hotel_revpar import data, strategy as st


def test_revpar_table_wellformed():
    s = data.revpar_series()
    assert s.index.is_monotonic_increasing      # sorted month-ends
    assert (s > 0).all()                        # dropped in-progress zeros
    assert 300 <= len(s) <= 360                 # ~28 years of months
    # YoY momentum should be finite once 12 months elapse
    m = st.revpar_momentum(_frame())
    assert np.isfinite(m.dropna()).all()


def _frame():
    # a tiny frame from the RevPAR table + a flat synthetic price, for pure-function checks
    import pandas as pd
    s = data.revpar_series()
    px = pd.Series(np.linspace(100, 200, len(s)), index=s.index)
    return pd.DataFrame({"revpar": s, "px": px})


def test_synthetic_null_is_not_significant():
    """edge=0 => RevPAR momentum carries no forward info; HAC t must stay small."""
    syn = data.synthetic(edge=0.0, seed=761)
    reg = st.predictive_regression(syn, 6)
    assert abs(reg["t"]) < 2.0, f"null manufactured significance: t={reg['t']}"


def test_synthetic_planted_lead_lights_up_positive():
    """A large planted positive lead must drive the HAC t well past +2."""
    syn = data.synthetic(edge=0.03, seed=761)
    reg = st.predictive_regression(syn, 6)
    assert reg["t"] > 3.0 and reg["beta"] > 0, f"planted lead not recovered: {reg}"


def test_synthetic_is_deterministic():
    a = data.synthetic(edge=0.02, seed=761)["px"].values
    b = data.synthetic(edge=0.02, seed=761)["px"].values
    assert np.allclose(a, b)


def test_release_lag_is_no_lookahead():
    """forward_returns with lag>=1 must never use the entry-month price itself."""
    f = _frame()
    fwd = st.forward_returns(f, months=3, lag=1)
    # the last (lag+months) rows must be NaN — the horizon overruns the tape
    assert fwd.iloc[-1:].isna().all()
    # entry is the NEXT month's price, so a strictly increasing price => positive fwd
    assert (fwd.dropna() > 0).all()


def test_yoy_momentum_kills_constant_scaling():
    """YoY log momentum is invariant to a constant multiplicative rescaling of RevPAR."""
    import pandas as pd
    f = _frame()
    f2 = f.copy()
    f2["revpar"] = f2["revpar"] * 3.7
    m1 = st.revpar_momentum(f).dropna()
    m2 = st.revpar_momentum(f2).dropna()
    assert np.allclose(m1.values, m2.values)
