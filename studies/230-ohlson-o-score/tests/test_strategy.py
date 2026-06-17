"""Strategy invariants: O-score formula, bucket sort, and the engine's spine."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ohlson_o_score import data, strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# ohlson_oscore formula invariants
# ---------------------------------------------------------------------------

def _trivial_inputs(n=30, seed=0):
    """Simple square DataFrames for unit-testing the formula."""
    rng = np.random.default_rng(seed)
    idx = pd.Index(range(2010, 2010 + n), name="year")
    cols = [f"T{i}" for i in range(10)]
    _df = lambda: pd.DataFrame(rng.uniform(1e6, 1e9, (n, 10)), index=idx, columns=cols)
    assets     = _df() * 2
    assets_cur = _df()
    liab_cur   = _df() * 0.5
    liab       = _df()
    ni         = _df() * 0.1  # positive net income
    ni_prev    = _df() * 0.1
    cfo        = _df() * 0.15
    return assets, assets_cur, liab_cur, liab, ni, ni_prev, cfo


def test_oscore_formula_shape():
    A, AC, LC, LB, NI, NIP, CFO = _trivial_inputs()
    o = st.ohlson_oscore(A, AC, LC, LB, NI, NIP, CFO)
    assert o.shape == A.shape


def test_oscore_all_finite_on_valid_inputs():
    A, AC, LC, LB, NI, NIP, CFO = _trivial_inputs()
    o = st.ohlson_oscore(A, AC, LC, LB, NI, NIP, CFO)
    assert o.notna().all().all()


def test_oscore_nan_where_assets_zero():
    A, AC, LC, LB, NI, NIP, CFO = _trivial_inputs()
    A.iloc[0, 0] = 0.0
    o = st.ohlson_oscore(A, AC, LC, LB, NI, NIP, CFO)
    assert np.isnan(o.iloc[0, 0])


def test_oscore_nan_where_liabilities_zero():
    A, AC, LC, LB, NI, NIP, CFO = _trivial_inputs()
    LB.iloc[2, 3] = 0.0
    o = st.ohlson_oscore(A, AC, LC, LB, NI, NIP, CFO)
    assert np.isnan(o.iloc[2, 3])


def test_oscore_low_for_healthy_firm():
    """A healthy firm with large assets, positive income, strong CFO -> low O-score (safe)."""
    idx = pd.Index([2020], name="year")
    cols = ["HEALTHY"]
    _s = lambda v: pd.DataFrame([[v]], index=idx, columns=cols)
    o = st.ohlson_oscore(
        assets=_s(1e9), assets_cur=_s(4e8), liab_cur=_s(1e8),
        liabilities=_s(2e8), net_income=_s(2e8), net_income_prev=_s(1.8e8),
        cfo=_s(2.5e8),
    )
    # Healthy firm should have a low (negative) O-score
    assert o.iloc[0, 0] < 0.0


def test_oscore_high_for_distressed_firm():
    """A distressed firm with losses, high leverage, negative CFO -> high O-score."""
    idx = pd.Index([2020], name="year")
    cols = ["DISTRESSED"]
    _s = lambda v: pd.DataFrame([[v]], index=idx, columns=cols)
    o = st.ohlson_oscore(
        assets=_s(1e9), assets_cur=_s(1.5e8), liab_cur=_s(3e8),
        liabilities=_s(9.5e8), net_income=_s(-1e8), net_income_prev=_s(-5e7),
        cfo=_s(-8e7),
    )
    # Distressed firm should have a higher O-score than a healthy firm
    # (which typically scores below -10 for large, profitable companies).
    # A distressed firm may still be negative but much closer to 0.
    assert o.iloc[0, 0] > -4.0


# ---------------------------------------------------------------------------
# tertile_hedge
# ---------------------------------------------------------------------------

def test_hedge_shape_and_columns(has_premium):
    o, fwd, _ = has_premium
    res = st.tertile_hedge(o, fwd)
    for col in ("n", "n_lo", "n_hi", "ret_lo", "ret_mid", "ret_hi", "ret_mkt", "hedge"):
        assert col in res.columns


def test_hedge_n_covers_expected_years(has_premium):
    o, fwd, truth = has_premium
    res = st.tertile_hedge(o, fwd)
    # Expect most years (min_firms=20, n_firms=200)
    assert len(res) >= truth["n_years"] - 2


def test_hedge_equals_lo_minus_hi(has_premium):
    """Hedge = ret_lo - ret_hi (safe minus distressed)."""
    o, fwd, _ = has_premium
    res = st.tertile_hedge(o, fwd)
    assert np.allclose(res["hedge"], res["ret_lo"] - res["ret_hi"])


def test_hedge_negative_when_distress_premium_planted(has_premium):
    """With o_premium > 0 (high-O earns more), safe-minus-distressed hedge is negative."""
    o, fwd, _ = has_premium
    res = st.tertile_hedge(o, fwd)
    # o_premium > 0 means high-O earns more => ret_hi > ret_lo => hedge < 0
    assert res["hedge"].mean() < 0.0


def test_hedge_near_zero_under_null(no_premium):
    """Under the null (o_premium=0) the hedge has no systematic sign."""
    o, fwd, _ = no_premium
    res = st.tertile_hedge(o, fwd)
    h = res["hedge"]
    assert abs(h.mean()) < h.std() + 0.05


def test_hedge_skips_sparse_years():
    """Years with fewer than min_firms valid observations are excluded."""
    rng = np.random.default_rng(0)
    idx = pd.Index([2010, 2011, 2012], name="year")
    cols = [f"T{i}" for i in range(50)]
    o = pd.DataFrame(rng.normal(-1, 2, (3, 50)), index=idx, columns=cols)
    fwd = pd.DataFrame(rng.normal(0.1, 0.3, (3, 50)), index=idx, columns=cols)
    # Make 2010 sparse by NaN-ing out all but 5 firms
    o.iloc[0, 10:] = np.nan
    res = st.tertile_hedge(o, fwd, min_firms=20)
    assert 2010 not in res.index
    assert 2011 in res.index


# ---------------------------------------------------------------------------
# summarize HAC
# ---------------------------------------------------------------------------

def test_summarize_keys():
    r = pd.Series([0.02, -0.01, 0.05, 0.03, -0.02, 0.04, 0.01, 0.06])
    s = st.summarize(r)
    assert set(s.keys()) >= {"mean", "vol", "sharpe", "tstat", "hit_rate", "n"}


def test_summarize_mean_correct():
    r = pd.Series([0.1, 0.2, 0.3])
    s = st.summarize(r)
    assert abs(s["mean"] - 0.2) < 1e-9


def test_summarize_short_series():
    """A series of length 1 returns nan for tstat/vol/sharpe."""
    s = st.summarize(pd.Series([0.1]))
    assert np.isnan(s["tstat"])


# ---------------------------------------------------------------------------
# Engine spine: recovers premium only when planted
# ---------------------------------------------------------------------------

def test_engine_recovers_planted_premium():
    """With a large o_premium the engine detects the planted distress-premium direction."""
    # o_premium > 0 means high-O earns more; hedge (lo - hi) should be negative
    o, fwd, _ = data.synthetic_panel(n_firms=300, n_years=25, o_premium=0.10, seed=77)
    res = st.tertile_hedge(o, fwd)
    assert res["hedge"].mean() < -0.01  # distressed earns more than safe

def test_engine_no_spurious_premium_under_null():
    """Under the null (o_premium=0) the hedge t-stat is not robustly large."""
    o, fwd, _ = data.synthetic_panel(n_firms=300, n_years=25, o_premium=0.0, seed=77)
    res = st.tertile_hedge(o, fwd)
    s = st.summarize(res["hedge"])
    assert abs(s["tstat"]) < 3.0  # no strong spurious signal
