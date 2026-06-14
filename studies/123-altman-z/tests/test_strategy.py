"""Strategy invariants: Z-score formula, bucket sort, and the engine's spine."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from altman_z import data, strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# altman_zscore formula invariants
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
    re         = _df()
    ebit_val   = _df()
    liab       = _df()
    rev        = _df()
    mkt_eq     = _df() * 3
    return assets, assets_cur, liab_cur, re, ebit_val, liab, rev, mkt_eq


def test_zscore_formula_shape():
    A, AC, LC, RE, EB, LB, R, ME = _trivial_inputs()
    z = st.altman_zscore(A, AC, LC, RE, EB, LB, R, ME)
    assert z.shape == A.shape


def test_zscore_all_finite_on_valid_inputs():
    A, AC, LC, RE, EB, LB, R, ME = _trivial_inputs()
    z = st.altman_zscore(A, AC, LC, RE, EB, LB, R, ME)
    assert z.notna().all().all()


def test_zscore_nan_where_assets_zero():
    A, AC, LC, RE, EB, LB, R, ME = _trivial_inputs()
    A.iloc[0, 0] = 0.0
    z = st.altman_zscore(A, AC, LC, RE, EB, LB, R, ME)
    assert np.isnan(z.iloc[0, 0])


def test_zscore_nan_where_liabilities_zero():
    A, AC, LC, RE, EB, LB, R, ME = _trivial_inputs()
    LB.iloc[2, 3] = 0.0
    z = st.altman_zscore(A, AC, LC, RE, EB, LB, R, ME)
    assert np.isnan(z.iloc[2, 3])


def test_zscore_positive_for_healthy_firm():
    """A healthy firm with large assets, strong profits, low debt → high Z."""
    idx = pd.Index([2020], name="year")
    cols = ["HEALTHY"]
    _s = lambda v: pd.DataFrame([[v]], index=idx, columns=cols)
    z = st.altman_zscore(
        assets=_s(1e9), assets_cur=_s(4e8), liab_cur=_s(1e8),
        retained_earnings=_s(5e8), ebit=_s(2e8), liabilities=_s(2e8),
        revenues=_s(9e8), mkt_equity=_s(5e9),
    )
    assert z.iloc[0, 0] > 3.0


def test_zscore_low_for_distressed_firm():
    """A distressed firm with negative equity and high leverage → low Z."""
    idx = pd.Index([2020], name="year")
    cols = ["DISTRESSED"]
    _s = lambda v: pd.DataFrame([[v]], index=idx, columns=cols)
    z = st.altman_zscore(
        assets=_s(1e9), assets_cur=_s(1.5e8), liab_cur=_s(2e8),
        retained_earnings=_s(-3e8), ebit=_s(-5e7), liabilities=_s(9e8),
        revenues=_s(3e8), mkt_equity=_s(5e7),
    )
    assert z.iloc[0, 0] < 1.81


# ---------------------------------------------------------------------------
# tertile_hedge
# ---------------------------------------------------------------------------

def test_hedge_shape_and_columns(has_premium):
    z, fwd, _ = has_premium
    res = st.tertile_hedge(z, fwd)
    for col in ("n", "n_lo", "n_hi", "ret_lo", "ret_mid", "ret_hi", "ret_mkt", "hedge"):
        assert col in res.columns


def test_hedge_n_covers_expected_years(has_premium):
    z, fwd, truth = has_premium
    res = st.tertile_hedge(z, fwd)
    # We expect most years (min_firms=20, n_firms=200)
    assert len(res) >= truth["n_years"] - 2


def test_hedge_equals_hi_minus_lo(has_premium):
    z, fwd, _ = has_premium
    res = st.tertile_hedge(z, fwd)
    assert np.allclose(res["hedge"], res["ret_hi"] - res["ret_lo"])


def test_hedge_positive_when_premium_planted(has_premium):
    """When z_premium > 0 (high Z → high return), the hedge should be positive on average."""
    z, fwd, _ = has_premium
    res = st.tertile_hedge(z, fwd)
    assert res["hedge"].mean() > 0.0


def test_hedge_near_zero_under_null(no_premium):
    """Under the null (z_premium = 0) the hedge has no systematic sign."""
    z, fwd, _ = no_premium
    res = st.tertile_hedge(z, fwd)
    # The mean hedge is small relative to vol — not necessarily zero but << planted premium
    h = res["hedge"]
    assert abs(h.mean()) < h.std() + 0.05


def test_hedge_skips_sparse_years():
    """Years with fewer than min_firms valid observations are excluded."""
    rng = np.random.default_rng(0)
    idx = pd.Index([2010, 2011, 2012], name="year")
    cols = [f"T{i}" for i in range(50)]
    z = pd.DataFrame(rng.uniform(0, 5, (3, 50)), index=idx, columns=cols)
    fwd = pd.DataFrame(rng.normal(0.1, 0.3, (3, 50)), index=idx, columns=cols)
    # Make 2010 sparse by NaN-ing out all but 5 firms
    z.iloc[0, 10:] = np.nan
    res = st.tertile_hedge(z, fwd, min_firms=20)
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
    """The machinery returns a positive hedge mean when z_premium is real."""
    z, fwd, _ = data.synthetic_panel(n_firms=300, n_years=25, z_premium=0.10, seed=77)
    res = st.tertile_hedge(z, fwd)
    assert res["hedge"].mean() > 0.01  # planted premium is findable


def test_engine_no_spurious_premium_under_null():
    """Under the null (z_premium=0) the hedge mean is not robustly large."""
    z, fwd, _ = data.synthetic_panel(n_firms=300, n_years=25, z_premium=0.0, seed=77)
    res = st.tertile_hedge(z, fwd)
    s = st.summarize(res["hedge"])
    assert abs(s["tstat"]) < 3.0  # no strong spurious signal
