"""Strategy invariants: M-score formula, bucket sort, and engine spine — Study 229."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from beneish_m_score import data, strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# mscore_from_components formula invariants
# ---------------------------------------------------------------------------

def _component_dfs(n: int = 30, seed: int = 0) -> tuple:
    """Return eight aligned DataFrames of realistic-ish component values."""
    rng = np.random.default_rng(seed)
    idx = pd.Index(range(2010, 2010 + n), name="year")
    cols = [f"T{i}" for i in range(10)]
    # Typical ranges from Beneish (1999): most indices cluster near 1.0
    def _df(lo, hi):
        return pd.DataFrame(rng.uniform(lo, hi, (n, 10)), index=idx, columns=cols)

    DSRI = _df(0.5, 2.0)
    GMI  = _df(0.5, 2.0)
    AQI  = _df(0.5, 2.0)
    SGI  = _df(0.8, 1.5)
    DEPI = _df(0.5, 2.0)
    SGAI = _df(0.5, 2.0)
    TATA = _df(-0.1, 0.1)
    LVGI = _df(0.5, 2.0)
    return DSRI, GMI, AQI, SGI, DEPI, SGAI, TATA, LVGI


def test_mscore_shape():
    comps = _component_dfs()
    m = st.mscore_from_components(*comps)
    DSRI = comps[0]
    assert m.shape == DSRI.shape


def test_mscore_all_finite_on_valid_inputs():
    comps = _component_dfs()
    m = st.mscore_from_components(*comps)
    assert m.notna().all().all()


def test_mscore_low_for_clean_firm():
    """A firm with stable ratios near 1 and low accruals → M well below threshold."""
    idx = pd.Index([2020], name="year")
    cols = ["CLEAN"]
    _s = lambda v: pd.DataFrame([[v]], index=idx, columns=cols)
    # All indices = 1 (no change) and TATA near zero
    m = st.mscore_from_components(
        DSRI=_s(1.0), GMI=_s(1.0), AQI=_s(1.0), SGI=_s(1.0),
        DEPI=_s(1.0), SGAI=_s(1.0), TATA=_s(0.0), LVGI=_s(1.0),
    )
    # M = -4.84 + sum(coefs) * 1 + 0 = -4.84 + 0.920+0.528+0.404+0.892+0.115-0.172-0.327 = about -2.48
    assert m.iloc[0, 0] < -1.78


def test_mscore_high_for_manipulator():
    """A firm with inflated DSRI, GMI>1, high accruals → M above threshold."""
    idx = pd.Index([2020], name="year")
    cols = ["MANIP"]
    _s = lambda v: pd.DataFrame([[v]], index=idx, columns=cols)
    m = st.mscore_from_components(
        DSRI=_s(2.5), GMI=_s(2.5), AQI=_s(2.5), SGI=_s(1.8),
        DEPI=_s(2.0), SGAI=_s(1.5), TATA=_s(0.15), LVGI=_s(1.5),
    )
    assert m.iloc[0, 0] > -1.78


def test_mscore_nan_propagates():
    """If any component is NaN, the M-score is NaN for that cell."""
    idx = pd.Index([2020], name="year")
    cols = ["A", "B"]
    _df = lambda: pd.DataFrame([[1.0, 1.0]], index=idx, columns=cols)
    comps = [_df() for _ in range(8)]
    comps[0].iloc[0, 0] = np.nan  # DSRI NaN for first firm
    m = st.mscore_from_components(*comps)
    assert np.isnan(m.iloc[0, 0])
    assert not np.isnan(m.iloc[0, 1])


# ---------------------------------------------------------------------------
# tertile_hedge
# ---------------------------------------------------------------------------

def test_hedge_shape_and_columns(has_premium):
    m, fwd, _ = has_premium
    res = st.tertile_hedge(m, fwd)
    for col in ("n", "n_lo", "n_hi", "ret_lo", "ret_mid", "ret_hi", "ret_mkt", "hedge"):
        assert col in res.columns


def test_hedge_n_covers_expected_years(has_premium):
    m, fwd, truth = has_premium
    res = st.tertile_hedge(m, fwd)
    assert len(res) >= truth["n_years"] - 2


def test_hedge_equals_lo_minus_hi(has_premium):
    """Hedge is long-low-M minus short-high-M (opposite of Altman convention)."""
    m, fwd, _ = has_premium
    res = st.tertile_hedge(m, fwd)
    assert np.allclose(res["hedge"], res["ret_lo"] - res["ret_hi"])


def test_hedge_positive_when_premium_planted(has_premium):
    """When m_premium < 0 (high-M earns less), low-M minus high-M should be positive."""
    m, fwd, _ = has_premium
    res = st.tertile_hedge(m, fwd)
    assert res["hedge"].mean() > 0.0


def test_hedge_near_zero_under_null(no_premium):
    """Under the null (m_premium=0) the hedge has no systematic sign."""
    m, fwd, _ = no_premium
    res = st.tertile_hedge(m, fwd)
    h = res["hedge"]
    assert abs(h.mean()) < h.std() + 0.05


def test_hedge_skips_sparse_years():
    """Years with fewer than min_firms valid observations are excluded."""
    rng = np.random.default_rng(0)
    idx = pd.Index([2010, 2011, 2012], name="year")
    cols = [f"T{i}" for i in range(50)]
    m = pd.DataFrame(rng.normal(-2.5, 1.5, (3, 50)), index=idx, columns=cols)
    fwd = pd.DataFrame(rng.normal(0.1, 0.3, (3, 50)), index=idx, columns=cols)
    # Make 2010 sparse
    m.iloc[0, 10:] = np.nan
    res = st.tertile_hedge(m, fwd, min_firms=20)
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
    """The machinery returns a positive hedge when m_premium < 0 is real."""
    m, fwd, _ = data.synthetic_panel(n_firms=300, n_years=25, m_premium=-0.10, seed=77)
    res = st.tertile_hedge(m, fwd)
    assert res["hedge"].mean() > 0.01


def test_engine_no_spurious_premium_under_null():
    """Under the null (m_premium=0) the hedge t-stat is not robustly large."""
    m, fwd, _ = data.synthetic_panel(n_firms=300, n_years=25, m_premium=0.0, seed=77)
    res = st.tertile_hedge(m, fwd)
    s = st.summarize(res["hedge"])
    assert abs(s["tstat"]) < 3.0
