"""Strategy invariants: NCAV screen, portfolio construction, and engine spine."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graham_ncav import data, strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# ncav_screen invariants
# ---------------------------------------------------------------------------

def test_screen_shape(has_premium):
    ncav, _, _ = has_premium
    mask = st.ncav_screen(ncav)
    assert mask.shape == ncav.shape


def test_screen_returns_boolean(has_premium):
    ncav, _, _ = has_premium
    mask = st.ncav_screen(ncav)
    assert mask.dtypes.unique().tolist() == [bool] or all(mask.dtypes == bool)


def test_screen_threshold_logic():
    """Values above threshold pass; below fail."""
    idx = pd.Index([2020], name="year")
    cols = ["A", "B", "C"]
    ncav = pd.DataFrame([[0.5, 1.0, 1.5]], index=idx, columns=cols)
    mask = st.ncav_screen(ncav, threshold=1.0)
    assert not mask.iloc[0, 0]  # 0.5 < 1.0
    assert mask.iloc[0, 1]      # 1.0 >= 1.0
    assert mask.iloc[0, 2]      # 1.5 >= 1.0


def test_screen_graham_two_thirds_stricter():
    """threshold=1.5 (Graham's 2/3 rule) is stricter than threshold=1.0."""
    idx = pd.Index([2020], name="year")
    cols = ["A", "B"]
    ncav = pd.DataFrame([[1.1, 1.6]], index=idx, columns=cols)
    mask_loose = st.ncav_screen(ncav, threshold=1.0)
    mask_strict = st.ncav_screen(ncav, threshold=1.5)
    # A (1.1) passes loose but not strict; B (1.6) passes both
    assert mask_loose.iloc[0, 0] and not mask_strict.iloc[0, 0]
    assert mask_loose.iloc[0, 1] and mask_strict.iloc[0, 1]


# ---------------------------------------------------------------------------
# net_net_portfolio invariants
# ---------------------------------------------------------------------------

def test_portfolio_shape_and_columns(has_premium):
    ncav, fwd, _ = has_premium
    res = st.net_net_portfolio(ncav, fwd)
    for col in ("n_netnets", "n_market", "ret_netnets", "ret_market", "hedge"):
        assert col in res.columns


def test_portfolio_hedge_equals_nn_minus_mkt(has_premium):
    ncav, fwd, _ = has_premium
    res = st.net_net_portfolio(ncav, fwd)
    valid = res["ret_netnets"].notna() & res["ret_market"].notna()
    assert np.allclose(
        res.loc[valid, "hedge"],
        res.loc[valid, "ret_netnets"] - res.loc[valid, "ret_market"],
    )


def test_portfolio_nan_when_no_netnets():
    """Years with zero net-nets (ncav_ratio all < threshold) give NaN returns."""
    idx = pd.Index([2020, 2021], name="year")
    cols = ["A", "B", "C"]
    # All ratios negative — no net-nets in any year
    ncav = pd.DataFrame([[-1.0, -0.5, -2.0], [-1.5, -0.3, -0.8]], index=idx, columns=cols)
    fwd = pd.DataFrame([[0.1, 0.2, 0.3], [0.05, 0.1, 0.15]], index=idx, columns=cols)
    res = st.net_net_portfolio(ncav, fwd)
    assert res["ret_netnets"].isna().all()
    assert res["hedge"].isna().all()


def test_portfolio_counts_correct():
    """n_netnets and n_market are correctly tallied."""
    idx = pd.Index([2020], name="year")
    cols = ["A", "B", "C", "D"]
    ncav = pd.DataFrame([[1.5, 0.5, -0.5, 1.2]], index=idx, columns=cols)
    fwd = pd.DataFrame([[0.1, 0.2, 0.3, 0.4]], index=idx, columns=cols)
    res = st.net_net_portfolio(ncav, fwd)
    assert res.loc[2020, "n_netnets"] == 2  # A (1.5) and D (1.2)
    assert res.loc[2020, "n_market"] == 4


def test_portfolio_positive_when_premium_planted(has_premium):
    """When ncav_premium > 0 (high NCAV -> high return), net-net portfolios outperform."""
    ncav, fwd, _ = has_premium
    res = st.net_net_portfolio(ncav, fwd)
    # The premium is planted; on average net-nets should beat market
    hedge_valid = res["hedge"].dropna()
    assert hedge_valid.mean() > 0.0


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


def test_summarize_ignores_nan():
    """NaN values in the input are dropped before computation."""
    r = pd.Series([0.1, float("nan"), 0.3])
    s = st.summarize(r)
    assert s["n"] == 2
    assert abs(s["mean"] - 0.2) < 1e-9


# ---------------------------------------------------------------------------
# Engine spine: recovers premium only when planted
# ---------------------------------------------------------------------------

def test_engine_recovers_planted_premium():
    """The machinery returns a positive hedge mean when ncav_premium is real."""
    ncav, fwd, _ = data.synthetic_panel(
        n_firms=300, n_years=25, ncav_premium=0.12, seed=77
    )
    res = st.net_net_portfolio(ncav, fwd)
    hedge = res["hedge"].dropna()
    assert len(hedge) > 0
    assert hedge.mean() > 0.0


def test_engine_no_spurious_premium_under_null():
    """Under the null (ncav_premium=0) the hedge mean is not robustly large."""
    ncav, fwd, _ = data.synthetic_panel(
        n_firms=300, n_years=25, ncav_premium=0.0, seed=77
    )
    res = st.net_net_portfolio(ncav, fwd)
    hedge = res["hedge"].dropna()
    s = st.summarize(hedge)
    assert abs(s["tstat"]) < 3.0  # no strong spurious signal
