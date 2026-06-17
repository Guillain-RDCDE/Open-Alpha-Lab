"""Tests for the strategy layer of Study 274 (Fine-Wine).

All tests are offline and deterministic — they use the synthetic generator and
the hardcoded Liv-ex 100 table only.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fine_wine import data, strategy as st  # noqa: E402


def _make_merged(corr: float = 0.0, smooth: float = 0.0,
                 seed: int = 274, n_years: int = 22) -> pd.DataFrame:
    """Build a synthetic merged frame shaped like fetch_sp500_annual()."""
    df, _ = data.synthetic_annual(n_years=n_years, corr=corr, smooth=smooth, seed=seed)
    df = df.rename(columns={"eq_return": "sp500_return"})
    df["both_up"] = (df["sp500_return"] > 0) & (df["wine_return"] > 0)
    return df[["year", "wine_return", "sp500_return", "both_up"]]


# ---------------------------------------------------------------------------
# De-smoothing
# ---------------------------------------------------------------------------
def test_estimate_ar1_detects_smoothing():
    df, _ = data.synthetic_annual(n_years=3000, corr=0.0, smooth=0.6, seed=274)
    rho = st.estimate_ar1(df["wine_return"].to_numpy())
    assert rho > 0.2


def test_estimate_ar1_near_zero_when_unsmoothed():
    df, _ = data.synthetic_annual(n_years=3000, corr=0.0, smooth=0.0, seed=274)
    rho = st.estimate_ar1(df["wine_return"].to_numpy())
    assert abs(rho) < 0.1


def test_desmooth_raises_volatility_when_smoothed():
    df, _ = data.synthetic_annual(n_years=3000, corr=0.0, smooth=0.5, seed=274)
    w = df["wine_return"].to_numpy()
    w_ds = st.desmooth(w)
    assert st.ann_vol(w_ds) > st.ann_vol(w)


def test_desmooth_recovers_hidden_correlation():
    """De-smoothing a smoothed, truly-correlated series raises measured correlation."""
    df, _ = data.synthetic_annual(n_years=4000, corr=0.6, smooth=0.6, seed=274)
    w = df["wine_return"].to_numpy()
    e = df["sp500_return"].to_numpy() if "sp500_return" in df else df["eq_return"].to_numpy()
    raw_c = np.corrcoef(w, e)[0, 1]
    ds_c = np.corrcoef(st.desmooth(w), e)[0, 1]
    assert ds_c > raw_c


def test_desmooth_identity_when_no_autocorr():
    rng = np.random.default_rng(0)
    w = rng.normal(0.05, 0.13, 5000)
    w_ds = st.desmooth(w)
    # With rho ~ 0, de-smoothing is approximately the identity.
    assert abs(st.ann_vol(w_ds) - st.ann_vol(w)) < 0.02


# ---------------------------------------------------------------------------
# Risk / return primitives
# ---------------------------------------------------------------------------
def test_cagr_matches_compound():
    r = np.array([0.10, -0.05, 0.20])
    expected = (1.10 * 0.95 * 1.20) ** (1 / 3) - 1
    assert abs(st.cagr(r) - expected) < 1e-12


def test_sharpe_sign():
    rng = np.random.default_rng(3)
    good = rng.normal(0.12, 0.10, 2000)
    assert st.sharpe(good, rf=0.02) > 0


def test_hac_tstat_zero_mean_is_small():
    rng = np.random.default_rng(5)
    x = rng.normal(0.0, 1.0, 3000)
    t, mu = st.hac_tstat(x, lags=2)
    assert abs(t) < 3.0


def test_hac_tstat_positive_mean_is_significant():
    rng = np.random.default_rng(5)
    x = rng.normal(0.5, 1.0, 3000)
    t, mu = st.hac_tstat(x, lags=2)
    assert t > 5.0
    assert mu > 0.3


# ---------------------------------------------------------------------------
# Correlation block
# ---------------------------------------------------------------------------
def test_correlation_block_keys():
    df = _make_merged(corr=0.3, smooth=0.3)
    cb = st.correlation_block(df)
    required = {"n", "raw_corr", "wine_ar1", "desmoothed_corr",
                "wine_vol_raw", "wine_vol_desmoothed", "eq_vol"}
    assert required.issubset(set(cb.keys()))


def test_correlation_block_desmoothed_ge_raw_under_smoothing():
    df = _make_merged(corr=0.5, smooth=0.6, n_years=2000)
    cb = st.correlation_block(df)
    assert cb["desmoothed_corr"] >= cb["raw_corr"]
    assert cb["wine_vol_desmoothed"] >= cb["wine_vol_raw"]


# ---------------------------------------------------------------------------
# Frontier
# ---------------------------------------------------------------------------
def test_frontier_endpoints():
    df = _make_merged(n_years=200)
    f = st.two_asset_frontier(df["wine_return"].values, df["sp500_return"].values)
    assert f["wine_weight"].iloc[0] == 0.0
    assert f["wine_weight"].iloc[-1] == 1.0
    # weight 0 = pure equity
    eq_sharpe = st.sharpe(df["sp500_return"].values)
    assert abs(f["sharpe"].iloc[0] - eq_sharpe) < 1e-9


# ---------------------------------------------------------------------------
# Diversification benefit — the spine of the study
# ---------------------------------------------------------------------------
def test_benefit_keys():
    df = _make_merged()
    db = st.diversification_benefit(df)
    for lens in ["gross_raw", "net_raw", "gross_desmoothed", "net_desmoothed"]:
        assert f"blend_sharpe_{lens}" in db
        assert f"sharpe_delta_{lens}" in db
    assert "benefit_hac_t_gross_raw" in db
    assert "benefit_hac_t_net_desmoothed" in db


def test_costs_lower_net_relative_to_gross():
    df = _make_merged(corr=0.0, smooth=0.0, n_years=500)
    db = st.diversification_benefit(df, wine_weight=0.3)
    assert db["blend_sharpe_net_raw"] <= db["blend_sharpe_gross_raw"] + 1e-9


def test_benefit_detected_when_uncorrelated_diversifier_planted():
    """A genuine zero-correlation sleeve with equal-ish return should help (gross)."""
    df, _ = data.synthetic_annual(n_years=1000, corr=0.0, smooth=0.0,
                                  wine_mu=0.08, eq_mu=0.08, seed=274)
    df = df.rename(columns={"eq_return": "sp500_return"})
    df["both_up"] = True
    db = st.diversification_benefit(df, wine_weight=0.3)
    # Equal returns + zero correlation -> blend Sharpe exceeds equity-only (gross).
    assert db["blend_sharpe_gross_raw"] > db["eq_only_sharpe"]
    assert db["benefit_hac_t_gross_raw"] > 0


def test_no_benefit_when_wine_dominated_and_correlated():
    """A lower-return, positively-correlated sleeve should NOT improve Sharpe."""
    df, _ = data.synthetic_annual(n_years=1000, corr=0.6, smooth=0.0,
                                  wine_mu=0.03, eq_mu=0.10, seed=274)
    df = df.rename(columns={"eq_return": "sp500_return"})
    df["both_up"] = True
    db = st.diversification_benefit(df, wine_weight=0.3)
    assert db["blend_sharpe_net_desmoothed"] < db["eq_only_sharpe"]


# ---------------------------------------------------------------------------
# summarize
# ---------------------------------------------------------------------------
def test_summarize_keys():
    df = _make_merged(n_years=22)
    s = st.summarize(df)
    required = {
        "n", "year_start", "year_end",
        "wine_cagr_pct", "eq_cagr_pct", "wine_vol_pct", "eq_vol_pct",
        "wine_sharpe", "eq_sharpe",
        "raw_corr", "wine_ar1", "desmoothed_corr",
        "eq_only_sharpe", "blend_sharpe_net_desmoothed",
        "sharpe_delta_net_desmoothed",
        "benefit_hac_t_gross_raw", "benefit_hac_t_net_desmoothed",
    }
    assert required.issubset(set(s.keys()))


def test_summarize_n_matches_input():
    df = _make_merged(n_years=18)
    s = st.summarize(df)
    assert s["n"] == 18
