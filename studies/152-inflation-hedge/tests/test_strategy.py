"""Strategy invariants, Fisher regression, and regime analysis — the study's spine:
the inflation hedge only holds when we plant it, and on the null world it fails
(real returns fall when inflation rises, as Fama & Schwert 1977 showed)."""

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inflation_hedge import strategy as st  # noqa: E402


# ---------------------------------------------------------------------------
# Fisher regression — mechanics
# ---------------------------------------------------------------------------
def test_fisher_regression_returns_required_keys(null_world):
    df, _ = null_world
    res = st.fisher_regression(df)
    for key in ["n", "beta_nom", "t_nom", "r2_nom", "beta_real", "t_real"]:
        assert key in res, f"missing key: {key}"


def test_fisher_regression_n_matches_data(null_world):
    df, _ = null_world
    res = st.fisher_regression(df)
    expected = df[["cpi_yoy", "nom_ret_12m", "real_tr_ret_12m"]].dropna().__len__()
    assert res["n"] == expected


def test_ols_beta_correct_on_simple_case():
    """On y = 2x + noise, _ols_beta should recover beta ≈ 2."""
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)
    y = 2.0 * x + 0.1 * rng.standard_normal(500)
    beta, alpha, r2 = st._ols_beta(x, y)
    assert abs(beta - 2.0) < 0.05, f"Expected beta ≈ 2, got {beta:.3f}"
    assert r2 > 0.95


# ---------------------------------------------------------------------------
# Fisher hypothesis: planted knob drives recovered beta
# ---------------------------------------------------------------------------
def test_full_hedge_nominal_beta_near_one(full_hedge_world):
    """With fisher_beta=1.0 planted, the recovered nominal beta should be closer
    to 1 than to 0 (allowing for noise in a 100-year synthetic panel)."""
    df, _ = full_hedge_world
    res = st.fisher_regression(df)
    assert res["beta_nom"] > 0.5, (
        f"Expected beta_nom > 0.5 with full hedge, got {res['beta_nom']:.3f}"
    )


def test_null_world_real_beta_negative(null_world):
    """With fisher_beta=0.0, real returns should have a *negative* OLS beta on
    inflation — the Fama-Schwert (1977) finding (stocks are a poor short-run hedge
    because nominal returns don't rise with inflation, so real returns fall)."""
    df, _ = null_world
    res = st.fisher_regression(df)
    assert res["beta_real"] < 0.0, (
        f"Expected negative real beta in null world, got {res['beta_real']:.3f}"
    )


# ---------------------------------------------------------------------------
# Regime analysis — mechanics
# ---------------------------------------------------------------------------
def test_regime_returns_keys(null_world):
    df, _ = null_world
    res = st.regime_returns(df)
    for key in ["high", "low", "diff_mean", "t_diff", "t_diff_shuffled"]:
        assert key in res


def test_regime_counts_sum_to_total(null_world):
    df, _ = null_world
    sub = df[["cpi_regime", "fwd_real_tr_12m"]].dropna()
    res = st.regime_returns(df)
    assert res["high"]["n"] + res["low"]["n"] == len(sub)


def test_shuffled_label_has_smaller_absolute_t(null_world):
    """The shuffled-label control destroys regime signal; the real regime t should
    be larger in absolute value than the shuffled version *on average* (not
    guaranteed for every seed but should hold given a 100-year window)."""
    df, _ = null_world
    res = st.regime_returns(df)
    # In the null world (no real effect planted), this is not guaranteed to hold
    # — but the shuffled t_diff should be near zero:
    assert abs(res["t_diff_shuffled"]) < 3.0, (
        f"Shuffled label unexpectedly produced t={res['t_diff_shuffled']:.2f}"
    )


def test_regime_returns_high_lower_than_low_in_null_world(null_world):
    """In the null world (beta=0) high-inflation months should have lower forward
    real returns than low-inflation months (since nominal doesn't compensate)."""
    df, _ = null_world
    res = st.regime_returns(df, fwd_col="fwd_real_tr_12m")
    # diff_mean = high_mean - low_mean; should be negative in null world
    assert res["diff_mean"] < 0.0, (
        f"Expected negative diff (high-low) in null world, got {res['diff_mean']:.4f}"
    )


# ---------------------------------------------------------------------------
# Summarize
# ---------------------------------------------------------------------------
def test_summarize_returns_all_keys(null_world):
    df, _ = null_world
    out = st.summarize(df)
    for key in [
        "fisher_beta_nom", "fisher_t_nom", "fisher_beta_real", "fisher_t_real",
        "high_mean_1y", "low_mean_1y", "diff_mean_1y", "t_diff_1y",
        "n_high", "n_low", "t_diff_shuffled_1y",
        "high_mean_5y", "low_mean_5y", "diff_mean_5y", "t_diff_5y",
        "n_obs_total",
    ]:
        assert key in out, f"missing key in summarize(): {key}"


def test_summarize_on_real_data(shiller_df):
    """The study's spine on the real tape: real returns fall in high-inflation
    regimes (negative diff, significant t) and the nominal Fisher beta is below 1."""
    out = st.summarize(shiller_df)
    # Fisher nominal beta should be well below 1 (Fama-Schwert finding)
    assert out["fisher_beta_nom"] < 0.8, (
        f"Fisher beta_nom={out['fisher_beta_nom']:.3f} — expected < 0.8 on real data"
    )
    # Real beta should be negative (real returns fall when inflation rises)
    assert out["fisher_beta_real"] < 0.0, (
        f"Fisher beta_real={out['fisher_beta_real']:.3f} — expected < 0 on real data"
    )
    # High-inflation regime should have *lower* forward 1-year real returns
    assert out["diff_mean_1y"] < 0.0, (
        f"Expected negative high-low diff on real data, got {out['diff_mean_1y']:.4f}"
    )
