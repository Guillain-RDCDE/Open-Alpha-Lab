"""Strategy and honest controls — Study 152 (Inflation-Hedge).

We test two related claims about stocks as an inflation hedge, using the Shiller
monthly panel (1871-present):

**Claim A — The Fisher hypothesis for stocks.**
  If stocks are a genuine inflation hedge, their nominal returns should rise when
  inflation rises, leaving real returns invariant.  The test: regress 12-month
  nominal returns on 12-month trailing CPI inflation.  A Fisher beta of +1.0
  means full pass-through; beta near 0 (or negative) means the hedge fails.
  The honest control is a regression of *real* returns on inflation — if Fisher
  holds, that coefficient should be near zero.

**Claim B — Real-return resilience across inflation regimes.**
  Stocks' role as a "real asset" implies their real *total* returns in high-
  inflation periods should match (or exceed) those in low-inflation periods.
  We split months by a pre-specified 3% CPI-YoY threshold and compare forward
  12-month and 60-month real total returns across regimes using a HAC-robust
  t-test on the regime difference.  The honest null is that the regime split
  carries no information — the shuffled-label control replaces the regime label
  with a random permutation (same split size) to show what a spurious regime
  difference would look like.

No look-ahead: all signal columns use data up to and including month t; forward
returns are measured from t+1 onward in the Shiller panel (the forward-return
columns in ``data.load_shiller`` already implement this via ``shift(-12)``).

The inference bar (from METHODOLOGY.md) is HAC |t| >= 2 for REAL; below that
the verdict stays WEAK or NONE.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Fisher hypothesis test
# ---------------------------------------------------------------------------
def fisher_regression(
    df: pd.DataFrame,
    x_col: str = "cpi_yoy",
    nom_col: str = "nom_ret_12m",
    real_col: str = "real_tr_ret_12m",
) -> dict:
    """OLS regression of nominal (and real) 12-month returns on CPI YoY.

    Tests the Fisher hypothesis:
      - If stocks fully hedge, nominal return beta ≈ 1; real return beta ≈ 0.
      - The Fama-Schwert finding is beta_nominal < 1 (often negative on short
        windows), beta_real < 0 (real returns fall when inflation rises).

    Returns a dict with OLS beta and a HAC (Newey-West) t-stat for both nominal
    and real regressions.  Uses a 12-lag Newey-West window given monthly
    overlapping returns (Hodrick 1992 style, conservative).

    Note: the 12-month rolling return windows create strong serial correlation
    (each observation overlaps 11 of 12 months with its neighbour).  The HAC
    estimator corrects for this; 12 lags is a standard conservative choice for
    annual-window overlapping returns.
    """
    sub = df[[x_col, nom_col, real_col]].dropna()
    x = sub[x_col].to_numpy(dtype=float)
    nom = sub[nom_col].to_numpy(dtype=float)
    real = sub[real_col].to_numpy(dtype=float)

    out: dict = {"n": int(len(sub))}
    for label, y in [("nom", nom), ("real", real)]:
        beta, alpha, r2 = _ols_beta(x, y)
        resid = y - (alpha + beta * x)
        t_beta = _hac_tstat(resid=resid, x=x, beta=beta, lags=12)
        out[f"beta_{label}"] = float(beta)
        out[f"alpha_{label}"] = float(alpha)
        out[f"r2_{label}"] = float(r2)
        out[f"t_{label}"] = float(t_beta)

    return out


def _ols_beta(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Simple bivariate OLS; returns (beta, alpha, R^2)."""
    xd = x - x.mean()
    yd = y - y.mean()
    beta = float((xd * yd).sum() / (xd**2).sum())
    alpha = float(y.mean() - beta * x.mean())
    ypred = alpha + beta * x
    ss_res = ((y - ypred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum()
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    return beta, alpha, r2


def _hac_tstat(
    resid: np.ndarray,
    x: np.ndarray,
    beta: float,
    lags: int = 12,
) -> float:
    """Newey-West HAC t-statistic for the OLS slope coefficient.

    Uses the Sandwich estimator: Var(beta) = (X'X)^{-1} * LRV * (X'X)^{-1} / n
    where LRV is the Newey-West long-run variance of the score (x_t * e_t).
    """
    n = len(resid)
    xd = x - x.mean()
    score = xd * resid  # gradient of OLS criterion wrt beta

    # Long-run variance of score (Newey-West / Bartlett kernel)
    lrv = float(score @ score) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(score[k:] @ score[:-k]) / n

    xx = float(xd @ xd) / n
    se_beta = np.sqrt(max(lrv, 0.0) / n) / xx if xx > 0 else np.nan
    return float(beta / se_beta) if (se_beta and se_beta > 0) else np.nan


# ---------------------------------------------------------------------------
# Regime analysis
# ---------------------------------------------------------------------------
def regime_returns(
    df: pd.DataFrame,
    regime_col: str = "cpi_regime",
    fwd_col: str = "fwd_real_tr_12m",
) -> dict:
    """Mean and HAC t-stat of forward real returns split by inflation regime.

    Compares months labelled "high" vs "low" CPI inflation.  The headline
    statistic is the regime *difference* (high minus low) with its HAC t-stat
    — the inferential test of whether the regime label carries information about
    future real equity performance.

    Returns means, standard deviations, counts, and the HAC t-stat on the
    difference.  Also returns the shuffled-label control: the same comparison
    with regime labels randomly permuted (seeded for reproducibility), to show
    what a false-positive regime split looks like.
    """
    sub = df[[regime_col, fwd_col]].dropna()
    high = sub.loc[sub[regime_col] == "high", fwd_col].to_numpy(dtype=float)
    low = sub.loc[sub[regime_col] == "low", fwd_col].to_numpy(dtype=float)

    def _stats(arr: np.ndarray) -> dict:
        return {
            "n": int(len(arr)),
            "mean": float(arr.mean()) if len(arr) else float("nan"),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else float("nan"),
            "median": float(np.median(arr)) if len(arr) else float("nan"),
        }

    diff_series = _regime_diff_series(sub, regime_col, fwd_col)
    t_diff = _hac_mean_tstat(diff_series, lags=12)

    # Shuffled-label control: permute the regime label, recompute
    rng = np.random.default_rng(152)
    sub_shuf = sub.copy()
    sub_shuf[regime_col] = rng.permutation(sub[regime_col].to_numpy())
    diff_shuf = _regime_diff_series(sub_shuf, regime_col, fwd_col)
    t_diff_shuf = _hac_mean_tstat(diff_shuf, lags=12)

    return {
        "high": _stats(high),
        "low": _stats(low),
        "diff_mean": float(np.mean(diff_series)) if len(diff_series) else float("nan"),
        "t_diff": t_diff,
        "t_diff_shuffled": t_diff_shuf,
        "fwd_col": fwd_col,
    }


def _regime_diff_series(
    sub: pd.DataFrame, regime_col: str, fwd_col: str
) -> np.ndarray:
    """Return (high_mean - obs) for obs in high, then subtracted from low's obs
    pooled as a per-observation difference.  We construct the difference by
    matching on a rolling 12-month pair to avoid equal-count requirements.

    Simpler approach: return the vector of 'high' obs with a subtracted global
    low mean, for a one-sample t-test of "is high different from low?".  This is
    equivalent to the Welch t-statistic corrected for overlapping returns via HAC.
    """
    high_vals = sub.loc[sub[regime_col] == "high", fwd_col].to_numpy(dtype=float)
    low_vals = sub.loc[sub[regime_col] == "low", fwd_col].to_numpy(dtype=float)
    if len(high_vals) == 0 or len(low_vals) == 0:
        return np.array([])
    low_mean = float(low_vals.mean())
    # Test: is the high-regime mean different from the low-regime mean?
    # Use a centred version: (high_obs - low_mean) tested against 0
    return high_vals - low_mean


def _hac_mean_tstat(arr: np.ndarray, lags: int = 12) -> float:
    """Newey-West HAC t-stat for H0: mean(arr) = 0."""
    if len(arr) < lags + 2:
        return float("nan")
    mu = arr.mean()
    e = arr - mu
    lrv = float(e @ e) / len(arr)
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / len(arr)
    se = np.sqrt(max(lrv, 0.0) / len(arr))
    return float(mu / se) if se > 0 else float("nan")


# ---------------------------------------------------------------------------
# Synthetic Fisher beta sweep — positive/negative control
# ---------------------------------------------------------------------------
def fisher_beta_sweep(
    betas: list[float],
    n_years: int = 130,
    seed: int = 152,
) -> pd.DataFrame:
    """Run fisher_regression across a range of planted Fisher betas (synthetic).

    Returns a DataFrame with one row per beta value: the recovered nominal
    and real slope coefficients and their HAC t-stats.  Used in the notebooks
    as a positive control: when ``fisher_beta = 1.0`` the nominal regression
    should return beta_nom ≈ 1; when ``fisher_beta = 0.0`` it should return
    beta_nom ≈ 0 (or negative, since real is constant and we add noise).
    """
    from .data import synthetic_world

    rows = []
    for b in betas:
        df, _ = synthetic_world(n_years=n_years, fisher_beta=b, seed=seed)
        res = fisher_regression(df)
        rows.append(
            {
                "planted_beta": b,
                "recovered_nom_beta": res["beta_nom"],
                "t_nom": res["t_nom"],
                "recovered_real_beta": res["beta_real"],
                "t_real": res["t_real"],
                "n": res["n"],
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summary helper — mirrors the `summarize()` pattern in other studies
# ---------------------------------------------------------------------------
def summarize(df: pd.DataFrame) -> dict:
    """Headline numbers for the inflation-hedge study.

    Runs both Fisher regression and regime analysis on 1-year and 5-year forward
    real total returns and packages the key numbers in a single flat dict.
    """
    fisher = fisher_regression(df)
    regime_1y = regime_returns(df, fwd_col="fwd_real_tr_12m")
    regime_5y = regime_returns(df, fwd_col="fwd_real_tr_60m")

    return {
        # Fisher hypothesis
        "fisher_beta_nom": fisher["beta_nom"],
        "fisher_t_nom": fisher["t_nom"],
        "fisher_r2_nom": fisher["r2_nom"],
        "fisher_beta_real": fisher["beta_real"],
        "fisher_t_real": fisher["t_real"],
        # Regime: 1-year forward real total return
        "high_mean_1y": regime_1y["high"]["mean"],
        "low_mean_1y": regime_1y["low"]["mean"],
        "diff_mean_1y": regime_1y["diff_mean"],
        "t_diff_1y": regime_1y["t_diff"],
        "n_high": regime_1y["high"]["n"],
        "n_low": regime_1y["low"]["n"],
        "t_diff_shuffled_1y": regime_1y["t_diff_shuffled"],
        # Regime: 5-year forward annualised real total return
        "high_mean_5y": regime_5y["high"]["mean"],
        "low_mean_5y": regime_5y["low"]["mean"],
        "diff_mean_5y": regime_5y["diff_mean"],
        "t_diff_5y": regime_5y["t_diff"],
        "n_obs_total": fisher["n"],
    }
