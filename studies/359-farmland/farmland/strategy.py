"""Strategy + inference for Study 359 — Farmland.

The claim has three testable halves:

* **Signal — is farmland a real low-beta inflation hedge / diversifier?** We regress each
  tradable proxy's excess return on the market (CAPM beta, with HAC-robust t) and test whether
  its inflation-beta is positive and significant. A *real* diversifier has a low, stable beta;
  a *real* inflation hedge has a positive, significant CPI loading. The verdict turns on a robust
  ``t >= 2`` on the REAL tape — literature support alone is only WEAK.

* **The smoothness myth — appraisal smoothing.** The famous NCREIF numbers are appraisal-based:
  the reported series is a moving average of true returns, which mechanically *shrinks* measured
  volatility and beta and *manufactures* autocorrelation. We measure the autocorrelation, then
  *un-smooth* (Geltner / Fisher-Geltner first-order reversal) and show how much of the "low risk,
  zero correlation" story is an artifact of the appraisal filter.

* **Tradability — can you actually hold it?** The only retail vehicles (LAND, FPI) are small,
  leveraged, illiquid REITs. We net one-way transaction costs off the proxy returns and report
  the Sharpe / inflation-hedge net of costs, and name the leverage + illiquidity caveats.

Pure numpy/pandas; offline; no scipy required (t via stdlib-style formulas; HAC-Newey-West NW
done by hand). Where scipy is available we use it only for the survival-function p-value.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

ANN = 12  # months per year for the monthly proxy panel


# --------------------------------------------------------------------------- #
# OLS with HAC (Newey-West) standard errors
# --------------------------------------------------------------------------- #
def ols_hac(y: np.ndarray, x: np.ndarray, lags: int = 6) -> dict:
    """OLS of ``y`` on ``[1, x]`` with Newey-West HAC standard errors.

    Returns alpha, beta, their HAC t-stats, n, and R². ``x`` is a 1-D regressor; the intercept
    is added. HAC with ``lags`` Bartlett-kernel lags guards against the autocorrelation/
    heteroskedasticity that appraisal smoothing and overlapping returns inject.
    """
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    k = X.shape[1]
    XtX_inv = np.linalg.inv(X.T @ X)
    beta_hat = XtX_inv @ (X.T @ y)
    resid = y - X @ beta_hat
    # Newey-West HAC meat
    S = (X * resid[:, None]).T @ (X * resid[:, None])
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        u = X * resid[:, None]
        G = u[L:].T @ u[:-L]
        S += w * (G + G.T)
    cov = XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(cov))
    t = beta_hat / se
    ss_tot = np.sum((y - y.mean()) ** 2)
    ss_res = np.sum(resid ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "alpha": float(beta_hat[0]), "beta": float(beta_hat[1]),
        "se_alpha": float(se[0]), "se_beta": float(se[1]),
        "t_alpha": float(t[0]), "t_beta": float(t[1]),
        "n": int(n), "r2": float(r2),
    }


def two_sided_p(t: float) -> float:
    """Two-sided p-value for a t-stat using a normal approximation (large-n, no scipy)."""
    return float(math.erfc(abs(t) / math.sqrt(2.0)))


# --------------------------------------------------------------------------- #
# Signal axis — market beta and inflation beta of the tradable proxies
# --------------------------------------------------------------------------- #
def market_beta(proxy_ret: pd.Series, mkt_ret: pd.Series, rf_monthly: float = 0.0,
                lags: int = 6) -> dict:
    """CAPM regression of a proxy's excess return on the market excess return (HAC t)."""
    df = pd.concat([proxy_ret, mkt_ret], axis=1).dropna()
    y = df.iloc[:, 0].values - rf_monthly
    x = df.iloc[:, 1].values - rf_monthly
    r = ols_hac(y, x, lags=lags)
    r["corr"] = float(np.corrcoef(y, x)[0, 1])
    r["ann_vol"] = float(df.iloc[:, 0].std() * math.sqrt(ANN))
    r["ann_mkt_vol"] = float(df.iloc[:, 1].std() * math.sqrt(ANN))
    return r


def inflation_beta(annual: pd.DataFrame, asset: str = "farmland",
                   lags: int = 2) -> dict:
    """Regress an annual real-return series on CPI to test the inflation-hedge claim.

    A *positive, significant* coefficient on CPI is the empirical content of "inflation hedge"
    (the asset's nominal return rises with inflation). ``annual`` must carry ``cpi`` and the
    ``asset`` column (fractions). HAC lags small for the short annual series.
    """
    df = annual[[asset, "cpi"]].dropna()
    r = ols_hac(df[asset].values, df["cpi"].values, lags=lags)
    r["mean_real"] = float((df[asset] - df["cpi"]).mean())
    r["corr_cpi"] = float(np.corrcoef(df[asset].values, df["cpi"].values)[0, 1])
    return r


# --------------------------------------------------------------------------- #
# The smoothness myth — appraisal smoothing diagnostics + un-smoothing
# --------------------------------------------------------------------------- #
def autocorr(x: np.ndarray, lag: int = 1) -> float:
    """Lag-``lag`` autocorrelation of a return series (appraisal smoothing -> high positive)."""
    x = np.asarray(x, float)
    x = x - x.mean()
    denom = np.sum(x * x)
    if denom == 0:
        return float("nan")
    return float(np.sum(x[lag:] * x[:-lag]) / denom)


def unsmooth(r_app: np.ndarray, rho: float | None = None) -> np.ndarray:
    """Geltner / Fisher-Geltner first-order un-smoothing.

    Given a reported series ``r_app`` with first-order smoothing
    ``r_app[t] = (1-rho)*r_true[t] + rho*r_app[t-1]``, recover the implied true series
    ``r_true[t] = (r_app[t] - rho*r_app[t-1]) / (1-rho)``. If ``rho`` is None we estimate it as
    the lag-1 autocorrelation of the reported series (the standard practical choice).
    Returns the un-smoothed series (length n-1; the first obs is dropped).
    """
    r_app = np.asarray(r_app, float)
    if rho is None:
        rho = autocorr(r_app, 1)
    rho = float(np.clip(rho, 0.0, 0.95))
    r_true = (r_app[1:] - rho * r_app[:-1]) / (1.0 - rho)
    return r_true


def smoothing_summary(r_app: np.ndarray) -> dict:
    """Volatility and autocorrelation of a reported series vs its un-smoothed version."""
    r_app = np.asarray(r_app, float)
    rho = autocorr(r_app, 1)
    r_true = unsmooth(r_app, rho)
    return {
        "rho1": float(rho),
        "vol_reported": float(np.std(r_app, ddof=1)),
        "vol_unsmoothed": float(np.std(r_true, ddof=1)),
        "vol_ratio": float(np.std(r_app, ddof=1) / np.std(r_true, ddof=1)),
        "n": int(len(r_app)),
    }


def expected_var_ratio(a: float) -> float:
    """Closed-form reported/true variance ratio for the smoothing filter
    ``r_app[t] = a*r_true[t] + (1-a)*r_app[t-1]`` with iid true returns: a/(2-a)."""
    return a / (2.0 - a)


def expected_rho1(a: float) -> float:
    """Closed-form lag-1 autocorrelation of the same filtered series: (1-a)."""
    return 1.0 - a


# --------------------------------------------------------------------------- #
# Tradability — costs on NAV, net Sharpe of the holdable proxies
# --------------------------------------------------------------------------- #
def net_sharpe(proxy_ret: pd.Series, cost_oneway: float = 0.0,
               turnover_per_year: float = 1.0, rf_annual: float = 0.0) -> dict:
    """Annualised Sharpe of a buy-and-hold proxy, net of one-way costs amortised over the year.

    ``cost_oneway`` is the round-trip-half cost as a fraction of NAV (bid/ask + commission);
    ``turnover_per_year`` is how many one-way trades per year the cost is charged on (a
    buy-and-hold investor pays ~one entry, so 1.0 is a generous low-turnover assumption).
    """
    r = proxy_ret.dropna()
    mu_m = r.mean()
    sd_m = r.std()
    cost_drag_m = cost_oneway * turnover_per_year / ANN
    mu_net_ann = (mu_m - cost_drag_m) * ANN - rf_annual
    sd_ann = sd_m * math.sqrt(ANN)
    return {
        "ann_return_gross": float(mu_m * ANN),
        "ann_return_net": float(mu_net_ann + rf_annual),
        "ann_vol": float(sd_ann),
        "sharpe_gross": float((mu_m * ANN - rf_annual) / sd_ann) if sd_ann > 0 else float("nan"),
        "sharpe_net": float(mu_net_ann / sd_ann) if sd_ann > 0 else float("nan"),
        "n": int(len(r)),
    }
