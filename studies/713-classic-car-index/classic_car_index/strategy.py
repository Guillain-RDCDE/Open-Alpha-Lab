"""Strategy + inference for Study 713 — "classic cars are an asset class that beats equities".

The claim: buying collector cars (Ferrari 250 GTO, Porsche 911, Mercedes 300 SL and the
like) is a real *investment* that beats the stock market — the HAGI / Knight Frank /
Hagerty indices "outperform the S&P." We test the strongest tradable version of that:

1. **The collector-car index itself** (hardcoded, cited, approximate) vs the S&P on the
   same window — CAGR, vol, max drawdown, and a *t*-stat of the annual excess return.
   We run it against BOTH a price-only benchmark (``^GSPC``, the fair apples-to-apples
   race, since the car index is a *price* index) AND a total-return benchmark (``SPY``,
   the real equity outcome, since stocks *pay you to hold*). This is the "did the cars
   actually beat stocks?" question on the index believers quote.
2. **The tradable equity proxies** (Ferrari ``RACE``, Aston Martin ``AML.L``) vs the S&P —
   the only listed things a public investor can actually buy that are *about* the trade.
   Monthly returns, Newey-West *t* of the alpha vs the market, Sharpe, max drawdown.
3. **The carry/illiquidity haircut** — what owning a classic car really costs: the auction
   round-trip spread (buyer's premium + seller's commission), plus insurance, climate
   storage and maintenance while it sits. We charge it once on the NAV and show where the
   "return" goes.

Inference: an annual-excess *t*-stat for the index, a Newey-West (HAC) *t* of the monthly
proxy alpha vs the S&P, Sharpe ratios, and a max-drawdown decomposition. Pure numpy/pandas;
scipy only for the *t*-distribution p-value (optional, guarded).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

MONTHS = 12.0


# --------------------------------------------------------------------------- #
# Return / risk primitives
# --------------------------------------------------------------------------- #
def cagr(level: pd.Series) -> float:
    """Compound annual growth rate of a level series (index carries the dates)."""
    yrs = (level.index[-1] - level.index[0]).days / 365.25
    if yrs <= 0 or level.iloc[0] <= 0:
        return float("nan")
    return (level.iloc[-1] / level.iloc[0]) ** (1.0 / yrs) - 1.0


def max_drawdown(level: pd.Series) -> float:
    """Worst peak-to-trough drawdown (a negative fraction)."""
    roll_max = level.cummax()
    dd = level / roll_max - 1.0
    return float(dd.min())


def ann_vol(returns: pd.Series, periods_per_year: float = MONTHS) -> float:
    """Annualised volatility of periodic simple returns."""
    return float(returns.std(ddof=1) * math.sqrt(periods_per_year))


def sharpe(returns: pd.Series, rf_annual: float = 0.0,
           periods_per_year: float = MONTHS) -> float:
    """Annualised Sharpe of periodic simple returns (excess of a flat rf)."""
    rf_p = (1 + rf_annual) ** (1 / periods_per_year) - 1
    ex = returns - rf_p
    sd = ex.std(ddof=1)
    if sd == 0:
        return float("nan")
    return float(ex.mean() / sd * math.sqrt(periods_per_year))


def summarize(level: pd.Series, periods_per_year: float = MONTHS) -> dict:
    """CAGR / annualised vol / Sharpe / max-drawdown of a level series."""
    rets = level.pct_change().dropna()
    return {
        "cagr": cagr(level),
        "vol": ann_vol(rets, periods_per_year),
        "sharpe": sharpe(rets, 0.0, periods_per_year),
        "mdd": max_drawdown(level),
        "n": int(len(rets)),
    }


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def _t_p_value(t: float, df: int) -> float:
    """Two-sided p-value for a t-stat (scipy if available, else a normal approx)."""
    if not np.isfinite(t):
        return float("nan")
    try:
        from scipy import stats
        return float(2 * stats.t.sf(abs(t), df))
    except Exception:  # pragma: no cover - scipy is in requirements but stay safe
        return float(2 * 0.5 * math.erfc(abs(t) / math.sqrt(2)))


def annual_excess_t(index_level: pd.Series, bench_level: pd.Series) -> dict:
    """*t*-stat that the car index's **annual** return beats the benchmark's.

    Aligns both to year-end, takes annual simple returns, and tests the mean of the
    paired excess (index - bench) against 0. With ~20 annual points this is a genuine
    (if modest-power) test — the sign and magnitude are the finding.
    """
    a = index_level.resample("YE").last().pct_change().dropna()
    b = bench_level.resample("YE").last().pct_change().dropna()
    j = pd.concat([a, b], axis=1, keys=["idx", "bench"]).dropna()
    ex = j["idx"] - j["bench"]
    n = len(ex)
    if n < 2:
        return {"mean_excess": float("nan"), "t": float("nan"), "p": float("nan"), "n": n}
    se = ex.std(ddof=1) / math.sqrt(n)
    t = ex.mean() / se if se > 0 else float("nan")
    return {"mean_excess": float(ex.mean()), "t": float(t),
            "p": _t_p_value(t, n - 1), "n": n}


def newey_west_alpha_t(proxy_ret: pd.Series, bench_ret: pd.Series,
                       lags: int = 6) -> dict:
    """Newey-West (HAC) *t* of the monthly alpha from r_proxy = a + b*r_bench + e.

    Returns the OLS alpha (monthly + annualised), beta, the HAC standard error of
    alpha and its *t*. ``lags`` is the Bartlett-kernel truncation. The Signal-axis
    statistic for a proxy: is there alpha vs the market the proxy is exposed to?
    """
    j = pd.concat([proxy_ret, bench_ret], axis=1, keys=["y", "x"]).dropna()
    y = j["y"].to_numpy()
    x = j["x"].to_numpy()
    n = len(y)
    X = np.column_stack([np.ones(n), x])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    # Newey-West HAC covariance with Bartlett weights.
    S = (X * resid[:, None]).T @ (X * resid[:, None])
    for L in range(1, lags + 1):
        w = 1.0 - L / (lags + 1.0)
        Xe = X * resid[:, None]
        Gamma = Xe[L:].T @ Xe[:-L]
        S += w * (Gamma + Gamma.T)
    cov = XtX_inv @ S @ XtX_inv
    se_alpha = math.sqrt(cov[0, 0])
    a_m = float(beta[0])
    t_a = a_m / se_alpha if se_alpha > 0 else float("nan")
    return {
        "alpha_m": a_m,
        "alpha_ann": (1 + a_m) ** 12 - 1,
        "beta": float(beta[1]),
        "se_alpha": float(se_alpha),
        "t_alpha": float(t_a),
        "p_alpha": _t_p_value(t_a, n - 2),
        "n": n,
    }


# --------------------------------------------------------------------------- #
# The carry / illiquidity haircut — what owning the metal actually costs
# --------------------------------------------------------------------------- #
def net_of_carry_cagr(gross_cagr: float, round_trip_spread: float = 0.22,
                      hold_years: float = 7.0, carry_per_year: float = 0.025) -> dict:
    """Charge the collector-car's real frictions once on the NAV.

    A round trip through the auction/dealer channel costs ``round_trip_spread`` — a
    buyer's premium (~12-15%) plus a seller's commission (~5-10%), transport and
    inspection folded in — amortised over a typical ``hold_years``; plus
    ``carry_per_year`` annual carry (specialist insurance + climate storage +
    maintenance/servicing). Returns the net CAGR after these.
    """
    spread_drag_annual = (1 - round_trip_spread) ** (1 / hold_years) - 1  # negative
    net = (1 + gross_cagr) * (1 + spread_drag_annual) * (1 - carry_per_year) - 1
    return {
        "gross_cagr": gross_cagr,
        "spread_drag_annual": spread_drag_annual,
        "carry_per_year": -carry_per_year,
        "net_cagr": net,
    }


def control_recovers(level: pd.Series, planted_sign: int) -> dict:
    """Positive control: the engine recovers the planted drift's sign + a finite Sharpe."""
    s = summarize(level, periods_per_year=1.0)
    return {"cagr": s["cagr"], "sharpe": s["sharpe"],
            "sign_ok": int(np.sign(s["cagr"]) == planted_sign)}


# --------------------------------------------------------------------------- #
# De-smoothing — the appraisal-index artefact that fakes a low vol / high Sharpe
# --------------------------------------------------------------------------- #
def desmooth_returns(level: pd.Series) -> dict:
    """Geltner-style first-order un-smoothing of an appraisal/transaction index.

    Collectible indices are built from sparse, lagged appraisals and rolling averages,
    so their *reported* returns are serially correlated and their measured volatility is
    biased **down** — which fakes a low vol and a high Sharpe. The AR(1) un-smoothing
    ``r_u = (r_o - rho*r_o[-1]) / (1 - rho)`` recovers a truer return series. We report
    the reported vs de-smoothed annual vol and Sharpe. A large gap means the "great
    risk-adjusted return" was a measurement artefact, not a real edge.
    """
    r = level.pct_change().dropna()
    if len(r) < 3:
        return {"rho": float("nan"), "vol_obs": float("nan"), "vol_desmoothed": float("nan"),
                "sharpe_obs": float("nan"), "sharpe_desmoothed": float("nan")}
    rho = float(np.corrcoef(r.values[1:], r.values[:-1])[0, 1])
    ru = (r - rho * r.shift(1)) / (1.0 - rho)
    ru = ru.dropna()
    vol_obs = float(r.std(ddof=1))
    vol_ds = float(ru.std(ddof=1))
    sh_obs = float(r.mean() / vol_obs) if vol_obs > 0 else float("nan")
    sh_ds = float(ru.mean() / vol_ds) if vol_ds > 0 else float("nan")
    return {"rho": rho, "vol_obs": vol_obs, "vol_desmoothed": vol_ds,
            "sharpe_obs": sh_obs, "sharpe_desmoothed": sh_ds}
