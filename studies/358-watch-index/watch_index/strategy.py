"""Strategy + inference for Study 358 — "watches are an asset class".

The claim: buying/flipping luxury watches (Rolex/Patek/AP on the secondary market) is a
real *investment* that beats the S&P. We test the strongest tradable version of that:

1. **The resale index itself** (hardcoded, cited, approximate) vs SPY on the same window
   — CAGR, vol, max drawdown, and a *t*-stat of the annual excess return. This is the
   "did the watches actually beat stocks?" question, on the index believers quote.
2. **The tradable equity proxies** (Watches of Switzerland, Richemont) vs SPY — the only
   thing a public investor can actually buy that is *about* the watch trade. Monthly
   returns, Newey-West *t* of the alpha vs SPY, Sharpe, max drawdown.
3. **The carry/illiquidity haircut** — what flipping really costs: dealer spread (buy
   retail-plus, sell at a wholesale discount), insurance, and the months a piece sits
   unsold. We charge it once on the NAV and show where the "return" goes.

Inference: an annual-excess *t*-stat for the index, a Newey-West (HAC) *t* of the monthly
proxy alpha vs SPY, Sharpe ratios, and a max-drawdown decomposition. Pure numpy/pandas;
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
    """*t*-stat that the resale index's **annual** return beats the benchmark's.

    Aligns both to year-end, takes annual simple returns, and tests the mean of the
    paired excess (index - bench) against 0. Small-sample (≈7 years), so this is a
    weak test by construction — which is itself the finding.
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
# The carry / illiquidity haircut — what flipping actually costs
# --------------------------------------------------------------------------- #
def net_of_carry_cagr(gross_cagr: float, round_trip_spread: float = 0.20,
                      hold_years: float = 3.0, insure_per_year: float = 0.01) -> dict:
    """Charge the watch-flip's real frictions once on the NAV.

    A buy-high / sell-at-a-wholesale-discount round trip costs ``round_trip_spread``
    (dealer margin + grey-market discount, one-way each side already folded in),
    amortised over a typical ``hold_years``; plus ``insure_per_year`` annual carry
    (insurance/storage/servicing). Returns the net CAGR after these.
    """
    spread_drag_annual = (1 - round_trip_spread) ** (1 / hold_years) - 1  # negative
    net = (1 + gross_cagr) * (1 + spread_drag_annual) * (1 - insure_per_year) - 1
    return {
        "gross_cagr": gross_cagr,
        "spread_drag_annual": spread_drag_annual,
        "insure_per_year": -insure_per_year,
        "net_cagr": net,
    }


def control_recovers(level: pd.Series, planted_sign: int) -> dict:
    """Positive control: the engine recovers the planted drift's sign + a finite Sharpe."""
    s = summarize(level)
    return {"cagr": s["cagr"], "sharpe": s["sharpe"],
            "sign_ok": int(np.sign(s["cagr"]) == planted_sign)}
