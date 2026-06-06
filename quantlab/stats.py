"""Statistical rigour helpers — the part that separates a chart from a claim.

Two questions a sceptical quant immediately asks about "overnight Sharpe 0.77":
  1. Is it *statistically* distinguishable from zero, or noise over a finite
     sample? -> ``sharpe_ci_bootstrap``.
  2. Is it *alpha*, or just compensation for carrying equity (gap) risk every
     night? -> ``beta_decomposition``: regress the overnight leg on the market
     (close-close) and split its mean into beta*market + residual alpha.

Everything here is deterministic given a seed, and unit-tested.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .decompose import TRADING_DAYS_PER_YEAR


def annualized_sharpe(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> float:
    r = np.asarray(returns, dtype=float)
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan")


def sharpe_ci_bootstrap(
    returns: pd.Series,
    n_boot: int = 2000,
    alpha: float = 0.05,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    seed: int = 0,
) -> dict:
    """Bootstrap confidence interval for the annualised Sharpe ratio.

    Resamples daily returns with replacement ``n_boot`` times. Returns the point
    estimate, the (1-alpha) percentile interval, and the share of resamples with
    a *negative* Sharpe — a blunt p-value-like read on "could this be zero?".
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    rng = np.random.default_rng(seed)
    point = annualized_sharpe(pd.Series(r), periods_per_year)

    boots = np.empty(n_boot)
    for b in range(n_boot):
        sample = r[rng.integers(0, n, n)]
        sd = sample.std(ddof=1)
        boots[b] = sample.mean() / sd * np.sqrt(periods_per_year) if sd > 0 else 0.0

    lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {
        "sharpe": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "frac_negative": float((boots < 0).mean()),
        "n_obs": int(n),
        "n_boot": int(n_boot),
    }


def beta_decomposition(
    dec: pd.DataFrame,
    leg: str = "overnight",
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> dict:
    """Split a leg's mean return into market beta vs residual alpha.

    Regress the leg's daily return on the daily close-close (buy & hold) return:
        r_leg = alpha + beta * r_market + eps
    Then the leg's average daily return decomposes as
        mean(r_leg) = alpha + beta * mean(r_market).
    A large ``beta`` means much of the "overnight edge" is just being long the
    market overnight (gap-risk premium / disguised beta), not a distinct alpha.
    Returns daily-and-annualised alpha plus beta and R^2.
    """
    if leg not in ("overnight", "intraday"):
        raise ValueError("leg must be 'overnight' or 'intraday'")
    y = dec[f"r_{leg}"].to_numpy(dtype=float)
    x = dec["r_close_close"].to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]

    # OLS with intercept via np.polyfit (degree 1): y = beta*x + alpha
    beta, alpha = np.polyfit(x, y, 1)
    resid = y - (beta * x + alpha)
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "alpha_daily_bps": float(alpha * 1e4),
        "alpha_ann_pct": float(((1 + alpha) ** periods_per_year - 1) * 100),
        "beta": float(beta),
        "r_squared": float(r2),
        "mean_leg_bps": float(y.mean() * 1e4),
        "beta_contrib_bps": float(beta * x.mean() * 1e4),
    }
