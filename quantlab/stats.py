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

from .decompose import TRADING_DAYS_PER_YEAR, _excess_returns


def annualized_sharpe(
    returns: pd.Series,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    rf: float | pd.Series = 0.0,
) -> float:
    """Annualised Sharpe ratio, mean/std * sqrt(periods).

    ``rf`` is a scalar annualised risk-free rate or a per-period series aligned
    on the index (default 0 — historical behaviour). With short rates near
    4-5% (2023-26), ignoring it overstates a thin edge by ~2 bps/day.
    """
    r = np.asarray(_excess_returns(returns, rf, periods_per_year), dtype=float)
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(periods_per_year)) if sd > 0 else float("nan")


def sharpe_ci_bootstrap(
    returns: pd.Series,
    n_boot: int = 2000,
    alpha: float = 0.05,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    seed: int = 0,
    block_size: int | None = None,
    method: str = "cbb",
    rf: float | pd.Series = 0.0,
) -> dict:
    """Block-bootstrap confidence interval for the annualised Sharpe ratio.

    Daily returns are not i.i.d. — volatility clusters — so an i.i.d.
    bootstrap destroys the serial dependence and yields intervals that are
    too narrow. The default is therefore a **circular block bootstrap**
    (Politis & Romano 1994, "The Stationary Bootstrap", JASA 89; see also
    Politis & White 2004 on block-length choice): each resample concatenates
    blocks of ``block_size`` consecutive observations, with starts drawn
    uniformly and indices wrapping around the end of the sample.

    Parameters
    ----------
    block_size : int, optional
        Block length. ``None`` (default) uses the simple rate-optimal rule
        ``round(n ** (1/3))`` — the n^(1/3) growth rate of Politis-Romano —
        which is crude but adequate for a percentile interval.
    method : {'cbb', 'iid'}
        ``'cbb'`` (default) is the circular block bootstrap; ``'iid'``
        reproduces the legacy independent resampling (block length 1).
    rf : float or Series
        Scalar annualised risk-free rate, or per-period series aligned on the
        index. Default 0 keeps historical behaviour; at 2023-26 short rates
        the omission flatters a thin daily edge by ~2 bps/day.

    Degenerate resamples (zero standard deviation, possible with constant
    blocks) are excluded as NaN rather than recorded as Sharpe 0, which would
    silently shrink the interval toward zero; ``n_boot_valid`` reports how
    many resamples survived.

    Returns the point estimate, the (1-alpha) percentile interval, and the
    share of valid resamples with a *negative* Sharpe — a blunt p-value-like
    read on "could this be zero?".
    """
    r = np.asarray(_excess_returns(returns, rf, periods_per_year), dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    rng = np.random.default_rng(seed)
    point = annualized_sharpe(pd.Series(r), periods_per_year)

    if method == "iid":
        blk = 1
    elif method == "cbb":
        blk = int(block_size) if block_size is not None else max(1, round(n ** (1.0 / 3.0)))
    else:
        raise ValueError("method must be 'cbb' or 'iid'")
    blk = max(1, min(blk, n))

    ann = np.sqrt(periods_per_year)
    n_blocks = int(np.ceil(n / blk))
    offsets = np.arange(blk)
    boots = np.full(n_boot, np.nan)
    for b in range(n_boot):
        if blk == 1:
            idx = rng.integers(0, n, n)
        else:
            starts = rng.integers(0, n, n_blocks)
            idx = ((starts[:, None] + offsets[None, :]) % n).ravel()[:n]
        sample = r[idx]
        sd = sample.std(ddof=1)
        if sd > 0:
            boots[b] = sample.mean() / sd * ann

    valid = boots[np.isfinite(boots)]
    if valid.size == 0:
        lo = hi = frac_neg = float("nan")
    else:
        lo, hi = np.percentile(valid, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        frac_neg = float((valid < 0).mean())
    return {
        "sharpe": point,
        "ci_low": float(lo),
        "ci_high": float(hi),
        "frac_negative": frac_neg,
        "n_obs": int(n),
        "n_boot": int(n_boot),
        "n_boot_valid": int(valid.size),
        "block_size": int(blk),
        "method": method,
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
