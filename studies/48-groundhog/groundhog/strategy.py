"""Return seasonality — the same-month predictor, the long-short, and the honest controls.

Heston & Sadka (2008): a stock's average return in a given calendar month forecasts its return in that
same month in the future. Each month we score every stock by its mean return in the *upcoming calendar
month* over the trailing (up to 20) years, go long the high-seasonal names and short the low. The
decisive control: do the *other* months' history predict too? If only the same month works, it's a
genuine seasonal — not generic momentum. We also charge turnover, because the book reshuffles monthly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS = 12


def seasonal_hedge(
    returns: pd.DataFrame, same_month: bool = True, q: float = 0.2,
    max_years: int = 20, min_hist: int = 5, min_names: int = 30, warmup: int = 60
) -> pd.Series:
    """Monthly long-high/short-low return-seasonality hedge.

    For each month *t* (after ``warmup``), score each stock by the mean of its **strictly past** returns
    in the same calendar month (lags 12, 24, … up to ``max_years``×12) when ``same_month``; or by its
    recent *other*-month returns when ``same_month=False`` (the control). Long the top-``q`` scorers,
    short the bottom-``q``, equal-weight; return the monthly hedge return. Requires at least
    ``min_hist`` same-month observations and ``min_names`` valid stocks.
    """
    R = returns.to_numpy(dtype=float)
    idx = returns.index
    month = idx.month.to_numpy()
    out = {}
    for t in range(warmup, len(idx)):
        m = month[t]
        if same_month:
            rows = [s for s in range(t - 12, max(t - max_years * 12 - 1, -1), -12) if s >= 0]
        else:
            rows = [s for s in range(t - 1, max(t - max_years * 12 - 1, -1), -1) if month[s] != m]
        if len(rows) < min_hist:
            continue
        with np.errstate(invalid="ignore"):
            pred = np.nanmean(R[rows, :], axis=0)
        nxt = R[t, :]
        mask = np.isfinite(pred) & np.isfinite(nxt)
        if mask.sum() < min_names:
            continue
        p, n = pred[mask], nxt[mask]
        lo, hi = p <= np.quantile(p, q), p >= np.quantile(p, 1.0 - q)
        out[idx[t]] = float(n[hi].mean() - n[lo].mean())
    return pd.Series(out, name="seasonal" if same_month else "control")


def net_of_cost(hedge: pd.Series, cost_bps: float, turnover: float = 1.6) -> pd.Series:
    """Approximate net hedge after costs. The book reshuffles ~fully each month; ``turnover`` (legs ×
    fraction replaced) times ``cost_bps`` is charged monthly. Default 1.6 ≈ 80% of a two-sided book."""
    return (hedge - turnover * cost_bps / 1e4).rename(hedge.name)


def stats(spread: pd.Series, periods_per_year: int = MONTHS) -> dict:
    """Annualised mean, Sharpe, the Lo (2002) t-stat, hit-rate for a monthly hedge series."""
    r = pd.Series(spread).astype(float).dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("mean_ann", "sharpe", "tstat", "hit_rate", "n")}
    sr_m = r.mean() / r.std(ddof=1) if r.std(ddof=1) > 0 else np.nan
    se = np.sqrt((1.0 + 0.5 * sr_m**2) / len(r))
    return {
        "mean_ann": float(r.mean() * periods_per_year),
        "sharpe": float(sr_m * np.sqrt(periods_per_year)),
        "tstat": float(sr_m / se) if se > 0 else np.nan,
        "hit_rate": float((r > 0).mean()),
        "n": int(len(r)),
    }


def breakeven_cost_bps(hedge: pd.Series, turnover: float = 1.6) -> float:
    """The per-trade cost (bp) at which the net mean hedge return hits zero."""
    return float(pd.Series(hedge).astype(float).dropna().mean() * 1e4 / turnover)
