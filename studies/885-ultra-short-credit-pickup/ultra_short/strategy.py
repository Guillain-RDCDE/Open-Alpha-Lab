"""Strategy & inference for Study 885 — Ultra-Short Credit Pickup.

The claim (a mechanical / structural / paid-premium idea): ultra-short investment-
grade credit ETFs (JPST / ICSH / MINT) hold ~AA-/A short-maturity IG paper and are
paid a small **spread over T-bills** for a tiny sliver of credit + duration risk.
If the pickup is a genuine structural premium, the ultra-short credit sleeve should
deliver a **higher excess-of-bills Sharpe** than the bill vehicles themselves —
i.e. a better *reward per unit of risk*, not merely a higher raw return — while
taking only marginally more drawdown. We test that on the live, fee-paying tape and
are honest about the 2020 (COVID) and 2022 (rate-hike) stress windows.

The machinery, all offline once the cache exists:

* ``daily_returns`` — daily simple total returns from the cached close panel.
* ``excess`` — daily return in excess of the cash leg (BIL by default) — every
  Sharpe race and every mean/premium is **excess-of-cash**.
* ``ann_sharpe`` / ``sharpe_ci`` — annualised excess Sharpe + a block-bootstrap CI
  (reused from ``quantlab``), the reward-per-risk race.
* ``newey_west_t`` / ``hac_mean`` — HAC (Newey-West) t of a daily mean series, the
  significance of the credit-minus-bill pickup.
* ``max_drawdown`` / ``calendar_year_table`` — the risk the pickup is paid for and
  the year-by-year (incl. the 2020 & 2022 stress) view.
* ``era_cut`` — a pre/post 2019 split, the sub-era robustness.
* ``net_of_cost_excess`` — a COSTED version: one-way spread × NAV × annual turnover
  per year drag on the (long-only) sleeve; no shorts, no borrow.

No RNG anywhere in the real-tape path — every number is a deterministic function of
the cached tape. The only randomness lives in the synthetic world
(``data.synthetic_world``, fixed-seed) and the fixed-seed bootstrap CI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Returns
# --------------------------------------------------------------------------- #
def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Wide daily simple total returns from the close panel."""
    return prices.sort_index().pct_change()


def align_common(returns: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Rows where every column in ``cols`` has a return (the common sample)."""
    return returns[cols].dropna()


def excess(returns: pd.DataFrame, col: str, rf: str = "BIL") -> pd.Series:
    """Daily excess return of ``col`` over the cash leg (BIL by default)."""
    return (returns[col] - returns[rf]).dropna()


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of the difference in means between two independent samples."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def newey_west_t(x: np.ndarray, lags: int | None = None) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0.

    ``lags=None`` uses the rule of thumb ``floor(4*(n/100)^(2/9))``.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    mu = x.mean()
    u = x - mu
    gamma0 = float(u @ u) / n
    var = gamma0
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        cov = float(u[l:] @ u[:-l]) / n
        var += 2.0 * w * cov
    if var <= 0:
        return float("nan")
    se = np.sqrt(var / n)
    return float(mu / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


def hac_mean(series: pd.Series, lags: int | None = None) -> dict:
    """HAC (Newey-West) mean + t of a daily series (the pickup significance test)."""
    x = np.asarray(pd.Series(series).dropna(), dtype=float)
    n = len(x)
    lags_used = lags if lags is not None else int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    t = newey_west_t(x, lags_used)
    mean = float(x.mean()) if n else float("nan")
    return {
        "mean_bps": mean * 1e4,
        "mean_bps_yr": mean * TRADING_DAYS * 1e4,
        "t_nw": t,
        "t_1s": one_sample_t(x),
        "n": n,
        "lags": lags_used,
    }


# --------------------------------------------------------------------------- #
# Reward per unit of risk — the excess-of-cash Sharpe race
# --------------------------------------------------------------------------- #
def ann_sharpe(excess_ret: pd.Series) -> float:
    """Annualised Sharpe of a daily EXCESS-of-cash return series."""
    e = pd.Series(excess_ret).dropna()
    sd = e.std(ddof=1)
    return float(e.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")


def ann_return(daily_ret: pd.Series) -> float:
    """Geometric annualised total return (%) from daily simple returns."""
    r = pd.Series(daily_ret).dropna()
    if len(r) == 0:
        return float("nan")
    return ((1.0 + r).prod() ** (TRADING_DAYS / len(r)) - 1.0) * 100.0


def ann_vol(daily_ret: pd.Series) -> float:
    """Annualised volatility (%) of a daily return series."""
    r = pd.Series(daily_ret).dropna()
    return float(r.std(ddof=1) * np.sqrt(TRADING_DAYS) * 100.0)


def sharpe_ci(excess_ret: pd.Series, n_boot: int = 2000, seed: int = 885) -> dict:
    """Circular-block-bootstrap CI for the annualised excess Sharpe.

    Thin wrapper over ``quantlab.stats.sharpe_ci_bootstrap`` (default cbb block
    length). Returns the point estimate, the 95% interval and the share of
    resamples with a negative Sharpe.
    """
    from quantlab.stats import sharpe_ci_bootstrap
    e = pd.Series(excess_ret).dropna()
    return sharpe_ci_bootstrap(e, n_boot=n_boot, seed=seed,
                               periods_per_year=TRADING_DAYS)


# --------------------------------------------------------------------------- #
# Risk — drawdown & the calendar-year table
# --------------------------------------------------------------------------- #
def max_drawdown(prices: pd.Series) -> dict:
    """Max drawdown of a daily total-return price series: depth (%), peak/trough."""
    px = pd.Series(prices).dropna()
    peak = px.cummax()
    dd = px / peak - 1.0
    trough = dd.idxmin()
    peak_date = px.loc[:trough].idxmax()
    return {"depth_pct": float(dd.min() * 100.0),
            "peak": str(peak_date.date()), "trough": str(trough.date())}


def calendar_year_table(returns: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Calendar-year total return (%) per column — the year-by-year (incl. stress)."""
    out = {}
    for c in cols:
        r = returns[c].dropna()
        out[c] = ((1.0 + r).groupby(r.index.year).prod() - 1.0) * 100.0
    return pd.DataFrame(out)


def era_cut(excess_ret: pd.Series, split: str = "2019-01-01") -> dict:
    """Pre/post-``split`` excess-mean + HAC t (the sub-era robustness cut)."""
    e = pd.Series(excess_ret).dropna()
    early = e[e.index < pd.Timestamp(split)]
    late = e[e.index >= pd.Timestamp(split)]
    return {
        "early": hac_mean(early), "late": hac_mean(late),
        "welch_t": welch_t(early.to_numpy(), late.to_numpy()),
        "split": split,
    }


# --------------------------------------------------------------------------- #
# The costed version (one-way spread x NAV x turnover; long-only, no borrow)
# --------------------------------------------------------------------------- #
def net_of_cost_excess(excess_ret: pd.Series, cost_bps_oneway: float = 1.0,
                       turnover_yr: float = 1.0) -> dict:
    """Cost the (long-only) sleeve: annual drag = one-way spread x NAV x turnover.

    An ultra-short credit sleeve is bought once and rolls with maturities inside the
    fund; the retail implementation cost is the ETF spread paid on entry plus any
    rebalancing you do yourself. We charge a conservative ``turnover_yr`` round-trips
    a year of ``cost_bps_oneway`` each way (2 sides), plus the ETF expense ratio is
    already inside the net-of-fee tape. No shorts, no borrow.

    Returns gross vs net annualised excess mean (bps/yr) and the net excess Sharpe.
    """
    e = pd.Series(excess_ret).dropna()
    gross_bps_yr = float(e.mean() * TRADING_DAYS * 1e4)
    drag_bps_yr = 2.0 * cost_bps_oneway * turnover_yr
    net_bps_yr = gross_bps_yr - drag_bps_yr
    # Net Sharpe: subtract the daily drag from the excess series (vol ~unchanged).
    drag_daily = drag_bps_yr / 1e4 / TRADING_DAYS
    net_series = e - drag_daily
    return {
        "gross_bps_yr": gross_bps_yr,
        "drag_bps_yr": drag_bps_yr,
        "net_bps_yr": net_bps_yr,
        "net_sharpe": ann_sharpe(net_series),
        "cost_bps_oneway": cost_bps_oneway,
        "turnover_yr": turnover_yr,
    }
