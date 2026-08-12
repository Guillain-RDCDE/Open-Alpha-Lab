"""Strategy + inference for Study 876 — Industry-Relative MAX.

The claim (a refinement of study 365 / Bali-Cakici-Whitelaw 2011): the raw MAX sort ranks a
name on its own **maximum daily return** last month and buys the low-MAX / sells the high-MAX
tail. Part of a name's MAX is just **sector-wide volatility** (a whole sector can be jumpy for
macro reasons), which is noise for a *lottery-demand* story. Subtract the sector peers' median
MAX to get the **industry-relative** MAX — a cleaner proxy for *idiosyncratic* lottery demand —
and ask whether the negative MAX→return relation **sharpens** or **dies**.

This module:

  * builds, each month, both the **raw** MAX and the **industry-relative** MAX (raw MAX minus
    the median MAX of the name's sector peers);
  * sorts the cross-section into **quintiles** each month and earns each quintile's *next-month*
    return (the panel is already lagged: MAX at month-end pairs with month t+1's return — one
    execution lag, documented);
  * forms the **long-short** spread Q1 (low MAX) − Q5 (high MAX);
  * tests its mean with a **Newey-West (HAC) t-stat**, a **sign-flip placebo null**, a two-era
    robustness cut, the monthly **win-rate** vs a 50% coin, and charges **one-way costs ×
    turnover** plus a **short borrow** on the high-MAX leg;
  * runs the identical machinery on **raw** vs **industry-relative** MAX so the two are graded
    head-to-head (the honest contrast with study 365).

The decisive object is the HAC t of the industry-relative long-short on the *real* tape.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12


# --------------------------------------------------------------------------- #
# The industry-relative MAX signal
# --------------------------------------------------------------------------- #
def industry_relative_max(mx: pd.DataFrame, sectors: pd.Series) -> pd.DataFrame:
    """Industry-relative MAX: each name's MAX minus the **median MAX of its sector**, month by
    month.

    Vectorised per sector (a handful of groups), not per date: for each sector we take the
    row-wise (cross-sectional, within-month) median MAX across its member columns and subtract
    it from every member. Using the full-sector median (own name included) rather than a
    strict leave-one-out is a documented choice — with 2–12 names per sector the leave-one-out
    shift is immaterial and the full median is a stabler peer benchmark.
    """
    adj = pd.DataFrame(np.nan, index=mx.index, columns=mx.columns)
    for sec in pd.unique(sectors.values):
        cols = [c for c in mx.columns if sectors.get(c) == sec]
        if not cols:
            continue
        med = mx[cols].median(axis=1)               # within-month sector median MAX
        adj[cols] = mx[cols].sub(med, axis=0)
    return adj


# --------------------------------------------------------------------------- #
# Quintile sort
# --------------------------------------------------------------------------- #
def quantile_returns(signal: pd.DataFrame, fwd: pd.DataFrame, n_q: int = 5) -> pd.DataFrame:
    """Equal-weight next-month return of each ``signal`` quantile, month by month.

    For each month with at least ``2*n_q`` valid (signal, fwd_ret) pairs, rank names by
    ``signal``, split into ``n_q`` quantiles (Q1 = lowest signal … Qn = highest), and average
    each quantile's forward return. Returns a DataFrame indexed by month with columns
    ``Q1..Qn``. ``signal`` and ``fwd`` must share index and columns.
    """
    rows = {}
    for dt in signal.index:
        s = signal.loc[dt]
        f = fwd.loc[dt]
        both = pd.concat([s.rename("sig"), f.rename("fwd")], axis=1).dropna()
        if len(both) < 2 * n_q:
            continue
        try:
            q = pd.qcut(both["sig"].rank(method="first"), n_q, labels=False)
        except ValueError:
            continue
        means = both.groupby(q)["fwd"].mean()
        rows[dt] = {f"Q{int(i) + 1}": means.get(i, np.nan) for i in range(n_q)}
    out = pd.DataFrame.from_dict(rows, orient="index")
    out.index.name = "date"
    return out.dropna(how="any")


def long_short(qret: pd.DataFrame, low: str = "Q1", high: str = "Q5",
               cost_bps: float = 0.0, borrow_ann_bps: float = 0.0,
               turnover: float = 1.0) -> pd.Series:
    """Long low-MAX (``low``) / short high-MAX (``high``) monthly spread, net of frictions.

    Gross spread is ``qret[low] - qret[high]`` (buy the dull tail, short the lottery tail).
    A one-way ``cost_bps`` × ``turnover`` charge hits *each* of the two legs every rebalance
    (a full two-sided round-trip is ``2 * cost_bps * turnover`` per month, the conservative
    monthly-rebalance case). ``borrow_ann_bps`` charges short financing on the high-MAX leg
    (annual, pro-rated monthly). ``turnover`` defaults to 1.0 (a one-month signal churns hard).
    """
    s = (qret[low] - qret[high]).dropna()
    monthly_cost = 2.0 * cost_bps * 1e-4 * turnover
    monthly_borrow = borrow_ann_bps * 1e-4 / MONTHS_PER_YEAR
    return s - monthly_cost - monthly_borrow


# --------------------------------------------------------------------------- #
# Inference primitives (shared house set, copied from study 803)
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def newey_west_t(x: np.ndarray, lags: int | None = None) -> float:
    """Newey-West (HAC, Bartlett kernel) t of mean(x) vs 0.

    Default lag rule ``floor(4*(n/100)^(2/9))`` — the desk's monthly-series convention.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 6:
        return float("nan")
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    mu = x.mean()
    e = x - mu
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# --------------------------------------------------------------------------- #
# Leg / spread statistics
# --------------------------------------------------------------------------- #
def ann_vol(r: pd.Series) -> float:
    r = r.dropna()
    return float(r.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR)) if len(r) > 1 else float("nan")


def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 2 or r.std(ddof=1) == 0:
        return float("nan")
    return float(r.mean() / r.std(ddof=1) * np.sqrt(MONTHS_PER_YEAR))


def win_rate(r: pd.Series) -> float:
    r = r.dropna()
    return float((r > 0).mean()) if len(r) else float("nan")


def placebo_pvalue(spread: pd.Series, n_draws: int = 20_000, seed: int = 876) -> dict:
    """Sign-flip placebo null for the long-short mean.

    Randomly flip the sign of each monthly spread observation ``n_draws`` times and ask how
    often the resampled mean is **>=** the observed mean (one-sided to the claim's direction:
    low-MAX beats high-MAX, so the edge should be *positive*). Returns the observed mean, the
    placebo mean (~0), and the empirical p-value.
    """
    x = spread.dropna().to_numpy(dtype=float)
    n = x.size
    if n < 6:
        return {"n": n, "obs_mean": float("nan"), "placebo_mean": float("nan"),
                "p_value": float("nan")}
    rng = np.random.default_rng(seed)
    obs = float(x.mean())
    means = np.empty(n_draws)
    for i in range(n_draws):
        signs = rng.choice([-1.0, 1.0], size=n)
        means[i] = (x * signs).mean()
    p = float((means >= obs).mean())
    return {"n": n, "obs_mean": obs, "placebo_mean": float(means.mean()), "p_value": p}


def spread_stats(s: pd.Series) -> dict:
    """Headline statistics for a spread series: n, annualised mean, vol, Sharpe, HAC t,
    win-rate, placebo p."""
    s = s.dropna()
    pl = placebo_pvalue(s)
    return {
        "n": int(len(s)),
        "mean_bps": float(s.mean() * 1e4) if len(s) else float("nan"),
        "mean_ann": float(s.mean() * MONTHS_PER_YEAR) if len(s) else float("nan"),
        "vol_ann": ann_vol(s),
        "sharpe": sharpe(s),
        "tstat": newey_west_t(s.to_numpy(dtype=float)),
        "t_1s": one_sample_t(s.to_numpy(dtype=float)),
        "win": win_rate(s),
        "p_placebo": pl["p_value"],
    }


def quantile_summary(qret: pd.DataFrame) -> pd.DataFrame:
    """Per-quantile annualised mean, vol, Sharpe — the cross-sectional monotonicity card."""
    rows = {}
    for c in qret.columns:
        r = qret[c].dropna()
        rows[c] = {
            "mean_ann": float(r.mean() * MONTHS_PER_YEAR),
            "vol_ann": ann_vol(r),
            "sharpe": sharpe(r),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


# --------------------------------------------------------------------------- #
# Head-to-head: raw MAX vs industry-relative MAX
# --------------------------------------------------------------------------- #
def run_sort(mx: pd.DataFrame, fwd: pd.DataFrame, sectors: pd.Series,
             adjusted: bool, n_q: int = 5) -> dict:
    """Build the signal (raw or industry-relative MAX), quantile-sort, and return the
    long-short (Q1−Q5) spread series plus its per-quantile summary."""
    sig = industry_relative_max(mx, sectors) if adjusted else mx
    qret = quantile_returns(sig, fwd, n_q=n_q)
    high = f"Q{n_q}"
    spread = long_short(qret, low="Q1", high=high)
    return {"qret": qret, "spread": spread, "high": high}


def timer_stats(spread_gross: pd.Series, qret: pd.DataFrame, high: str = "Q5",
                cost_bps: float = 1.0, borrow_ann_bps: float = 50.0,
                turnover: float = 1.0) -> dict:
    """Cost the long-low-MAX / short-high-MAX book, monthly rebalance.

    2 sides × one-way cost × turnover × NAV per rebalance; the high-MAX short leg pays
    ``borrow_ann_bps`` annual borrow, pro-rated monthly.
    """
    net = long_short(qret, low="Q1", high=high, cost_bps=cost_bps,
                     borrow_ann_bps=borrow_ann_bps, turnover=turnover).dropna()
    g = spread_gross.dropna()
    monthly_cost = 2.0 * cost_bps * 1e-4 * turnover
    monthly_borrow = borrow_ann_bps * 1e-4 / MONTHS_PER_YEAR
    return {
        "n": int(len(net)),
        "gross_bps": float(g.mean() * 1e4),
        "cost_bps": (monthly_cost + monthly_borrow) * 1e4,
        "net_bps": float(net.mean() * 1e4),
        "t_net": one_sample_t(net.to_numpy(dtype=float)),
        "sharpe_net": sharpe(net),
        "ann_net_pct": float(net.mean() * MONTHS_PER_YEAR * 100),
    }


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(panel: dict, adjusted: bool = True, n_q: int = 5) -> dict:
    """Run the headline spread stats on a synthetic panel, raw or industry-adjusted."""
    r = run_sort(panel["max"], panel["fwd_ret"], panel["sectors"], adjusted=adjusted, n_q=n_q)
    ss = spread_stats(r["spread"])
    return {"mean_bps": ss["mean_bps"], "t_nw": ss["tstat"], "n": ss["n"],
            "sharpe": ss["sharpe"]}
