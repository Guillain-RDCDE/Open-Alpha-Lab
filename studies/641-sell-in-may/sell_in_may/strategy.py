"""Strategy + inference for Study 641 — Sell in May (the Halloween indicator).

The claim: **"Sell in May and go away"** (Bouman & Jacobsen 2002) — U.S. equities earn almost
all of their return Nov->Apr; May->Oct is dead money (or worse), so a rational investor should
hold stocks in winter and sit in cash all summer.

Measurements:

* **The headline split** — mean monthly log return, Nov->Apr ("winter") vs May->Oct ("summer"),
  Welch *t* (the planned primary — the two buckets have unequal, autocorrelated variances so a
  Newey-West HAC *t* on the winter-dummy regression is the serial-correlation-robust
  cross-check).
* **The year-block test** — because a single "winter" (Nov(Y)->Apr(Y+1)) and its neighbouring
  "summer" (May(Y)->Oct(Y)) are each six non-overlapping months, we also pair them **one point
  per Halloween year** (~76 years): a **sign test** (does winter beat summer more than half the
  time?) and a **year-block bootstrap** of the mean paired gap — resampling whole *years*, never
  individual months, so no resample can split a season across two different market regimes.
* **The "handful of bad Octobers" decomposition** — the by-calendar-month mean return, and what
  the summer-minus-winter gap does once the worst individual Septembers/Octobers (1987, 2001,
  2008, ...) are pulled out.
* **The Halloween timer** — long equities Nov->Apr, cash (T-bill proxy) May->Oct, vs buy-and-hold
  and vs the mechanical reverse (long May->Oct, cash Nov->Apr). CAGR, **Sharpe excess-of-cash**
  (both legs — a timer that's in cash part of the year must not race a raw buy-and-hold Sharpe),
  max drawdown, turnover and cost drag.

The decisive number is the year-block-bootstrap / sign-test pair on the REAL monthly tape; the
honest question on top is whether the seasonal is worth trading once you charge it costs and
compare it to a *positive*-returning summer, not a hypothetical zero.
"""

from __future__ import annotations

from math import comb

import numpy as np
import pandas as pd

from .data import SUMMER_MONTHS, WINTER_MONTHS

MONTHS_PER_YEAR = 12


# --------------------------------------------------------------------------- #
# Inference primitives
# --------------------------------------------------------------------------- #
def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    """Welch t of mean(a) - mean(b) (unequal variances). NaN if either < 2 obs."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def newey_west_t(y: np.ndarray, d: np.ndarray, lags: int = 3) -> float:
    """HAC (Newey-West, Bartlett kernel) t of the slope in y = a + b*d.

    b is exactly the winter-minus-summer mean difference; the NW t is the
    serial-correlation-robust cross-check for the monthly log-return series (3 lags — monthly
    data, far less serial correlation than the daily case, but still worth a HAC check).
    """
    y = np.asarray(y, dtype=float)
    d = np.asarray(d, dtype=float)
    keep = ~np.isnan(y)
    y, d = y[keep], d[keep]
    n = len(y)
    X = np.column_stack([np.ones(n), d])
    XtX_inv = np.linalg.inv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    u = y - X @ beta
    s = X * u[:, None]
    S = s.T @ s
    for l in range(1, lags + 1):
        w = 1.0 - l / (lags + 1.0)
        G = s[l:].T @ s[:-l]
        S += w * (G + G.T)
    V = XtX_inv @ S @ XtX_inv
    se = np.sqrt(V[1, 1])
    return float(beta[1] / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial share k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


def binom_two_sided_p(k: int, n: int, p0: float = 0.5) -> float:
    """Exact two-sided binomial-test p-value: P(as-or-more-extreme than k successes of n, p0)."""
    if n == 0:
        return float("nan")
    pmf = np.array([comb(n, i) * p0 ** i * (1 - p0) ** (n - i) for i in range(n + 1)])
    obs = pmf[k]
    return float(pmf[pmf <= obs + 1e-12].sum())


# --------------------------------------------------------------------------- #
# The headline split
# --------------------------------------------------------------------------- #
def month_flags(ret: pd.Series) -> pd.Series:
    """True where the month-end date falls in a winter (Nov->Apr) month."""
    return pd.Series(ret.index.month, index=ret.index).isin(WINTER_MONTHS)


def headline_split(ret: pd.Series, nw_lags: int = 3) -> dict:
    """Winter vs summer monthly log return: means, gap, Welch t, NW t, Wilson-bounded hit rate."""
    w = month_flags(ret).values
    x = ret.values
    a, b = x[w], x[~w]
    return {
        "n_winter": int(w.sum()), "n_summer": int((~w).sum()),
        "winter_pct": float(np.nanmean(a) * 100), "summer_pct": float(np.nanmean(b) * 100),
        "gap_pct": float(np.nanmean(a) - np.nanmean(b)) * 100,
        # 6-month CUMULATIVE contribution of an "average" half (compounded, not simple x6):
        "winter_half_pct": float(np.expm1(np.nanmean(a) * 6) * 100),
        "summer_half_pct": float(np.expm1(np.nanmean(b) * 6) * 100),
        "welch_t": welch_t(a, b),
        "nw_t": newey_west_t(x, w.astype(float), lags=nw_lags),
    }


def by_calendar_month(ret: pd.Series) -> pd.DataFrame:
    """Mean monthly log return (%) and count, one row per calendar month 1..12."""
    df = pd.DataFrame({"ret": ret, "month": ret.index.month})
    g = df.groupby("month")["ret"].agg(["mean", "count"])
    g["mean_pct"] = g["mean"] * 100
    return g[["mean_pct", "count"]].reindex(range(1, 13))


# --------------------------------------------------------------------------- #
# Year-block pairing: one winter, one summer observation per "Halloween year"
# --------------------------------------------------------------------------- #
def halloween_year_pairs(ret: pd.Series) -> pd.DataFrame:
    """One row per Halloween year Y: cumulative log return of summer(Y) = May..Oct(Y) and
    winter(Y) = Nov(Y)..Apr(Y+1). Only years with all 6 summer AND all 6 winter months present
    are kept (partial edges of the sample are dropped, never padded)."""
    years = sorted(set(ret.index.year))
    rows = []
    for y in years:
        summer = ret[(ret.index.year == y) & (ret.index.month.isin(SUMMER_MONTHS))]
        winter = ret[((ret.index.year == y) & (ret.index.month.isin({11, 12}))) |
                     ((ret.index.year == y + 1) & (ret.index.month.isin({1, 2, 3, 4})))]
        if len(summer) == 6 and len(winter) == 6:
            rows.append({"year": y, "summer": float(summer.sum()), "winter": float(winter.sum())})
    out = pd.DataFrame(rows).set_index("year")
    out["gap"] = out["winter"] - out["summer"]
    return out


def sign_test_stats(pairs: pd.DataFrame) -> dict:
    """Does winter beat summer more often than a coin flip? Wilson CI + exact binomial p."""
    n = len(pairs)
    k = int((pairs["gap"] > 0).sum())
    lo, hi = wilson_interval(k, n)
    return {"n": n, "k_winter_wins": k, "hit_rate": k / n if n else float("nan"),
            "hit_lo": lo, "hit_hi": hi, "p_value": binom_two_sided_p(k, n, 0.5)}


def year_block_bootstrap(pairs: pd.DataFrame, n_boot: int = 10_000, seed: int = 641) -> dict:
    """Bootstrap the mean paired (winter - summer) gap by resampling whole Halloween YEARS.

    Resampling entire years (not individual months) respects the within-year correlation
    between the two legs and never lets a bootstrap draw straddle a season boundary.
    """
    rng = np.random.default_rng(seed)
    g = pairs["gap"].values
    n = len(g)
    obs = float(g.mean())
    se_analytic = g.std(ddof=1) / np.sqrt(n)
    boot = rng.choice(g, size=(n_boot, n), replace=True).mean(axis=1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {
        "n_years": n, "mean_gap_pct": obs * 100,
        "t_analytic": float(obs / se_analytic) if se_analytic > 0 else float("nan"),
        "boot_mean_pct": float(boot.mean() * 100), "boot_lo_pct": float(lo * 100),
        "boot_hi_pct": float(hi * 100),
        "boot_p_le0": float((boot <= 0).mean()),
    }


def gap_excluding_worst_months(ret: pd.Series, k: int = 5) -> dict:
    """Recompute the headline winter-summer gap after dropping the k worst individual
    Septembers and the k worst individual Octobers (the "handful of bad autumns" the folklore
    is secretly leaning on)."""
    sept = ret[ret.index.month == 9].sort_values().index[:k]
    octo = ret[ret.index.month == 10].sort_values().index[:k]
    dropped = sept.union(octo)
    trimmed = ret.drop(index=dropped)
    full = headline_split(ret)
    trimmed_s = headline_split(trimmed)
    return {
        "n_dropped": len(dropped),
        "full_gap_pct": full["gap_pct"], "full_welch_t": full["welch_t"],
        "trimmed_gap_pct": trimmed_s["gap_pct"], "trimmed_welch_t": trimmed_s["welch_t"],
        "dropped_dates": [str(d.date()) for d in sorted(dropped)],
    }


# --------------------------------------------------------------------------- #
# The Halloween timer — can you bank it?
# --------------------------------------------------------------------------- #
def _max_drawdown(curve: pd.Series) -> float:
    peak = curve.cummax()
    dd = curve / peak - 1.0
    return float(dd.min())


def halloween_timer(ret: pd.Series, cash_pct: pd.Series, cost_bps: float = 5.0,
                    reverse: bool = False) -> dict:
    """Backtest: long the equity leg in winter months, cash (T-bill proxy) in summer months
    (``reverse=True`` flips it — long summer, cash winter — the mechanical straw-man).

    Execution convention (the study's one documented lag): the Nov->Apr / May->Oct split is a
    fixed calendar rule known years in advance, so a position is held for the ENTIRE named month
    with zero look-ahead — no shift is needed (unlike an event signal, the rule doesn't need to
    "discover" anything intramonth). One switch = one round trip = 2 x one-way cost x NAV,
    charged in the two transition months (Apr->May and Oct->Nov), 2 switches/year.
    """
    idx = ret.index.intersection(cash_pct.index)
    ret, cash_pct = ret.loc[idx], cash_pct.loc[idx]
    cash_m = (cash_pct / 100.0) / 12.0          # ^IRX annualized % -> approximate monthly rate
    w = month_flags(ret)
    in_equity = w if not reverse else ~w
    port_ret = np.where(in_equity, ret.values, cash_m.values)
    switch = in_equity.astype(int).diff().fillna(0).abs().values   # 1 on a transition month
    cost = switch * (2 * cost_bps / 1e4)
    net_ret = port_ret - cost
    curve = pd.Series((1 + net_ret).cumprod(), index=idx)
    bh_curve = pd.Series((1 + ret.values).cumprod(), index=idx)
    n_years = len(idx) / 12.0
    cagr = float(curve.iloc[-1] ** (1 / n_years) - 1)
    bh_cagr = float(bh_curve.iloc[-1] ** (1 / n_years) - 1)
    excess = net_ret - cash_m.values
    bh_excess = ret.values - cash_m.values
    sharpe_excess = float(np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(12)) if np.std(excess, ddof=1) > 0 else float("nan")
    bh_sharpe_excess = float(np.mean(bh_excess) / np.std(bh_excess, ddof=1) * np.sqrt(12)) if np.std(bh_excess, ddof=1) > 0 else float("nan")
    n_switches = int(switch.sum())
    return {
        "n_months": len(idx), "n_years": n_years,
        "cagr_pct": cagr * 100, "bh_cagr_pct": bh_cagr * 100,
        "sharpe_excess": sharpe_excess, "bh_sharpe_excess": bh_sharpe_excess,
        "max_dd_pct": _max_drawdown(curve) * 100, "bh_max_dd_pct": _max_drawdown(bh_curve) * 100,
        "n_switches": n_switches, "cost_bps": cost_bps,
        "one_way_trades_per_yr": n_switches / n_years,
        "total_cost_drag_pct": float(cost.sum()) * 100,
        "final_wealth": float(curve.iloc[-1]), "bh_final_wealth": float(bh_curve.iloc[-1]),
    }


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(ret: pd.Series) -> dict:
    """Run the headline Welch/NW split on a synthetic monthly-return world."""
    return headline_split(ret)
