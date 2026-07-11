"""Strategy + inference for Study 650 — Heating-Oil-Seasonality.

The claim: **heating oil rallies into winter** — cold-weather distillate demand pulls the
futures price up through the year, in two stages the folklore names separately: an **autumn
build** (Sep-Nov, the market pricing the coming season ahead of time) and a **winter draw**
(Dec-Feb, physical inventories fall as furnaces run and price should hold or extend the rally).

Measurements:

* **Per-month mean return** (12 one-sample *t*-stats vs 0) with a **Bonferroni-12** bar
  (|t| >= ~3.0, alpha=0.05/12 two-sided at n~25) — testing 12 months is 12 chances to find one
  by luck, so the naive |t| >= 2 bar is the wrong one here.
* **Group Welch *t*'s** — autumn-build vs off-season (Mar-Aug), winter-draw vs off-season, and
  the combined heating window (Sep-Feb) vs off-season. Non-overlapping monthly observations,
  Welch is the planned primary (unequal variances allowed).
* **A seasonal timer** — long the heating window (Sep-Feb), the 13-week T-bill otherwise, one
  round trip in and one out per year; gross and net of one-way costs x NAV, raced against
  buy-and-hold on excess-of-cash Sharpe (like-for-like).
* **Third axis — the ETF wrapper gap.** UHN (the actual retail vehicle) held over the same
  Sep-Feb window vs the HO=F splice over the identical calendar dates, paired per season: does
  the wrapper eat further into whatever the futures roll already costs?

The decisive numbers are the group Welch *t*'s and the Bonferroni month table on the REAL HO=F
tape; the honest question on top is whether a real, investable vehicle ever paid for any of it —
and whether that vehicle still exists.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12
BONFERRONI_T = 3.0   # approx |t| for alpha=0.05/12 two-sided at n~25 (exact: t.ppf ~ 3.15)


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


def one_sample_t(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


# --------------------------------------------------------------------------- #
# Per-month table
# --------------------------------------------------------------------------- #
def month_stats(returns: pd.Series) -> pd.DataFrame:
    """Per-calendar-month mean, std, count and one-sample t-stat (vs 0) for a monthly series.

    Returns a frame indexed 1..12. Bonferroni-12 bar for "survives multiple testing" is
    ``BONFERRONI_T`` (|t| >= ~3.0).
    """
    s = pd.Series(returns).astype(float)
    s.index = pd.DatetimeIndex(s.index)
    rows = {}
    for m in range(1, 13):
        vals = s[s.index.month == m].dropna().to_numpy()
        n = len(vals)
        if n < 2:
            rows[m] = {"mean": np.nan, "std": np.nan, "n": n, "t": np.nan}
            continue
        rows[m] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)),
                   "n": int(n), "t": one_sample_t(vals)}
    return pd.DataFrame(rows).T.rename_axis("month")


# --------------------------------------------------------------------------- #
# Group tests — autumn-build / winter-draw / heating window vs off-season
# --------------------------------------------------------------------------- #
def group_welch(returns: pd.Series, months_a: list[int], months_b: list[int]) -> dict:
    """Welch t of mean(months_a) - mean(months_b) for a monthly return series."""
    s = pd.Series(returns).astype(float)
    s.index = pd.DatetimeIndex(s.index)
    a = s[s.index.month.isin(months_a)].dropna().to_numpy()
    b = s[s.index.month.isin(months_b)].dropna().to_numpy()
    return {
        "mean_a": float(np.nanmean(a)) if len(a) else float("nan"),
        "mean_b": float(np.nanmean(b)) if len(b) else float("nan"),
        "n_a": len(a), "n_b": len(b),
        "t": welch_t(a, b),
    }


# --------------------------------------------------------------------------- #
# The seasonal timer
# --------------------------------------------------------------------------- #
def seasonal_timer(returns: pd.Series, tbill: pd.Series, heat_months: list[int],
                    cost_bps: float = 0.0) -> pd.Series:
    """Long the heating window, T-bill otherwise; one-way cost charged at every switch.

    ``tbill`` is a monthly cash-return series aligned on ``returns.index``. A position switch
    (flat->long or long->flat) charges ``cost_bps`` one-way x NAV once, on the switching month —
    two switches per year (enter Sep, exit end-Feb) = one round trip.
    """
    r = pd.Series(returns).astype(float)
    r.index = pd.DatetimeIndex(r.index)
    cash = pd.Series(tbill).astype(float).reindex(r.index).fillna(0.0)

    pos = pd.Series(0.0, index=r.index)
    pos[r.index.month.isin(heat_months)] = 1.0

    gross = pos * r + (pos == 0).astype(float) * cash
    switches = pos.diff().abs()
    switches.iloc[0] = pos.iloc[0]        # first observation counts as a switch if starting long
    cost = switches * (cost_bps / 1e4)
    return (gross - cost).rename("seasonal_timer")


def buy_hold(returns: pd.Series) -> pd.Series:
    return pd.Series(returns).astype(float).dropna().rename("buy_hold")


def summary(returns: pd.Series, rf: pd.Series | None = None,
            periods_per_year: int = MONTHS_PER_YEAR) -> dict:
    """Annualised Sharpe (excess-of-cash if ``rf`` given), CAGR, vol, max-drawdown, n."""
    r = pd.Series(returns).astype(float).dropna()
    if len(r) < 2:
        return {k: float("nan") for k in ("sharpe", "cagr", "vol_ann", "max_drawdown", "n")}
    ex = (r - pd.Series(rf).astype(float).reindex(r.index).fillna(0.0)) if rf is not None else r
    ex_mean, ex_std = ex.mean(), ex.std(ddof=1)
    eq = (1.0 + r).cumprod()
    dd = (eq / eq.cummax() - 1.0).min()
    years = len(r) / periods_per_year
    cagr = eq.iloc[-1] ** (1.0 / years) - 1.0 if eq.iloc[-1] > 0 else float("nan")
    return {
        "sharpe": float(ex_mean / ex_std * np.sqrt(periods_per_year)) if ex_std > 0 else float("nan"),
        "cagr": float(cagr),
        "vol_ann": float(r.std(ddof=1) * np.sqrt(periods_per_year)),
        "max_drawdown": float(dd),
        "n": int(len(r)),
    }


# --------------------------------------------------------------------------- #
# Third axis — UHN (the real retail holder) vs the HO=F splice, paired per season
# --------------------------------------------------------------------------- #
def uhn_vs_splice(uhn_close: pd.Series, ho_close: pd.Series,
                   start_year: int, end_year: int) -> pd.DataFrame:
    """Per-heating-season (Aug 31 -> next Feb 28/29) UHN return vs HO=F splice return, paired.

    UHN holds front-month HO=F futures directly for retail; the wrapper adds a management fee
    and any tracking slippage on top of the roll cost already embedded in the HO=F splice
    itself. ``gap = uhn_ret - ho_ret``: negative means the wrapper cost MORE than the raw
    futures roll. One row per season with >=50 trading days of overlap on both legs.
    """
    rows = []
    for y in range(start_year, end_year):
        lo = pd.Timestamp(f"{y}-08-31")
        hi = pd.Timestamp(f"{y + 1}-02-28")
        u = uhn_close[(uhn_close.index >= lo) & (uhn_close.index <= hi)]
        h = ho_close[(ho_close.index >= lo) & (ho_close.index <= hi)]
        if len(u) < 50 or len(h) < 50:
            continue
        u_ret = float(u.iloc[-1] / u.iloc[0] - 1.0)
        h_ret = float(h.iloc[-1] / h.iloc[0] - 1.0)
        rows.append({"season": f"{y}-{y + 1}", "uhn_ret": u_ret, "ho_splice_ret": h_ret,
                     "gap": u_ret - h_ret})
    return pd.DataFrame(rows)


def uhn_gap_stats(gap_df: pd.DataFrame) -> dict:
    x = gap_df["gap"].to_numpy(dtype=float)
    return {"n": len(x), "mean_gap": float(np.nanmean(x)) if len(x) else float("nan"),
            "t": one_sample_t(x)}


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(series: pd.Series, heat_months: list[int], off_months: list[int]) -> float:
    """Run the headline heating-vs-off-season Welch split on a synthetic monthly-return world."""
    g = group_welch(series, heat_months, off_months)
    return g["t"]
