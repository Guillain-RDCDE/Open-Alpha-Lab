"""Strategy + inference for Study 367 — Closed-End Fund Discount.

The pitch (steelmanned): closed-end funds trade away from NAV; a deep **discount** is a
coiled spring that mean-reverts, so buying the *widest-discount* funds and waiting pays an
excess return. We test it on a **proxy discount** (price relative to a per-fund benchmark;
true NAV is not on yfinance) with the desk's standard gauntlet:

  * **Signal.** Each month, rank funds by proxy-discount; go long the cheapest tercile
    (widest discount) and short the richest tercile (narrowest / premium). Hold one month,
    rebalance. The headline is the mean monthly **long-short** return, its Welch *t* vs
    zero, a **block-bootstrap** null on the monthly series, and the **win-rate** vs 50%.
  * **Execution lag.** The discount is known at the close of month *t*; the portfolio earns
    month *t+1* — one lag, applied once, no look-ahead.
  * **Costs.** A one-way cost in bps × the realised one-way turnover of each leg, charged
    against the gross spread (the long-short turns over both legs every month it rebalances).
  * **Synthetic control.** On a panel where the truth is known, ``edge = 0`` must stay below
    *t* = 2 and a large planted edge must light up — the machinery proof.

The decisive object is whether the long-short spread is statistically distinguishable from
zero on the *real* proxy tape, net of costs and named survivorship.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_MONTH = 21


# --------------------------------------------------------------------------- #
# Monthly panel construction (real tape)
# --------------------------------------------------------------------------- #
def monthly_panels(disc: pd.DataFrame, px: pd.DataFrame) -> dict:
    """Resample the daily proxy-discount panel and CEF prices to month-end and build the
    monthly **forward return** panel (price_{t+1}/price_t − 1) aligned to month *t*.

    Returns a dict with ``disc`` (month-end discount, funds in columns) and ``fret``
    (the return *earned over month t+1*, indexed at month *t* — already lagged once, so a
    sort on ``disc`` at row *t* is paired with the return you actually capture, no
    look-ahead).
    """
    dm = disc.resample("ME").last()
    pm = px.resample("ME").last()
    # return over the NEXT month, labelled at month t (one-lag convention)
    fret = pm.shift(-1) / pm - 1.0
    # align columns
    cols = [c for c in dm.columns if c in fret.columns]
    return {"disc": dm[cols], "fret": fret[cols]}


# --------------------------------------------------------------------------- #
# The discount sort -> long-short monthly return series
# --------------------------------------------------------------------------- #
def long_short_returns(disc: pd.DataFrame, fret: pd.DataFrame,
                       q: float = 1 / 3, min_names: int = 6) -> pd.Series:
    """Monthly long-short return: long the widest-discount (cheapest) tercile, short the
    narrowest (richest) tercile, equal-weighted within each leg.

    ``disc`` is the month-*t* proxy discount (lower = cheaper); ``fret`` is the month-*t+1*
    return already aligned to *t*. A month is used only if at least ``min_names`` funds have
    both a discount and a forward return that month. Returns a Series of monthly long-short
    returns indexed by month *t*.
    """
    out = {}
    for t in disc.index:
        d = disc.loc[t].dropna()
        r = fret.loc[t] if t in fret.index else pd.Series(dtype=float)
        common = d.index.intersection(r.dropna().index)
        if len(common) < min_names:
            continue
        d = d[common]
        r = r[common]
        k = max(1, int(round(len(common) * q)))
        order = d.sort_values()                 # ascending: cheapest (widest disc) first
        wide = order.index[:k]                  # long
        narrow = order.index[-k:]               # short
        out[t] = r[wide].mean() - r[narrow].mean()
    return pd.Series(out).sort_index()


def long_only_excess(disc: pd.DataFrame, fret: pd.DataFrame,
                     q: float = 1 / 3, min_names: int = 6) -> pd.Series:
    """Monthly return of the widest-discount tercile minus the equal-weight universe
    (the 'just buy the cheap ones' version, vs holding everything)."""
    out = {}
    for t in disc.index:
        d = disc.loc[t].dropna()
        r = fret.loc[t] if t in fret.index else pd.Series(dtype=float)
        common = d.index.intersection(r.dropna().index)
        if len(common) < min_names:
            continue
        d = d[common]; r = r[common]
        k = max(1, int(round(len(common) * q)))
        wide = d.sort_values().index[:k]
        out[t] = r[wide].mean() - r.mean()
    return pd.Series(out).sort_index()


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def welch_t_vs_zero(x: np.ndarray) -> float:
    """One-sample t of mean(x) vs 0 (the long-short spread's significance). NaN if <2."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    if se == 0:
        return float("nan")
    return float(x.mean() / se)


def block_bootstrap_p(x: np.ndarray, block: int = 6, n_boot: int = 20_000,
                      seed: int = 367) -> dict:
    """Circular block-bootstrap null for a monthly mean.

    Resamples the *sign-flipped-around-zero* series in blocks of ``block`` months to build
    the null distribution of the mean under "no edge" while preserving short-run
    autocorrelation, and returns the two-sided p-value P[|boot mean| >= |obs mean|]. The
    block bootstrap (not i.i.d.) respects the persistence in a discount-driven series.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 2:
        return {"mean": float("nan"), "p_value": float("nan")}
    obs = x.mean()
    centred = x - obs                              # null: mean zero
    rng = np.random.default_rng(seed)
    n_blocks = int(np.ceil(n / block))
    means = np.empty(n_boot)
    for i in range(n_boot):
        starts = rng.integers(0, n, size=n_blocks)
        idx = (starts[:, None] + np.arange(block)[None, :]) % n
        means[i] = centred[idx.ravel()[:n]].mean()
    p = float((np.abs(means) >= abs(obs)).mean())
    return {"mean": float(obs), "p_value": p}


def summarize(ls: pd.Series, ann: int = 12) -> dict:
    """Headline stats for a monthly long-short series: n months, mean, annualised mean,
    win-rate vs 50%, Welch t, block-bootstrap p, annualised Sharpe."""
    x = ls.dropna().values
    if len(x) < 2:
        return {"n": int(len(x)), "mean": float("nan"), "ann_mean": float("nan"),
                "win": float("nan"), "t": float("nan"), "p_boot": float("nan"),
                "sharpe": float("nan")}
    mean = float(x.mean())
    sd = float(x.std(ddof=1))
    bb = block_bootstrap_p(x)
    return {
        "n": int(len(x)),
        "mean": mean,
        "ann_mean": mean * ann,
        "win": float((x > 0).mean()),
        "t": welch_t_vs_zero(x),
        "p_boot": bb["p_value"],
        "sharpe": float(mean / sd * np.sqrt(ann)) if sd > 0 else float("nan"),
    }


def net_of_costs(ls: pd.Series, cost_bps: float = 20.0,
                 turnover: float = 2.0, ann: int = 12) -> dict:
    """Charge a one-way cost in bps × turnover per month against the gross spread.

    ``turnover`` defaults to 2.0 one-way legs/month (a long-short that fully rebalances both
    sleeves each month is the conservative worst case). Returns gross/net monthly & annual
    means and the implied break-even cost (the per-month cost that zeroes the spread)."""
    x = ls.dropna().values
    if len(x) < 2:
        return {"gross_mean": float("nan"), "net_mean": float("nan"),
                "cost_month": float("nan"), "breakeven_bps": float("nan")}
    gross = float(x.mean())
    cost_month = (cost_bps / 1e4) * turnover
    net = gross - cost_month
    breakeven = gross / turnover * 1e4 if turnover > 0 else float("nan")
    return {
        "gross_mean": gross,
        "net_mean": net,
        "ann_gross": gross * ann,
        "ann_net": net * ann,
        "cost_month": cost_month,
        "breakeven_bps": float(breakeven),
    }
