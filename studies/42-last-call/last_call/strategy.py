"""The turn-of-the-month window, the per-day comparison, and the tradable rule.

The effect (Lakonishok & Smidt 1988; McConnell & Xu 2008): daily returns concentrate in a window
around the month boundary — conventionally the **last trading day** of a month through the **first
three** of the next (the [-1, +3] window). The honest tests: (1) do TOM days really out-earn the rest,
and (2) does *trading* only the window beat just holding the index, after costs?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def tom_mask(index: pd.DatetimeIndex, last: int = 1, first: int = 3) -> pd.Series:
    """Boolean mask of turn-of-the-month days for a daily index.

    Marks the final ``last`` trading day(s) of each month and the first ``first`` of the next — the
    classic [-1, +3] window with the defaults. Uses trading-day position within the calendar month, so
    it adapts to holidays automatically.
    """
    idx = pd.DatetimeIndex(index)
    ym = idx.to_period("M")
    pos_start = pd.Series(np.arange(len(idx)), index=idx).groupby(ym).cumcount() + 1
    pos_end = pd.Series(np.arange(len(idx)), index=idx).groupby(ym).cumcount(ascending=False) + 1
    mask = (pos_start <= first) | (pos_end <= last)
    return pd.Series(mask.values, index=idx, name="tom")


def tom_vs_rest(returns: pd.Series, last: int = 1, first: int = 3) -> dict:
    """Mean daily return (bp) on turn-of-the-month days vs the rest, plus counts and a t-stat."""
    r = pd.Series(returns).astype(float).dropna()
    m = tom_mask(r.index, last, first)
    tom, rest = r[m.values], r[~m.values]
    # Welch t for the difference in means
    s = np.sqrt(tom.var(ddof=1) / len(tom) + rest.var(ddof=1) / len(rest))
    t = (tom.mean() - rest.mean()) / s if s > 0 else np.nan
    return {
        "tom_bp": float(tom.mean() * 1e4),
        "rest_bp": float(rest.mean() * 1e4),
        "all_bp": float(r.mean() * 1e4),
        "n_tom": int(len(tom)),
        "n_rest": int(len(rest)),
        "tom_share": float(m.mean()),
        "welch_t": float(t),
    }


def _per_period_rf(r: pd.Series, rf, periods_per_year: int) -> pd.Series:
    """Per-period cash return under the quantlab convention: scalar ``rf`` = annualised rate,
    Series = per-period return, aligned (ffilled) on the index."""
    if np.ndim(rf) == 0:
        return pd.Series(float(rf) / periods_per_year, index=r.index)
    return pd.Series(rf).astype(float).reindex(r.index).ffill().fillna(0.0)


def tom_returns(
    returns: pd.Series,
    cost_bps: float = 2.0,
    last: int = 1,
    first: int = 3,
    rf: float | pd.Series = 0.0,
) -> pd.Series:
    """The tradable rule: hold the index on exactly the turn-of-the-month days, cash otherwise.

    The position **is** the mask — no lag. The [-1, +3] window is a pure function of the exchange
    calendar, known before the month begins, so the position for day *t* is put on at the close of
    day *t−1*: holding exactly the masked days is tradable with no information lag (nothing about
    day *t* is used to decide day *t*, unlike a signal computed from day-*t* data). Lagging the mask
    — a previous version did — silently trades the *wrong* window: it misses day −1 (the single
    richest day of the effect) and holds day +4 instead.

    ``rf`` is what the cash leg earns on the ~81% of days out of the market (scalar annualised rate
    or a per-period series, quantlab convention; default 0 keeps the cash-earns-nothing book). Each
    entry/exit pays ``cost_bps``. Returns the net daily return series.
    """
    r = pd.Series(returns).astype(float).dropna()
    m = tom_mask(r.index, last, first)
    pos = m.astype(float)
    cash = _per_period_rf(r, rf, TRADING_DAYS)
    gross = r.where(m, cash)
    cost = (cost_bps * 1e-4) * pos.diff().abs().fillna(0.0)
    return (gross - cost).rename("tom")


def buy_hold(returns: pd.Series) -> pd.Series:
    return pd.Series(returns).astype(float).dropna().rename("buy_hold")


def summary(
    returns: pd.Series, periods_per_year: int = TRADING_DAYS, rf: float | pd.Series = 0.0
) -> dict:
    """Annualised Sharpe, CAGR, vol, max-drawdown, time-in-market for a daily return series.

    ``rf`` (scalar annualised, or per-period series — quantlab convention) makes the **Sharpe an
    excess-of-cash Sharpe**; CAGR / vol / drawdown stay raw. A book that idles in cash 81% of the
    time must not be raced on raw Sharpe against an always-invested benchmark — either credit the
    cash leg and compare excess vs excess, or you are just measuring who held more T-bills.
    ``time_in_market`` counts non-zero days, which is only meaningful for the ``rf=0`` book.
    """
    r = pd.Series(returns).astype(float).dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("sharpe", "cagr", "vol_ann", "max_drawdown", "time_in_market", "n")}
    std = r.std(ddof=1)
    ex = r - _per_period_rf(r, rf, periods_per_year)
    ex_mean, ex_std = ex.mean(), ex.std(ddof=1)
    eq = (1.0 + r).cumprod()
    dd = (eq / eq.cummax() - 1.0).min()
    years = len(r) / periods_per_year
    cagr = eq.iloc[-1] ** (1.0 / years) - 1.0 if eq.iloc[-1] > 0 else np.nan
    return {
        "sharpe": float(ex_mean / ex_std * np.sqrt(periods_per_year)) if ex_std > 0 else np.nan,
        "cagr": float(cagr),
        "vol_ann": float(std * np.sqrt(periods_per_year)),
        "max_drawdown": float(dd),
        "time_in_market": float((r != 0).mean()),
        "n": int(len(r)),
    }


def tom_premium_change(returns: pd.Series, split: str = "2008", last: int = 1, first: int = 3) -> dict:
    """Welch-style test for the *decay* claim: did the TOM premium shrink across ``split``?

    For each half, the premium is d = mean(TOM) − mean(rest) with the Welch standard error
    se = sqrt(var_tom/n_tom + var_rest/n_rest); the change statistic is
    t = (d_pre − d_post) / sqrt(se_pre² + se_post²). Also returns each half's own Welch t, so
    "faded" can be stated as two numbers: the premium fell (t_change) and the modern half no longer
    clears zero (welch_t_post) — instead of a bare assertion on two point estimates.
    """
    r = pd.Series(returns).astype(float).dropna()
    halves = {"pre": r[r.index < pd.Timestamp(split)], "post": r[r.index >= pd.Timestamp(split)]}
    out: dict = {"split": split}
    d, se = {}, {}
    for k, s in halves.items():
        m = tom_mask(s.index, last, first)
        tom, rest = s[m.values], s[~m.values]
        d[k] = tom.mean() - rest.mean()
        se[k] = float(np.sqrt(tom.var(ddof=1) / len(tom) + rest.var(ddof=1) / len(rest)))
        out[f"premium_{k}_bp"] = float(d[k] * 1e4)
        out[f"welch_t_{k}"] = float(d[k] / se[k]) if se[k] > 0 else np.nan
        out[f"n_{k}"] = int(len(s))
    denom = float(np.sqrt(se["pre"] ** 2 + se["post"] ** 2))
    out["t_change"] = float((d["pre"] - d["post"]) / denom) if denom > 0 else np.nan
    return out
