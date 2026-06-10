"""The book — a post-earnings-announcement-drift (PEAD) momentum book on the cross-section of names.

The signal is each name's **most-recent earnings surprise**, carried forward for ``holding_days`` after
the announcement and then dropped. Each day:

  1. **Signal** — for every name, look up its latest earnings event *on or before* yesterday; if that
     event is within the last ``holding_days`` trading days, the name's signal is its surprise ``s``,
     else 0. (Lagged one day → strictly causal: the book only ever trades on a surprise it could already
     have read.)
  2. **Dollar-neutral weights** — demean the live signals across the cross-section and normalise so the
     long and short legs net to zero and gross exposure sums to 1. Long the most-positive surprises,
     short the most-negative — riding the drift, not fading it.
  3. **Hold to the horizon** — a name stays in the book until its surprise ages past ``holding_days`` or
     a fresh event overwrites it, so turnover is set by the earnings calendar, not a daily reshuffle.

The believer's claim (Ball-Brown 1968; Bernard-Thomas 1989): prices *under-react* to the earnings
surprise, so the surprise keeps predicting return for weeks. The whole tradability question is whether
that slow drift clears the cost of rolling the book as earnings land.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def surprise_signal(panel: pd.DataFrame, events: pd.DataFrame, holding_days: int = 50) -> pd.DataFrame:
    """A ``dates × ticker`` frame of the live surprise signal: each name carries its most-recent earnings
    surprise for ``holding_days`` trading days after the announcement, then 0. Not yet weighted; this is
    the raw, forward-filled SUE per name, aligned to the panel's trading calendar.

    Built causally: an event dated ``d`` becomes visible from day ``d`` onward (the surprise is public
    once reported), and the per-day weighting in :func:`pead_weights` lags it one more day before trading.
    """
    sig = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
    idx = panel.index
    pos = pd.Series(np.arange(len(idx)), index=idx)            # trading-day ordinal for each date
    for _, row in events.iterrows():
        tk = row["ticker"]
        if tk not in sig.columns:
            continue
        # snap the event to the first trading day on/after its calendar date
        loc = idx.searchsorted(pd.Timestamp(row["date"]), side="left")
        if loc >= len(idx):
            continue
        hi = min(loc + holding_days, len(idx))
        sig.iloc[loc:hi, sig.columns.get_loc(tk)] = row["surprise"]
    return sig


def pead_weights(panel: pd.DataFrame, events: pd.DataFrame, holding_days: int = 50) -> pd.DataFrame:
    """Dollar-neutral PEAD weights: the cross-sectionally demeaned live surprise, normalised so gross
    exposure is 1 each day, **lagged one day** so the book is tradable. Long positive-surprise names,
    short negative-surprise names."""
    sig = surprise_signal(panel, events, holding_days=holding_days)
    x = sig.sub(sig.mean(axis=1), axis=0)                      # relative to the live cross-section
    gross = x.abs().sum(axis=1).replace(0.0, np.nan)
    w = x.div(gross, axis=0).fillna(0.0)
    return w.shift(1).fillna(0.0)


def book_returns(panel: pd.DataFrame, events: pd.DataFrame, holding_days: int = 50,
                 cost_bps: float = 5.0) -> pd.Series:
    """Net daily return of the dollar-neutral PEAD book: weights applied to next-day returns, minus
    turnover cost (``cost_bps`` per unit traded). Turnover is paced by the earnings calendar (names enter
    on a surprise and leave when it ages out), so the cost term is far gentler than a daily-rebalanced
    book — but the drift it harvests is also small."""
    w = pead_weights(panel, events, holding_days=holding_days)
    gross = (w.shift(1) * panel).sum(axis=1)
    cost = (cost_bps * 1e-4) * w.diff().abs().sum(axis=1)
    return (gross - cost).rename("pead")


def turnover(panel: pd.DataFrame, events: pd.DataFrame, holding_days: int = 50) -> float:
    """Average daily one-way turnover (Σ|Δw|) of the PEAD book — small relative to a daily-rebalanced
    book, because positions only change as earnings land and age out."""
    w = pead_weights(panel, events, holding_days=holding_days)
    return float(w.diff().abs().sum(axis=1).mean())


def summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS) -> dict:
    """Annualised Sharpe, CAGR, vol, max-drawdown, Calmar, skew for a daily return series."""
    r = pd.Series(returns).astype(float).dropna()
    if len(r) < 2:
        return {k: np.nan for k in ("sharpe", "cagr", "vol_ann", "max_drawdown", "calmar", "skew", "n_days")}
    mean, std = r.mean(), r.std(ddof=1)
    eq = (1.0 + r).cumprod()
    dd = (eq / eq.cummax() - 1.0).min()
    years = len(r) / periods_per_year
    cagr = eq.iloc[-1] ** (1.0 / years) - 1.0 if eq.iloc[-1] > 0 else np.nan
    return {"sharpe": float(mean / std * np.sqrt(periods_per_year)) if std > 0 else np.nan,
            "cagr": float(cagr), "vol_ann": float(std * np.sqrt(periods_per_year)),
            "max_drawdown": float(dd), "calmar": float(cagr / abs(dd)) if dd < 0 else np.nan,
            "skew": float(r.skew()), "n_days": int(len(r))}
