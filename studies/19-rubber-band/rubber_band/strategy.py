"""The investable book — and the cost that decides this one: **it trades every single day.**

IBS is a one-day signal, so any book built on it churns its whole position daily. That makes the cost
model, not the signal, the protagonist. Two constructions:

    * **Single-asset timing** — on one instrument, take a position ``w_t = 1 − 2·IBS_{t−1}`` (long when
      yesterday closed near its low, short when near its high), earning today's close-to-close return.
      Run across a basket, it is the cleanest read on the raw edge.
    * **Cross-sectional dollar-neutral** — across the basket, long the lowest-IBS names and short the
      highest, dollar-neutral. This is the §4.4 construction; it nets out the basket's common move.

``cost_bps`` is charged on turnover (here ~the full book, every day), so the break-even cost — the
slippage at which the net edge hits zero — is *low*, and the whole tradability question is whether it
clears a real bid-ask spread. :func:`cost_sweep` makes that concrete.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .ibs import ibs

TRADING_DAYS_PER_YEAR = 252


def summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> dict:
    """Headline stats for a return stream: Sharpe is the number that matters; maxDD the one you feel."""
    r = pd.Series(returns).astype(float).dropna()
    mean, std = r.mean(), r.std(ddof=1)
    equity = (1.0 + r).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    years = len(r) / periods_per_year
    cagr = equity.iloc[-1] ** (1.0 / years) - 1.0 if years > 0 and equity.iloc[-1] > 0 else np.nan
    return {
        "sharpe": float(mean / std * np.sqrt(periods_per_year)) if std > 0 else np.nan,
        "ann_return": float(mean * periods_per_year),
        "vol_ann": float(std * np.sqrt(periods_per_year)),
        "cagr": float(cagr),
        "max_drawdown": float(drawdown.min()),
        "n_days": int(len(r)),
    }


def timing_weights(ohlc: pd.DataFrame) -> pd.Series:
    """Single-asset position ``w_t = 1 − 2·IBS_{t−1}`` ∈ [−1, +1] — long after a low close, short a high.

    The weight is set from yesterday's bar (``shift(1)``), so it earns today's return with no
    look-ahead: long (+1) when yesterday closed on its low, short (−1) when on its high, scaled linearly
    in between.
    """
    return (1.0 - 2.0 * ibs(ohlc)).shift(1).rename("weight")


def timing_returns(ohlc: pd.DataFrame, cost_bps: float = 0.0) -> pd.Series:
    """The single-asset IBS timing stream, net of ``cost_bps`` charged on each unit of turnover."""
    w = timing_weights(ohlc)
    r = ohlc["Close"].pct_change()
    gross = w * r
    cost = (cost_bps * 1e-4) * w.diff().abs()
    return (gross - cost).dropna().rename("timing")


def timing_panel_returns(basket: dict, cost_bps: float = 0.0) -> pd.Series:
    """Equal-weight average of the single-asset IBS timing streams across a basket of OHLC frames.

    Diversifying the one-day signal over many instruments smooths it without changing its nature; the
    average is the cleanest estimate of the raw IBS edge and the series the inference runs on.
    """
    streams = {tk: timing_returns(ohlc, cost_bps=cost_bps) for tk, ohlc in basket.items()}
    return pd.DataFrame(streams).mean(axis=1).dropna().rename("timing_ew")


def cross_sectional_returns(basket: dict, decile_frac: float = 0.2, cost_bps: float = 0.0) -> pd.Series:
    """Dollar-neutral §4.4 book: long the lowest-IBS names, short the highest, rebalanced daily.

    On each day the names are ranked by yesterday's IBS; the bottom and top ``decile_frac`` are equal-
    weighted long and short. Returns the net daily stream. Daily reranking ⇒ near-full turnover, the
    cost the verdict turns on.
    """
    ibs_df = pd.DataFrame({tk: ibs(ohlc) for tk, ohlc in basket.items()}).shift(1)
    ret_df = pd.DataFrame({tk: ohlc["Close"].pct_change() for tk, ohlc in basket.items()})
    common = ibs_df.dropna(how="all").index.intersection(ret_df.index)
    ibs_df, ret_df = ibs_df.loc[common], ret_df.loc[common]

    out = []
    prev_w = None
    cost_acc = []
    for day in common:
        row = ibs_df.loc[day].dropna()
        if len(row) < 4:
            out.append(np.nan); cost_acc.append(0.0); continue
        k = max(1, int(len(row) * decile_frac))
        order = row.sort_values()
        longs, shorts = order.index[:k], order.index[-k:]
        w = pd.Series(0.0, index=ret_df.columns)
        w[longs] = 0.5 / k
        w[shorts] = -0.5 / k
        out.append(float((w * ret_df.loc[day]).sum()))
        if prev_w is not None:
            cost_acc.append((cost_bps * 1e-4) * (w - prev_w).abs().sum())
        else:
            cost_acc.append(0.0)
        prev_w = w
    gross = pd.Series(out, index=common)
    cost = pd.Series(cost_acc, index=common)
    return (gross - cost).dropna().rename("xsec")


def turnover_ann(ohlc: pd.DataFrame) -> float:
    """Annualised turnover of the single-asset timing book — ~daily, the point being it's enormous."""
    w = timing_weights(ohlc)
    return float(w.diff().abs().mean() * TRADING_DAYS_PER_YEAR)


def cost_sweep(basket: dict, roundtrip_bps=(0, 1, 2, 3, 5, 10), periods_per_year=TRADING_DAYS_PER_YEAR):
    """Net Sharpe of the basket IBS timing book as cost per unit traded rises — and where it dies.

    Because the book turns over ~daily, the break-even cost is small: this table is the tradability
    argument, showing the edge crossing zero somewhere inside the range of a real ETF bid-ask spread.
    """
    rows = {}
    for c in roundtrip_bps:
        s = summary(timing_panel_returns(basket, cost_bps=c), periods_per_year)
        rows[c] = {"sharpe": s["sharpe"], "ann_return": s["ann_return"]}
    out = pd.DataFrame(rows).T
    out.index.name = "cost_bps"
    return out
