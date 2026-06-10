"""The heart — what financing a levered position *really* costs.

Two cost models for the SAME exposure path, and the gap between them is the whole study:

  * ``mode="idealized"`` — the flattering accounting most levered-timing backtests use (often
    silently): you only pay financing on the part of exposure *above* 100% — as if the first
    unit of leverage were free. Dividends ignored.
  * ``mode="honest"`` — how a real margin/CFD/futures position actually funds: the **entire
    notional** is financed at the short rate + a broker markup, every day it's held; a long
    index position is credited a dividend yield; idle cash earns the short rate. (For a cash
    equity account the financing is an *opportunity* cost of the capital tied up — same sign,
    same place.)

`net_returns` returns the daily net return of holding ``exposure`` (decided at the close, applied
to the next day's index move), so two strategies are compared on identical exposure with only the
cost model changing. The difference in CAGR is the broker's **house edge** on your leverage.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def net_returns(exposure: pd.Series, price: pd.Series, short_rate: pd.Series,
                mode: str = "honest", markup: float = 0.025, div_yield: float = 0.018,
                spread_bps: float = 2.0, cash_earns: bool = True) -> pd.Series:
    """Daily net return of holding ``exposure`` in the index, under the chosen cost model.

    ``mode="honest"`` finances the FULL notional (``exposure``) at ``short_rate + markup``, credits
    a ``div_yield`` on the long position, charges ``spread_bps`` per unit of turnover, and (if
    ``cash_earns``) pays ``short_rate`` on the un-invested fraction. ``mode="idealized"`` finances
    only ``max(exposure − 1, 0)`` and ignores dividends — the optimistic accounting that flatters
    levered backtests.
    """
    ret = price.pct_change()
    e = exposure.reindex(price.index).shift(1).clip(lower=0).fillna(0.0)   # decided at prior close
    turn = e.diff().abs().fillna(0.0)
    rf = short_rate.reindex(price.index).ffill().fillna(0.0) / TRADING_DAYS
    if mode == "idealized":
        fin = (e - 1).clip(lower=0) * (rf + markup / TRADING_DAYS)
        div = 0.0
        cash = 0.0
    elif mode == "honest":
        fin = e * (rf + markup / TRADING_DAYS)            # full notional financed
        div = e * div_yield / TRADING_DAYS                # dividends credited on the long
        cash = (1 - e).clip(lower=0) * rf if cash_earns else 0.0
    else:
        raise ValueError("mode must be 'idealized' or 'honest'")
    spread = turn * spread_bps * 1e-4
    return (e * ret + div - fin - spread + cash).dropna().rename(f"net_{mode}")


def buy_and_hold(price: pd.Series, div_yield: float = 0.018) -> pd.Series:
    """A plain, un-levered total-return index hold (the honest benchmark): price return + dividends."""
    return (price.pct_change() + div_yield / TRADING_DAYS).dropna().rename("buy_and_hold")


def house_edge(exposure: pd.Series, price: pd.Series, short_rate: pd.Series, **kw) -> dict:
    """The financing gap: idealized CAGR − honest CAGR for the same exposure path.

    Returns the two CAGRs and their difference (the annualised cost the optimistic accounting
    hides — the broker's *house edge* on your leverage).
    """
    from .strategy import summary
    ideal = summary(net_returns(exposure, price, short_rate, mode="idealized", **kw))
    hon = summary(net_returns(exposure, price, short_rate, mode="honest", **kw))
    return {"cagr_idealized": ideal["cagr"], "cagr_honest": hon["cagr"],
            "house_edge_ann": float(ideal["cagr"] - hon["cagr"]),
            "avg_exposure": float(exposure.reindex(price.index).shift(1).clip(lower=0).fillna(0).mean())}
