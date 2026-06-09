"""The investable books — and the cost that actually decides this one: **shorting the wild leg**.

Three portfolios come out of the vol sort, and the gap between them is the whole story:

    * **Low-vol long-only** — just hold the calm decile. Cheap, sticky, no shorting. This is what a
      real min-vol product (USMV, SPLV) does.
    * **High-vol** — hold the wild decile. The thing you'd *short* to harvest the anomaly cleanly.
    * **Naive long-short** — dollar-neutral ``low − high``. The headline "anomaly portfolio" — and the
      one that forces you to borrow the high-vol names, which are exactly the small, lottery-like,
      hard-to-borrow stocks where the borrow fee bites hardest (D'Avolio 2002; Beneish-Lee-Nichols 2015).

So the cost model here has **two** knobs, not one. ``cost_bps`` is the ordinary rebalancing slippage
charged on turnover (small — vol deciles are sticky). ``borrow_bps_ann`` is the annual stock-loan fee
on the *short* notional, charged only against the high-vol leg — the knob the tradability verdict
turns on. :func:`borrow_sweep` traces the long-short's net Sharpe as that fee climbs from "free" to
the hundreds-of-bps a real desk pays to borrow lottery stocks.

Sharpe is reported throughout; it is scale-invariant, so a leg's risk-adjusted quality doesn't depend
on how it's notionally sized. The *beta* honesty — that the dollar-neutral book is structurally short
the market, and that the long-only book is just low-beta — is the job of :mod:`decompose`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .sort import quantile_portfolios, turnover_ann

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


def _borrow_drag(high_leg: pd.Series, borrow_bps_ann: float) -> pd.Series:
    """Daily stock-loan fee on a unit short notional in the high-vol leg (annual bps / 252)."""
    return pd.Series(borrow_bps_ann * 1e-4 / TRADING_DAYS_PER_YEAR, index=high_leg.index)


def build(
    panel: pd.DataFrame,
    market: pd.Series | None = None,
    lookback: int = 126,
    rebal: int = 21,
    decile_frac: float = 0.1,
    cost_bps: float = 1.0,
    borrow_bps_ann: float = 0.0,
) -> dict:
    """Form the three books, charge realistic costs, and return their daily **net** return streams.

    Turnover slippage (``cost_bps`` per unit traded) is charged on every leg; the annual stock-loan fee
    (``borrow_bps_ann``) is charged only on the *short* (high-vol) notional inside the long-short. The
    ``market`` benchmark defaults to the equal-weight universe the sort came from.
    """
    book = quantile_portfolios(panel, lookback=lookback, rebal=rebal, decile_frac=decile_frac)
    mkt = (market.reindex(book.low.index) if market is not None else book.market).rename("market")

    cost_low = turnover_ann(book.w_low, rebal) * cost_bps * 1e-4 / TRADING_DAYS_PER_YEAR
    cost_high = turnover_ann(book.w_high, rebal) * cost_bps * 1e-4 / TRADING_DAYS_PER_YEAR

    low_net = (book.low - cost_low).rename("low_only")
    high_net = (book.high + cost_high)                       # shorting it: cost adds to the leg you pay
    borrow = _borrow_drag(book.high, borrow_bps_ann)
    long_short_net = (book.low - cost_low - book.high - cost_high - borrow).rename("long_short")

    return {
        "market": mkt,
        "low_only": low_net,
        "high": book.high.rename("high"),
        "long_short": long_short_net,
        "raw_low": book.low,
        "raw_high": book.high,
        "w_low": book.w_low,
        "w_high": book.w_high,
        "rebalances": book.rebalances,
        "turnover_low": turnover_ann(book.w_low, rebal),
        "turnover_high": turnover_ann(book.w_high, rebal),
    }


def compare(
    panel: pd.DataFrame,
    market: pd.Series | None = None,
    cost_bps: float = 1.0,
    borrow_bps_ann: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    **build_kw,
) -> dict:
    """Equal-weight market vs low-vol long-only vs naive long-short — summaries, gross and net.

    The headline reads are: does the **low-vol long-only** beat the market on Sharpe (the defensive
    tilt), and does the **dollar-neutral long-short** stand up once you pay to borrow the high-vol leg.
    Both questions get their honest answer in :mod:`decompose` (how much is just low beta / a short-leg
    artefact); this function lays out the raw scoreboard.
    """
    b = build(panel, market=market, cost_bps=cost_bps, borrow_bps_ann=borrow_bps_ann, **build_kw)
    s_mkt = summary(b["market"], periods_per_year)
    s_low = summary(b["low_only"], periods_per_year)
    s_ls = summary(b["long_short"], periods_per_year)
    return {
        "market": s_mkt,
        "low_only": s_low,
        "long_short": s_ls,
        "low_minus_market_sharpe": float(s_low["sharpe"] - s_mkt["sharpe"]),
        "turnover_low": float(b["turnover_low"]),
        "turnover_high": float(b["turnover_high"]),
        "rebalances": int(b["rebalances"]),
        "borrow_bps_ann": float(borrow_bps_ann),
        "n_days": int(s_low["n_days"]),
    }


def borrow_sweep(
    panel: pd.DataFrame,
    market: pd.Series | None = None,
    borrow_grid_bps=(0, 50, 100, 200, 400, 800),
    cost_bps: float = 1.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    **build_kw,
) -> pd.DataFrame:
    """Long-short net Sharpe and return as the **annual borrow fee on the short leg** climbs.

    This is the tradability argument made concrete: the anomaly's dollar-neutral form is only as good
    as your ability to short the high-vol decile cheaply, and those names — small, volatile,
    lottery-like — are precisely the expensive-to-borrow ones. The edge that looks healthy at a 0-bp
    "free short" assumption erodes as the fee approaches what those borrows actually cost.
    """
    rows = {}
    for fee in borrow_grid_bps:
        c = compare(panel, market=market, cost_bps=cost_bps, borrow_bps_ann=fee,
                    periods_per_year=periods_per_year, **build_kw)
        rows[fee] = {
            "long_short_sharpe": c["long_short"]["sharpe"],
            "long_short_ann_return": c["long_short"]["ann_return"],
        }
    out = pd.DataFrame(rows).T
    out.index.name = "borrow_bps_ann"
    return out


def cost_sweep(
    panel: pd.DataFrame,
    market: pd.Series | None = None,
    roundtrip_bps=(0, 1, 2, 5, 10, 20),
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
    **build_kw,
) -> pd.DataFrame:
    """Low-vol long-only Sharpe vs the market as rebalancing slippage rises — the *cheap* leg's check.

    The long-only book is sticky (vol deciles barely churn), so unlike the short-leg borrow this knob
    barely moves the result — the point being that what threatens this anomaly is the *short* leg, not
    ordinary turnover.
    """
    rows = {}
    for c in roundtrip_bps:
        cmp = compare(panel, market=market, cost_bps=c, periods_per_year=periods_per_year, **build_kw)
        rows[c] = {
            "market_sharpe": cmp["market"]["sharpe"],
            "low_only_sharpe": cmp["low_only"]["sharpe"],
            "low_minus_market": cmp["low_minus_market_sharpe"],
        }
    out = pd.DataFrame(rows).T
    out.index.name = "cost_bps"
    return out
