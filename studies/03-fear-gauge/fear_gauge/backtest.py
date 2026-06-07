"""Backtest: turn a (VIX trigger, exit, cost) triple into an honest equity curve.

Event-driven and single-position by default: enter at the close of a gauge-trigger
day, hold until the exit fires, sit in cash between trades, never stack positions.
Full size per trade, no leverage unless you add financing — so the numbers are what
a price-taker would actually live through.

The cost model carries the same twist as Study 02, only more so: a separate
``panic_slippage_bps`` charged on the *entry*, because a VIX≥30 / +30%-spike day is
exactly when S&P spreads gape and liquidity thins. And a reminder the spot index
hides — you cannot trade ``^GSPC``; the realistic curve is the ``SPY`` one with
these costs charged.

Nothing here decides whether a result is *real*. A pretty Sharpe from one
cherry-picked (trigger, exit) cell is data-mining until it survives
:mod:`fear_gauge.robustness` (random-day null, the price-drop control, the
clustering bootstrap and the window-selection test).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .exits import ExitRule, resolve_trade, Trade

TRADING_DAYS = 252


@dataclass(frozen=True)
class CostModel:
    """Per-side and per-day costs, in basis points (1 bp = 0.01%).

    Round-trip ≈ ``2*(half_spread + commission + slippage) + panic_slippage
    + financing*holding_days``. Defaults are an optimistic retail scenario for a
    liquid S&P ETF; bump them to stress the result.
    """

    half_spread_bps: float = 1.0
    commission_bps: float = 0.5
    slippage_bps: float = 1.0
    panic_slippage_bps: float = 5.0     # EXTRA, entry only: buying into the spike
    financing_bps_per_day: float = 0.0

    def entry_cost_frac(self) -> float:
        return (self.half_spread_bps + self.commission_bps + self.slippage_bps
                + self.panic_slippage_bps) / 1e4

    def exit_cost_frac(self) -> float:
        return (self.half_spread_bps + self.commission_bps + self.slippage_bps) / 1e4


@dataclass
class BacktestResult:
    trades: list[Trade]
    daily: pd.Series
    equity: pd.Series
    stats: dict
    rule: ExitRule
    costs: CostModel


def _collect_trades(market: pd.DataFrame, signal: pd.Series, rule: ExitRule) -> list[Trade]:
    """Resolve non-overlapping trades from fresh signals, in date order."""
    pos = np.flatnonzero(signal.reindex(market.index).fillna(False).to_numpy())
    close = market["Close"].to_numpy()
    dates = market.index

    trades: list[Trade] = []
    busy_until = -1
    for e in pos:
        if e <= busy_until:
            continue
        tr = resolve_trade(market, e, entry_price=float(close[e]), rule=rule)
        if tr is None:
            continue
        trades.append(tr)
        busy_until = dates.get_loc(tr.exit_date)
    return trades


def run(
    market: pd.DataFrame,
    signal: pd.Series,
    rule: ExitRule,
    costs: CostModel = CostModel(),
) -> BacktestResult:
    """Run the event-driven backtest and compute performance stats.

    ``signal`` should be a debounced gauge trigger
    (:func:`fear_gauge.triggers.first_crossings`); ``run`` additionally enforces
    non-overlap so a held position is never double-counted.
    """
    close = market["Close"].to_numpy()
    dates = market.index
    pos_of = {d: i for i, d in enumerate(dates)}

    trades = _collect_trades(market, signal, rule)

    daily = np.zeros(len(close))
    e_cost, x_cost = costs.entry_cost_frac(), costs.exit_cost_frac()
    fin = costs.financing_bps_per_day / 1e4

    for tr in trades:
        e = pos_of[tr.entry_date]
        x = pos_of[tr.exit_date]
        for d in range(e + 1, x):
            daily[d] += close[d] / close[d - 1] - 1.0 - fin
        daily[x] += tr.exit_price / close[x - 1] - 1.0 - fin
        daily[e + 1] -= e_cost
        daily[x] -= x_cost

    daily_s = pd.Series(daily, index=dates, name="strategy_ret")
    equity = (1.0 + daily_s).cumprod().rename("equity")
    stats = _performance(daily_s, equity, trades, costs)
    return BacktestResult(trades, daily_s, equity, stats, rule, costs)


def _performance(daily: pd.Series, equity: pd.Series, trades: list[Trade], costs: CostModel) -> dict:
    n_days = len(daily)
    years = n_days / TRADING_DAYS if n_days else np.nan
    total_return = float(equity.iloc[-1] - 1.0) if n_days else np.nan
    cagr = float(equity.iloc[-1] ** (1 / years) - 1.0) if years and years > 0 else np.nan

    active = daily[daily != 0.0]
    vol = float(active.std(ddof=1) * np.sqrt(TRADING_DAYS)) if len(active) > 1 else np.nan
    mean_d = float(active.mean()) if len(active) else np.nan
    sharpe = (mean_d / active.std(ddof=1) * np.sqrt(TRADING_DAYS)
              if len(active) > 1 and active.std(ddof=1) > 0 else np.nan)

    running_max = equity.cummax()
    max_dd = float((equity / running_max - 1.0).min()) if n_days else np.nan

    net_rets = []
    for tr in trades:
        net = tr.ret - costs.entry_cost_frac() - costs.exit_cost_frac() \
            - costs.financing_bps_per_day / 1e4 * tr.holding_days
        net_rets.append(net)
    net_rets = np.array(net_rets) if net_rets else np.array([np.nan])

    reasons = pd.Series([t.reason for t in trades]).value_counts().to_dict() if trades else {}
    exposure = float((daily != 0.0).mean())

    return {
        "n_trades": len(trades),
        "total_return": total_return,
        "cagr": cagr,
        "sharpe": float(sharpe) if sharpe == sharpe else np.nan,
        "ann_vol": vol,
        "max_drawdown": max_dd,
        "win_rate": float((net_rets > 0).mean()),
        "avg_trade_net": float(np.nanmean(net_rets)),
        "median_trade_net": float(np.nanmedian(net_rets)),
        "avg_hold_days": float(np.mean([t.holding_days for t in trades])) if trades else np.nan,
        "exposure": exposure,
        "exit_reasons": reasons,
        "years": years,
    }


def cost_sweep(
    market: pd.DataFrame,
    signal: pd.Series,
    rule: ExitRule,
    panic_bps_grid=(0, 5, 10, 20, 40),
    base: CostModel = CostModel(),
) -> pd.DataFrame:
    """Net performance as the entry panic-slippage rises.

    Turnover is low (a handful of trades a year), so ordinary spread rarely kills
    it — the entry slippage during the spike does. This sweep isolates it.
    """
    rows = []
    for p in panic_bps_grid:
        c = CostModel(base.half_spread_bps, base.commission_bps, base.slippage_bps,
                      float(p), base.financing_bps_per_day)
        r = run(market, signal, rule, c)
        rows.append({"panic_slippage_bps": p, "cagr": r.stats["cagr"],
                     "sharpe": r.stats["sharpe"], "total_return": r.stats["total_return"],
                     "max_drawdown": r.stats["max_drawdown"], "n_trades": r.stats["n_trades"]})
    return pd.DataFrame(rows).set_index("panic_slippage_bps")


def family_scan(
    market: pd.DataFrame,
    trigger_signals: dict[str, pd.Series],
    exit_grid: list[ExitRule],
    costs: CostModel = CostModel(),
    cooldown: int = 21,
) -> pd.DataFrame:
    """Backtest every (trigger x exit) combination — the data-mining surface.

    One row per combination with the headline stats. This table is *expected* to
    contain a few great-looking cells purely by chance; that is the point, and what
    :mod:`fear_gauge.robustness` exists to discount. Do not read the best row as
    "the strategy".
    """
    from .triggers import first_crossings

    rows = []
    for tname, raw_sig in trigger_signals.items():
        for rule in exit_grid:
            cd = max(cooldown, rule.max_hold)
            sig = first_crossings(raw_sig, cooldown=cd)
            res = run(market, sig, rule, costs)
            s = res.stats
            rows.append({
                "trigger": tname,
                "exit": rule.label(),
                "n_trades": s["n_trades"],
                "cagr": s["cagr"],
                "sharpe": s["sharpe"],
                "total_return": s["total_return"],
                "max_drawdown": s["max_drawdown"],
                "win_rate": s["win_rate"],
                "avg_hold_days": s["avg_hold_days"],
            })
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
