"""Backtest: turn a (trigger, exit, cost) triple into an honest equity curve.

Event-driven and single-position by default: we enter at the close of a trigger
day, hold until the exit rule fires, sit in cash in between, and never stack
positions. Capital is on full size per trade (no leverage unless you add
financing), so the headline numbers are what a price-taker would actually live
through — idle cash and all.

The cost model has a twist specific to falling knives: a separate
``panic_slippage_bps`` charged only on the *entry*, because you are buying into a
-3% day when spreads gape and liquidity thins. This is exactly the cost most
buy-the-dip backtests forget, and (unlike the overnight strategy) it — not
turnover — is the likely killer here.

Nothing in this module decides whether a result is *real*: a positive Sharpe from
one cherry-picked (trigger, exit) pair is data-mining until it survives
:mod:`falling_knife.robustness`.
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

    Round-trip cost ≈ ``2*(half_spread + commission + slippage) + panic_slippage
    + financing*holding_days``. Defaults are a deliberately *optimistic* retail
    scenario for a liquid ETF; bump them to stress the result.
    """

    half_spread_bps: float = 1.0       # half the bid/ask, paid each side
    commission_bps: float = 0.5        # broker commission per side
    slippage_bps: float = 1.0          # ordinary execution slippage per side
    panic_slippage_bps: float = 5.0    # EXTRA, entry only: buying into the crash
    financing_bps_per_day: float = 0.0 # only if levered / CFD swap; 0 for cash

    def entry_cost_frac(self) -> float:
        return (self.half_spread_bps + self.commission_bps + self.slippage_bps
                + self.panic_slippage_bps) / 1e4

    def exit_cost_frac(self) -> float:
        return (self.half_spread_bps + self.commission_bps + self.slippage_bps) / 1e4


@dataclass
class BacktestResult:
    trades: list[Trade]
    daily: pd.Series          # daily strategy returns (0 when flat)
    equity: pd.Series         # cumulative (1+daily).cumprod()
    stats: dict
    rule: ExitRule
    costs: CostModel


def _collect_trades(ohlc: pd.DataFrame, signal: pd.Series, rule: ExitRule) -> list[Trade]:
    """Resolve non-overlapping trades from fresh signals, in date order."""
    pos = np.flatnonzero(signal.reindex(ohlc.index).fillna(False).to_numpy())
    close = ohlc["Close"].to_numpy()
    dates = ohlc.index

    trades: list[Trade] = []
    busy_until = -1
    for e in pos:
        if e <= busy_until:
            continue  # still holding a position from an earlier event
        tr = resolve_trade(ohlc, e, entry_price=float(close[e]), rule=rule)
        if tr is None:
            continue
        trades.append(tr)
        busy_until = dates.get_loc(tr.exit_date)
    return trades


def run(
    ohlc: pd.DataFrame,
    signal: pd.Series,
    rule: ExitRule,
    costs: CostModel = CostModel(),
) -> BacktestResult:
    """Run the event-driven backtest and compute performance stats.

    ``signal`` should be debounced into fresh events
    (:func:`falling_knife.triggers.first_crossings`); ``run`` additionally
    enforces non-overlap so you never double-count a held position.
    """
    close = ohlc["Close"].to_numpy()
    dates = ohlc.index
    pos_of = {d: i for i, d in enumerate(dates)}

    trades = _collect_trades(ohlc, signal, rule)

    daily = np.zeros(len(close))
    e_cost, x_cost = costs.entry_cost_frac(), costs.exit_cost_frac()
    fin = costs.financing_bps_per_day / 1e4

    for tr in trades:
        e = pos_of[tr.entry_date]
        x = pos_of[tr.exit_date]
        # Days e+1 .. x-1: ordinary close-to-close market exposure.
        for d in range(e + 1, x):
            daily[d] += close[d] / close[d - 1] - 1.0 - fin
        # Exit day: realised at the actual exit price (close, target or stop).
        daily[x] += tr.exit_price / close[x - 1] - 1.0 - fin
        # Transaction costs booked on the entry's first held day and the exit day.
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
        gross = tr.ret
        net = gross - costs.entry_cost_frac() - costs.exit_cost_frac() \
            - costs.financing_bps_per_day / 1e4 * tr.holding_days
        net_rets.append(net)
    net_rets = np.array(net_rets) if net_rets else np.array([np.nan])

    reasons = pd.Series([t.reason for t in trades]).value_counts().to_dict() if trades else {}
    exposure = float((daily != 0.0).mean())

    return {
        "n_trades": len(trades),
        # Days actually in the market — the observation count behind `sharpe`
        # (computed on active days only). Deflated-Sharpe checks must use THIS,
        # not the calendar length: an event strategy that is flat 93% of the
        # time has far fewer real observations than the sample has days.
        "active_days": int(len(active)),
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


# --------------------------------------------------------------------------- #
# Cost analysis: the same lens as the overnight study, adapted to low turnover.
# --------------------------------------------------------------------------- #

def cost_sweep(
    ohlc: pd.DataFrame,
    signal: pd.Series,
    rule: ExitRule,
    panic_bps_grid=(0, 5, 10, 20, 40),
    base: CostModel = CostModel(),
) -> pd.DataFrame:
    """Net performance as the entry panic-slippage rises.

    Turnover here is low (a handful of trades a year), so ordinary spread is rarely
    the killer — the entry slippage during the crash is. This sweep isolates it.
    """
    rows = []
    for p in panic_bps_grid:
        c = CostModel(base.half_spread_bps, base.commission_bps, base.slippage_bps,
                      float(p), base.financing_bps_per_day)
        r = run(ohlc, signal, rule, c)
        rows.append({"panic_slippage_bps": p, "cagr": r.stats["cagr"],
                     "sharpe": r.stats["sharpe"], "total_return": r.stats["total_return"],
                     "max_drawdown": r.stats["max_drawdown"], "n_trades": r.stats["n_trades"]})
    return pd.DataFrame(rows).set_index("panic_slippage_bps")


def family_scan(
    ohlc: pd.DataFrame,
    returns: pd.DataFrame,
    trigger_signals: dict[str, pd.Series],
    exit_grid: list[ExitRule],
    costs: CostModel = CostModel(),
    cooldown: int = 20,
) -> pd.DataFrame:
    """Backtest every (trigger x exit) combination — the data-mining surface.

    Returns one row per combination with the headline stats. This table is
    *expected* to contain a few great-looking cells purely by chance; that is the
    point, and exactly what :mod:`falling_knife.robustness` exists to discount.
    Do not read the best row as "the strategy".
    """
    from .triggers import first_crossings

    rows = []
    for tname, raw_sig in trigger_signals.items():
        for rule in exit_grid:
            cd = max(cooldown, rule.max_hold)
            sig = first_crossings(raw_sig, cooldown=cd)
            res = run(ohlc, sig, rule, costs)
            s = res.stats
            rows.append({
                "trigger": tname,
                "exit": rule.label(),
                "n_trades": s["n_trades"],
                "active_days": s["active_days"],
                "cagr": s["cagr"],
                "sharpe": s["sharpe"],
                "total_return": s["total_return"],
                "max_drawdown": s["max_drawdown"],
                "win_rate": s["win_rate"],
                "avg_hold_days": s["avg_hold_days"],
            })
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
