"""Backtest: turn the mentions into trades a *follower* could actually place — and
watch the micro-cap costs eat the pop.

The honest entry is the part the dashboards skip: you read the tweet *after* it's
posted, so you can't have the mention-day close. We buy at the **next session's
open**, hold a fixed number of sessions, and sell at the close — a price-taker who
saw the call when everyone else did. Each event is one trade; the sleeve holds them
equal-weight and lets them overlap (mentions cluster, so you're often in several at
once).

Two costs decide it, and both are brutal for the names a viral feed surfaces:

    * the **spread** — a $1–3 stock can quote 50–200 bps wide; you pay half of it
      twice. The defaults here are far higher than the liquid-ETF defaults in Studies
      02–03, on purpose.
    * **market impact** — these names trade thin, so any real size moves the print.
      :func:`capacity` reports, via a square-root impact model, the dollar size at
      which impact alone equals the edge — usually embarrassingly small.

Nothing here decides whether a result is *real*; that's :mod:`social_oracle.benchmark`
and :mod:`social_oracle.robustness`. This module decides whether anything *survives*
once a human pays to act on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass(frozen=True)
class CostModel:
    """Round-trip costs, in basis points (1 bp = 0.01%).

    Defaults are a *thin small-cap* scenario, not a liquid ETF: a wide spread and
    real slippage, because that's the universe a viral cashtag feed lives in. Bump
    them to stress; the whole point of beat 6 is how little it takes to kill the pop.
    """

    half_spread_bps: float = 25.0   # a $1-3 name can quote 50+ bps wide
    commission_bps: float = 1.0
    slippage_bps: float = 10.0

    def round_trip_frac(self) -> float:
        return 2.0 * (self.half_spread_bps + self.commission_bps + self.slippage_bps) / 1e4


@dataclass
class Trade:
    ticker: str
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_price: float
    exit_price: float
    holding_days: int
    gross_ret: float        # raw P&L you'd actually book holding the name
    abn_ret: float          # abnormal (name minus market) over the hold
    net_ret: float          # gross minus round-trip costs


@dataclass
class BacktestResult:
    trades: list[Trade]
    daily: pd.Series
    equity: pd.Series
    stats: dict
    hold_days: int
    costs: CostModel


def _market_car(frame: pd.DataFrame, e: int, hold: int) -> float:
    """Market cumulative return over the hold window, for abnormal attribution."""
    mkt = np.nan_to_num(frame["r_mkt"].to_numpy(), nan=0.0)
    return float(mkt[e + 1: e + hold + 1].sum())


def run(
    panel: dict[str, pd.DataFrame],
    events: pd.DataFrame,
    hold_days: int = 10,
    costs: CostModel = CostModel(),
) -> BacktestResult:
    """Run the event-sleeve backtest: buy next open, hold ``hold_days``, sell close.

    Returns trades, the equal-weight daily sleeve return, its equity curve, and a
    stats dict. The equity sleeve averages, each session, the daily return of every
    trade open that day — a diversified follower who sizes each call equally.
    """
    trades: list[Trade] = []
    # per-date accumulators for the equal-weight sleeve (union calendar)
    day_sum: dict[pd.Timestamp, float] = {}
    day_cnt: dict[pd.Timestamp, int] = {}
    rt = costs.round_trip_frac()

    for t, grp in events.groupby("ticker"):
        frame = panel.get(t)
        if frame is None:
            continue
        n = len(frame)
        op = frame["Open"].to_numpy()
        cl = frame["Close"].to_numpy()
        dates = frame.index
        for e in grp["entry_pos"].to_numpy():
            e = int(e)
            if e + 1 >= n or e + hold_days >= n:
                continue
            entry = op[e + 1]
            exit_ = cl[e + hold_days]
            gross = exit_ / entry - 1.0
            abn = gross - _market_car(frame, e, hold_days)
            trades.append(Trade(
                ticker=t, entry_date=dates[e + 1], exit_date=dates[e + hold_days],
                entry_price=float(entry), exit_price=float(exit_),
                holding_days=hold_days, gross_ret=float(gross),
                abn_ret=float(abn), net_ret=float(gross - rt),
            ))
            # spread the trade's P&L across its held sessions for the sleeve curve
            for j in range(e + 1, e + hold_days + 1):
                d = dates[j]
                if j == e + 1:
                    r = cl[j] / entry - 1.0 - rt / 2.0   # open->close, half the cost on entry
                elif j == e + hold_days:
                    r = cl[j] / cl[j - 1] - 1.0 - rt / 2.0  # ...other half on exit
                else:
                    r = cl[j] / cl[j - 1] - 1.0
                day_sum[d] = day_sum.get(d, 0.0) + r
                day_cnt[d] = day_cnt.get(d, 0) + 1

    if day_sum:
        idx = pd.DatetimeIndex(sorted(day_sum)).sort_values()
        daily = pd.Series([day_sum[d] / day_cnt[d] for d in idx], index=idx, name="sleeve_ret")
    else:
        daily = pd.Series(dtype=float, name="sleeve_ret")
    equity = (1.0 + daily).cumprod().rename("equity")
    stats = _performance(trades, daily, equity)
    return BacktestResult(trades, daily, equity, stats, hold_days, costs)


def _performance(trades: list[Trade], daily: pd.Series, equity: pd.Series) -> dict:
    if not trades:
        return {"n_trades": 0}
    gross = np.array([t.gross_ret for t in trades])
    net = np.array([t.net_ret for t in trades])
    abn = np.array([t.abn_ret for t in trades])
    se = net.std(ddof=1) / np.sqrt(len(net)) if len(net) > 1 else np.nan

    max_dd = np.nan
    sharpe = np.nan
    if len(daily) > 1:
        running_max = equity.cummax()
        max_dd = float((equity / running_max - 1.0).min())
        sd = daily.std(ddof=1)
        sharpe = float(daily.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else np.nan

    return {
        "n_trades": len(trades),
        "mean_gross": float(gross.mean()),
        "mean_abnormal": float(abn.mean()),
        "mean_net": float(net.mean()),
        "median_net": float(np.median(net)),
        "win_rate_net": float((net > 0).mean()),
        "tstat_net": float(net.mean() / se) if se and se == se else np.nan,
        "sleeve_sharpe": sharpe,
        "sleeve_max_drawdown": max_dd,
        "total_return": float(equity.iloc[-1] - 1.0) if len(equity) else np.nan,
    }


def cost_sweep(
    panel: dict[str, pd.DataFrame],
    events: pd.DataFrame,
    hold_days: int = 10,
    half_spread_grid=(5, 15, 25, 50, 100),
    base: CostModel = CostModel(),
) -> pd.DataFrame:
    """Mean net trade return as the half-spread rises — the micro-cap reality sweep.

    A liquid name lives at ~1–5 bps; the names a viral feed surfaces live at 25–100.
    This isolates the spread you'd actually cross and shows where the mean trade goes
    red.
    """
    rows = []
    for hs in half_spread_grid:
        c = CostModel(float(hs), base.commission_bps, base.slippage_bps)
        res = run(panel, events, hold_days, c)
        s = res.stats
        rows.append({
            "half_spread_bps": hs,
            "round_trip_bps": c.round_trip_frac() * 1e4,
            "mean_net": s.get("mean_net", np.nan),
            "win_rate_net": s.get("win_rate_net", np.nan),
            "n_trades": s.get("n_trades", 0),
        })
    return pd.DataFrame(rows).set_index("half_spread_bps")


def capacity(
    panel: dict[str, pd.DataFrame],
    events: pd.DataFrame,
    edge_bps: float,
    impact_coef: float = 0.1,
    adv_lookback: int = 21,
) -> dict:
    """Dollar size at which square-root market impact alone equals ``edge_bps``.

    Impact model: ``impact_bps(N) = impact_coef * 1e4 * sqrt(N / ADV$)`` where ``N``
    is the trade notional and ``ADV$`` the median dollar volume across the event
    names. Solving ``impact_bps = edge_bps`` gives the capacity per trade::

        N* = ADV$ * (edge_bps / (impact_coef * 1e4)) ** 2

    For a thin feed this is typically a few thousand dollars — below which you have
    an "edge", above which your own order is the move. Returns ``median_adv_usd,
    capacity_usd_per_trade, edge_bps``.
    """
    advs = []
    for t, grp in events.groupby("ticker"):
        frame = panel.get(t)
        if frame is None or "Volume" not in frame:
            continue
        dollar_vol = (frame["Close"] * frame["Volume"]).rolling(adv_lookback, min_periods=1).mean()
        for e in grp["entry_pos"].to_numpy():
            advs.append(float(dollar_vol.iloc[int(e)]))
    if not advs or edge_bps <= 0:
        return {"median_adv_usd": np.nan, "capacity_usd_per_trade": np.nan, "edge_bps": edge_bps}
    adv = float(np.median(advs))
    cap = adv * (edge_bps / (impact_coef * 1e4)) ** 2
    return {"median_adv_usd": adv, "capacity_usd_per_trade": cap, "edge_bps": float(edge_bps)}
