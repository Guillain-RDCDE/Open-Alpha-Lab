"""From an oscillator to a costed P&L — the same engine for all three, so a comparison
is fair by construction.

One function, :func:`run_strategy`, turns *any* position series into a daily-return ledger:
the position is **lagged one bar** (you act tomorrow on today's close, no look-ahead), the
gross return is ``position × next-day return``, and a **turnover cost** in basis points is
charged on every change in position. Whether the position came from the TSI, the MACD or the
RSI, it goes through this identical pipe — so if their equity curves differ, it's the signal
talking, not the harness.

:func:`trades_from_position` segments the position into holding *spells* so we can report the
broker-statement numbers QuantifiedStrategies quotes — number of trades, win ratio, profit
factor, average gain — on the same footing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quantlab.stats import annualized_sharpe

from . import oscillators as osc

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Position builders — one per oscillator, plus a generic dispatcher
# --------------------------------------------------------------------------- #

def tsi_position(close: pd.Series, params: osc.TSIParams = osc.TSIParams(),
                 rule: str = "zero", long_short: bool = False) -> pd.Series:
    """TSI position. ``rule='zero'`` = zero-line cross; ``rule='signal'`` = signal-line cross."""
    t = osc.tsi(close, params)
    if rule == "signal":
        return osc.position_signal_cross(t["tsi"], t["signal"], long_short)
    return osc.position_zero_cross(t["tsi"], long_short)


def macd_position(close: pd.Series, params: osc.MACDParams = osc.MACDParams(),
                  rule: str = "zero", long_short: bool = False) -> pd.Series:
    m = osc.macd(close, params)
    if rule == "signal":
        return osc.position_signal_cross(m["macd"], m["signal"], long_short)
    return osc.position_zero_cross(m["macd"], long_short)   # MACD line vs zero


def rsi_position(close: pd.Series, period: int = 14, long_short: bool = False) -> pd.Series:
    """RSI position: long above the 50 mid-line (its zero-cross), else flat/short."""
    return osc.position_zero_cross(osc.rsi(close, period) - 50.0, long_short)


def positions_by_oscillator(close: pd.Series, rule: str = "zero",
                            long_short: bool = False) -> pd.DataFrame:
    """All three positions on one index, default params. Columns: ``tsi``, ``macd``, ``rsi``."""
    return pd.DataFrame({
        "tsi": tsi_position(close, rule=rule, long_short=long_short),
        "macd": macd_position(close, rule=rule, long_short=long_short),
        "rsi": rsi_position(close, long_short=long_short),
    })


# --------------------------------------------------------------------------- #
# The engine
# --------------------------------------------------------------------------- #

def run_strategy(close: pd.Series, position: pd.Series, cost_bps: float = 10.0) -> pd.DataFrame:
    """Daily ledger for a position series. Columns: ``pos``, ``ret``, ``gross``, ``net``.

    The position is lagged one bar (act next session), the market return is the close-to-
    close pct change, and ``cost_bps`` is charged on the **turnover** ``|Δpos|`` each bar
    (a flat->long->flat round trip pays the cost twice — once in, once out). One-sided cost.
    """
    pos = position.reindex(close.index).shift(1).fillna(0.0).astype(float)
    ret = close.astype(float).pct_change().fillna(0.0)
    gross = pos * ret
    turnover = pos.diff().abs().fillna(pos.abs())
    cost = turnover * (cost_bps / 1e4)
    net = gross - cost
    return pd.DataFrame({"pos": pos, "ret": ret, "gross": gross, "net": net, "turnover": turnover})


def trades_from_position(ledger: pd.DataFrame) -> pd.DataFrame:
    """Segment the (already-lagged) position into holding spells; one row per trade.

    A trade is a maximal run of constant non-zero position. Its return is the compounded
    net daily return over the spell. Returns columns ``start``, ``end``, ``bars``, ``side``,
    ``ret`` — the object the win-rate / profit-factor stats are built on.
    """
    pos = ledger["pos"].to_numpy(float)
    net = ledger["net"].to_numpy(float)
    idx = ledger.index
    rows = []
    i, n = 0, len(pos)
    while i < n:
        if pos[i] == 0.0:
            i += 1
            continue
        side = np.sign(pos[i])
        j = i
        comp = 1.0
        while j < n and np.sign(pos[j]) == side and pos[j] != 0.0:
            comp *= (1.0 + net[j])
            j += 1
        rows.append({"start": idx[i], "end": idx[j - 1], "bars": j - i,
                     "side": float(side), "ret": comp - 1.0})
        i = j
    return pd.DataFrame(rows, columns=["start", "end", "bars", "side", "ret"])


def strategy_stats(ledger: pd.DataFrame, periods_per_year: int = TRADING_DAYS) -> dict:
    """Broker-statement summary for one strategy — the table QuantifiedStrategies quotes.

    CAGR, annualised Sharpe (on net daily returns), time in market, max drawdown, and the
    per-trade win ratio / profit factor / average gain, so our numbers line up with theirs.
    """
    if ledger.empty:
        return {"n_trades": 0}
    net = ledger["net"]
    equity = (1.0 + net).cumprod()
    years = len(net) / periods_per_year
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and equity.iloc[-1] > 0 else float("nan")
    dd = float((equity / equity.cummax() - 1.0).min())
    exposure = float((ledger["pos"] != 0).mean())

    tr = trades_from_position(ledger)
    if len(tr):
        wins = tr.loc[tr["ret"] > 0, "ret"]
        losses = tr.loc[tr["ret"] <= 0, "ret"]
        gross_win = float(wins.sum())
        gross_loss = float(-losses.sum())
        pf = gross_win / gross_loss if gross_loss > 0 else float("inf")
        win_rate = float((tr["ret"] > 0).mean())
        avg_gain = float(tr["ret"].mean())
    else:
        pf, win_rate, avg_gain = float("nan"), float("nan"), float("nan")

    return {
        "n_trades": int(len(tr)),
        "cagr": cagr,
        "sharpe": float(annualized_sharpe(net, periods_per_year)),
        "exposure": exposure,
        "max_drawdown": dd,
        "profit_factor": pf,
        "win_rate": win_rate,
        "avg_gain": avg_gain,
    }


# --------------------------------------------------------------------------- #
# Portfolio helper — equal-weight a strategy across a universe
# --------------------------------------------------------------------------- #

def universe_stats(frames: dict[str, pd.DataFrame], position_func, cost_bps: float = 10.0,
                   periods_per_year: int = TRADING_DAYS) -> dict:
    """Broker statement for a strategy across the whole universe — QS's table, honestly.

    Portfolio metrics (CAGR, Sharpe, max drawdown, time-in-market) come from the equal-weight
    net daily return; the per-trade metrics (count, win ratio, profit factor, average gain)
    are **pooled over every name's trades**, so a flat-position EW series can't smear them
    into one giant position. This is the apples-to-apples version of the numbers
    QuantifiedStrategies quotes for their single-asset GLD backtest.
    """
    net_cols, pos_cols, trade_frames = {}, {}, []
    for tk, f in frames.items():
        led = run_strategy(f["Close"], position_func(f["Close"]), cost_bps=cost_bps)
        net_cols[tk] = led["net"]
        pos_cols[tk] = led["pos"].abs()
        tr = trades_from_position(led)
        if len(tr):
            trade_frames.append(tr)
    if not net_cols:
        return {"n_trades": 0}
    ew = pd.DataFrame(net_cols).sort_index().mean(axis=1, skipna=True)
    exposure = float(pd.DataFrame(pos_cols).mean(axis=1, skipna=True).mean())
    equity = (1.0 + ew).cumprod()
    years = len(ew) / periods_per_year
    cagr = float(equity.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and equity.iloc[-1] > 0 else float("nan")
    dd = float((equity / equity.cummax() - 1.0).min())

    tr = pd.concat(trade_frames, ignore_index=True) if trade_frames else pd.DataFrame(columns=["ret"])
    wins = tr.loc[tr["ret"] > 0, "ret"]
    losses = tr.loc[tr["ret"] <= 0, "ret"]
    pf = float(wins.sum() / -losses.sum()) if (len(losses) and losses.sum() < 0) else float("inf")
    return {
        "n_trades": int(len(tr)),
        "cagr": cagr,
        "sharpe": float(annualized_sharpe(ew, periods_per_year)),
        "exposure": exposure,
        "max_drawdown": dd,
        "profit_factor": pf,
        "win_rate": float((tr["ret"] > 0).mean()) if len(tr) else float("nan"),
        "avg_gain": float(tr["ret"].mean()) if len(tr) else float("nan"),
    }


def equal_weight_net(frames: dict[str, pd.DataFrame], position_func, cost_bps: float = 10.0) -> pd.Series:
    """Equal-weight the per-name net daily returns of a strategy across the universe.

    ``position_func(close) -> position`` is applied to each name. Returns the cross-sectional
    mean net daily return (a tradable EW portfolio), reindexed to the union of dates. This is
    the single series each Reality-Check grid column is built from.
    """
    cols = {}
    for tk, f in frames.items():
        led = run_strategy(f["Close"], position_func(f["Close"]), cost_bps=cost_bps)
        cols[tk] = led["net"]
    if not cols:
        return pd.Series(dtype=float)
    panel = pd.DataFrame(cols).sort_index()
    return panel.mean(axis=1, skipna=True).rename("ew_net")
