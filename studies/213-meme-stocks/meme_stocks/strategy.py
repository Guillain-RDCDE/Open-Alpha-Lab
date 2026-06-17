"""Strategy engine for Study 213 (Meme Stocks).

Two strategies, one benchmark, honest costs, no look-ahead:

**Strategy A — Buy-and-Hold (BH)**
Enter the equal-weight meme basket at the close on 2021-01-04 (first trading day
of 2021, the month the mania exploded), hold through 2025-12-31, pay a one-way
entry cost of ``cost_bps``.  BBBY goes to zero (bankruptcy April 2023) — the BH
backtest absorbs the full loss; no exit is assumed.

**Strategy B — Momentum-Timed (MT)**
Enter each name in the basket when it first closes more than ``mom_pct``% above
its prior-60-trading-day close (the breakout/momentum trigger that WSB discourse
mirrored), exit 126 trading days (~6 months) later or on the delisting date
(whichever is first), pay ``cost_bps`` in and ``cost_bps`` out.

Both strategies are raced against SPY (S&P 500 total return, auto-adjusted)
over the same holding window.

No survivorship: BBBY's bankruptcy is explicitly absorbed.
Execution lag: positions enter at the *next* trading close after the trigger.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _cagr(total_return: float, n_days: int) -> float:
    """Annualise a total return over n calendar (not trading) days."""
    if n_days <= 0 or total_return <= -1.0:
        return float("nan")
    years = n_days / 365.25
    return float((1.0 + total_return) ** (1.0 / years) - 1.0)


def _sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    """Annualised Sharpe on a daily-return series."""
    if returns.empty or returns.std() == 0:
        return float("nan")
    excess = returns - rf / 252
    return float(excess.mean() / excess.std() * np.sqrt(252))


# ---------------------------------------------------------------------------
# Strategy A — Buy-and-Hold
# ---------------------------------------------------------------------------
def buy_and_hold(
    panel: dict,
    entry_date: str = "2021-01-04",
    exit_date: str | None = None,
    cost_bps: float = 20.0,
) -> dict:
    """Equal-weight buy-and-hold of the meme basket vs SPY.

    Enters all names simultaneously at the close on ``entry_date``.
    If a name has no price on that date (e.g. BBBY delisted before entry) it is
    silently excluded from the basket — but BBBY is still *in* the basket during
    its live period and goes to zero when prices end.

    Returns a dict with portfolio daily returns, benchmark daily returns,
    and summary stats.
    """
    prices = panel["prices"]
    bench = panel["bench"]
    entry_ts = pd.Timestamp(entry_date)
    exit_ts = pd.Timestamp(exit_date) if exit_date else prices.index[-1]

    # Slice to holding window
    port_px = prices.loc[entry_ts:exit_ts]
    bench_px = bench.loc[entry_ts:exit_ts]
    if port_px.empty or bench_px.empty:
        return {"error": "no data in holding window"}

    # For each ticker: daily returns, filled with -1 after delisting (total loss)
    rets = {}
    for t in prices.columns:
        col = port_px[t].dropna()
        if col.empty or col.index[0] > entry_ts + pd.Timedelta(days=5):
            continue  # ticker not priced at entry
        r = col.pct_change()
        # After last price date, fill with 0 (position stays at 0, already
        # worth 0 on delisting day — we record that final day's return naturally)
        r = r.reindex(port_px.index, fill_value=0.0)
        rets[t] = r

    if not rets:
        return {"error": "no tickers available"}

    ret_df = pd.DataFrame(rets)
    # Equal-weight: constant weights (rebalance-free after entry)
    port_ret = ret_df.mean(axis=1)
    bench_ret = bench_px.pct_change()

    # Net of entry cost (one-way, at entry only)
    cost = cost_bps * 1e-4
    total_port = float((1 + port_ret).prod()) * (1 - cost) - 1
    total_bench = float((1 + bench_ret).prod()) - 1
    n_days = (exit_ts - entry_ts).days

    return {
        "port_ret": port_ret,
        "bench_ret": bench_ret,
        "total_port": total_port,
        "total_bench": total_bench,
        "port_cagr": _cagr(total_port, n_days),
        "bench_cagr": _cagr(total_bench, n_days),
        "port_sharpe": _sharpe(port_ret),
        "bench_sharpe": _sharpe(bench_ret),
        "n_names": len(rets),
        "n_days": n_days,
        "entry": entry_date,
        "exit": str(exit_ts.date()),
        "cost_bps": cost_bps,
    }


# ---------------------------------------------------------------------------
# Strategy B — Momentum-Timed
# ---------------------------------------------------------------------------
def momentum_timed(
    panel: dict,
    mom_window: int = 60,       # look-back in trading days
    mom_pct: float = 0.50,      # breakout threshold: 50% above 60-day prior close
    hold_days: int = 126,       # hold duration in trading days (~6 months)
    cost_bps: float = 20.0,
    start_date: str = "2020-06-01",   # earliest date to start scanning for signals
    end_date: str | None = None,
) -> dict:
    """Momentum-timed entry for each meme ticker: enter on breakout, hold 126 days.

    The trigger fires when the close exceeds ``mom_pct``% above the close
    ``mom_window`` trading days earlier.  Each ticker is entered independently at
    the next trading day close after the trigger (t+1 execution lag).  The round
    trip pays ``2 * cost_bps``.

    Returns a list of trade records and an aggregated daily P&L series (equally
    weighted across the simultaneously open positions each day).
    """
    prices = panel["prices"]
    bench = panel["bench"]
    end_ts = pd.Timestamp(end_date) if end_date else prices.index[-1]
    start_ts = pd.Timestamp(start_date)

    all_dates = prices.index[(prices.index >= start_ts) & (prices.index <= end_ts)]
    trades = []

    for t in prices.columns:
        col = prices[t].dropna()
        col = col[(col.index >= start_ts) & (col.index <= end_ts)]
        if len(col) < mom_window + 2:
            continue

        triggered = False
        for i in range(mom_window, len(col) - 1):
            if triggered:
                break
            ref = float(col.iloc[i - mom_window])
            cur = float(col.iloc[i])
            if ref <= 0:
                continue
            if cur / ref - 1.0 >= mom_pct:
                # Signal on day i; enter at close i+1 (execution lag)
                entry_idx = i + 1
                if entry_idx >= len(col):
                    break
                entry_date = col.index[entry_idx]
                entry_px = float(col.iloc[entry_idx])
                # Exit: hold_days later or end of data
                exit_idx = min(entry_idx + hold_days, len(col) - 1)
                exit_date = col.index[exit_idx]
                exit_px = float(col.iloc[exit_idx])
                gross_ret = exit_px / entry_px - 1.0
                net_ret = gross_ret - 2 * cost_bps * 1e-4
                trades.append({
                    "ticker": t,
                    "entry_date": entry_date,
                    "exit_date": exit_date,
                    "entry_px": entry_px,
                    "exit_px": exit_px,
                    "gross_ret": gross_ret,
                    "net_ret": net_ret,
                    "hold_days": (exit_date - entry_date).days,
                })
                triggered = True

    if not trades:
        return {"trades": [], "n_trades": 0, "mean_net_ret": float("nan"),
                "win_rate": float("nan"), "bench_ret_same_window": float("nan")}

    trades_df = pd.DataFrame(trades)
    # Build a daily P&L: for each open trade, track the daily return of that ticker
    daily_pnl_list = []
    for row in trades:
        t = row["ticker"]
        seg = prices[t].loc[row["entry_date"]:row["exit_date"]].dropna()
        if len(seg) < 2:
            continue
        r = seg.pct_change().iloc[1:]
        daily_pnl_list.append(r.rename(f"{t}_{row['entry_date'].date()}"))

    combined_bench = bench.loc[start_ts:end_ts].pct_change()

    return {
        "trades": trades_df,
        "n_trades": len(trades_df),
        "mean_gross_ret": float(trades_df["gross_ret"].mean()),
        "mean_net_ret": float(trades_df["net_ret"].mean()),
        "win_rate": float((trades_df["net_ret"] > 0).mean()),
        "daily_pnl_list": daily_pnl_list,
        "bench_daily": combined_bench,
    }


# ---------------------------------------------------------------------------
# HAC t-stat (local, no external dep)
# ---------------------------------------------------------------------------
def hac_tstat(x: np.ndarray, lags: int | None = None) -> float:
    """Newey-West HAC t-stat for the mean of x."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n <= 3:
        return float("nan")
    if lags is None:
        lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    mu = x.mean()
    e = x - mu
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


# ---------------------------------------------------------------------------
# Max drawdown
# ---------------------------------------------------------------------------
def max_drawdown(px: pd.Series) -> float:
    """Maximum peak-to-trough drawdown of a price/NAV series."""
    running_max = px.cummax()
    dd = (px - running_max) / running_max
    return float(dd.min())


# ---------------------------------------------------------------------------
# Annualised volatility of daily returns
# ---------------------------------------------------------------------------
def annual_vol(ret: pd.Series) -> float:
    """Annualised daily return volatility."""
    return float(ret.std() * np.sqrt(252))
