"""The strategy and its honest controls — Study 241 (Buy-the-Dip).

The folk rule: "buy the dip" — deploy cash into SPY after a drawdown of X% from
the recent high; otherwise sit in cash (or short-term bonds). The idea is that dips
are buying opportunities and the disciplined investor who waits for dips will
outperform the passive buy-and-hold investor.

We test a grid of drawdown thresholds (2%, 5%, 10%, 15%, 20%) and compare THREE
strategies on the *same* daily price tape:

  1. **Buy-and-hold (BH)**: invest 100% on day 1, never trade.
  2. **Dip-buyer (DB)**: hold cash (0% return) when in a drawdown shallower than
     threshold; buy at next day's open when drawdown >= threshold; exit when price
     recovers to a new all-time high.
  3. **DCA control**: invest 1/N each month regardless of price (monthly installment
     over the same N years).

The critical insight is the **cash drag**: while waiting for a dip, the dip-buyer
earns zero (or a risk-free rate, which we test). During a secular bull market, the
probability of sitting in cash while the market rises is substantial — and this
cash drag is the primary reason dip-buying typically underperforms buy-and-hold.

Key design choices:
- Drawdown is computed from the running maximum of daily closes (the "high-water mark").
- Entry is at the *next* day's open after the threshold is first breached (no look-ahead).
- Exit back to cash is at the *next* day's open after a new ATH close.
- Costs: 1 one-way bp per trade (negligible for ETFs, included for completeness).
- No leverage, no short selling.
- Cash earns 0% (conservative) or a risk-free proxy (tested separately).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

COST_BPS = 1.0  # one-way commission in bps (very low for ETF)


# ---------------------------------------------------------------------------
# Core utility: drawdown from running high
# ---------------------------------------------------------------------------
def running_drawdown(close: pd.Series) -> pd.Series:
    """Compute the running drawdown from the cumulative maximum close.

    Returns a Series of negative-or-zero values: 0.0 means at an ATH,
    -0.10 means 10% below the running high.
    """
    running_max = close.cummax()
    return (close - running_max) / running_max


# ---------------------------------------------------------------------------
# Buy-and-hold benchmark
# ---------------------------------------------------------------------------
def buy_and_hold_equity(bars: pd.DataFrame) -> pd.Series:
    """Normalised equity curve for always-invested buy-and-hold (starts at 1.0).

    Entry at first bar's open; all subsequent daily changes compound on the close.
    """
    close = bars["close"]
    return close / close.iloc[0]


# ---------------------------------------------------------------------------
# Dip-buying strategy
# ---------------------------------------------------------------------------
def dip_buyer_equity(
    bars: pd.DataFrame,
    threshold: float = 0.10,
    cost_bps: float = COST_BPS,
    cash_rate_annual: float = 0.0,
) -> tuple[pd.Series, dict]:
    """Run the dip-buying strategy and return a daily equity curve + metadata.

    The rule:
    - Start in **cash** (equity = 1.0, earning ``cash_rate_annual`` p.a.).
    - When the running drawdown first reaches ``-threshold`` (e.g. -0.10 for 10%):
      **enter** at the next bar's open (deduct one-way cost).
    - While **invested**: earn the daily close-to-close return.
    - When price makes a new **all-time-high close**: **exit** at the next bar's open
      (deduct one-way cost), return to cash.
    - Repeat until end of tape.

    Parameters
    ----------
    threshold :
        Drawdown depth that triggers a buy, as a positive fraction (0.10 = 10% off ATH).
    cost_bps :
        One-way cost in basis points (applied at each buy and each sell).
    cash_rate_annual :
        Annual return earned while in cash (0.0 = pure cash drag; ~0.04 = T-bill proxy).

    Returns
    -------
    equity : pd.Series
        Daily portfolio equity starting at 1.0.
    meta : dict
        Trade statistics: n_buys, n_days_invested, fraction_invested, total_trades.
    """
    close = bars["close"]
    open_ = bars["open"]
    n = len(bars)

    daily_cash = (1.0 + cash_rate_annual) ** (1.0 / 252) - 1.0
    cost_factor = 1.0 - cost_bps * 1e-4

    equity = np.ones(n)
    in_market = False
    entry_equity = 1.0  # equity when we entered the market
    entry_px = None
    running_max_close = close.iloc[0]

    # Track the running max of *closes seen so far* at signal time
    max_close_at_signal = close.iloc[0]

    n_buys = 0
    n_days_invested = 0
    signal_to_enter = False  # flag: buy at next bar's open
    signal_to_exit = False   # flag: sell at next bar's open

    for i in range(n):
        c = close.iloc[i]
        o = open_.iloc[i]

        # Execute pending entry signal
        if signal_to_enter and not in_market:
            # Enter at today's open
            entry_px = o * (1.0 / cost_factor)  # cost increases effective entry price
            in_market = True
            n_buys += 1
            signal_to_enter = False

        # Execute pending exit signal
        if signal_to_exit and in_market:
            # Exit at today's open; equity already correct from close prices
            # Apply one-way exit cost: reduce equity by cost
            equity[i] = equity[i - 1] * cost_factor if i > 0 else cost_factor
            # Mark the exit: we now hold cash at this equity level
            entry_equity = equity[i]
            entry_px = None
            in_market = False
            signal_to_exit = False

        # Compute today's equity
        if i == 0:
            equity[i] = 1.0
        elif in_market:
            ret = (close.iloc[i] / close.iloc[i - 1]) - 1.0
            equity[i] = equity[i - 1] * (1.0 + ret)
            n_days_invested += 1
        else:
            # In cash
            equity[i] = equity[i - 1] * (1.0 + daily_cash)

        # Update running max CLOSE (signal uses closes only, no look-ahead)
        if c > running_max_close:
            running_max_close = c

        # Generate signals based on today's close (executed at tomorrow's open)
        dd = (c - running_max_close) / running_max_close  # <= 0
        if not in_market and dd <= -threshold:
            signal_to_enter = True
        if in_market and c >= running_max_close:
            # At or above ATH close → exit tomorrow
            signal_to_exit = True

    fraction_invested = n_days_invested / n if n > 0 else 0.0
    meta = {
        "n_buys": n_buys,
        "n_days_invested": n_days_invested,
        "fraction_invested": fraction_invested,
        "total_trades": n_buys * 2,  # round trips
    }
    eq = pd.Series(equity, index=bars.index, name=f"dip_{int(threshold*100)}pct")
    return eq, meta


# ---------------------------------------------------------------------------
# DCA control
# ---------------------------------------------------------------------------
def dca_equity(bars: pd.DataFrame, freq: str = "ME") -> pd.Series:
    """Dollar-cost-averaging: invest equal amounts each calendar month-end.

    Simulates an investor who has the full capital on day 1 but drip-feeds it
    in equal installments of 1/N at the close of each month (first installment
    at first month-end after start). This is the fairest DCA comparison: same
    total capital, same start, regular cadence.

    Returns a normalised equity curve starting at 0.0 (no capital invested yet)
    and rising to reflect the growing invested portion.
    """
    close = bars["close"]
    monthly_dates = pd.date_range(close.index[0], close.index[-1], freq=freq)
    # Find actual trading days closest to each month-end
    invest_dates = []
    for d in monthly_dates:
        candidates = close.index[close.index <= d]
        if len(candidates):
            invest_dates.append(candidates[-1])

    invest_dates = sorted(set(invest_dates))
    N = len(invest_dates)
    if N == 0:
        return pd.Series(np.ones(len(close)), index=close.index, name="dca")

    # Build equity: track each tranche separately, then sum
    equity = np.zeros(len(close))
    for inv_date in invest_dates:
        idx = close.index.get_loc(inv_date)
        # Each tranche is 1/N of total; grows from close[idx] onward
        tranche_equity = np.zeros(len(close))
        px0 = close.iloc[idx]
        for j in range(idx, len(close)):
            tranche_equity[j] = (1.0 / N) * (close.iloc[j] / px0)
        equity += tranche_equity

    # Before any investment, return 1.0 in cash (normalise so we can compare)
    # Actually: scale so total equity starts at ~1.0 when fully deployed
    # Normalise to first invested tranche = 1.0 baseline
    eq = pd.Series(equity, index=close.index, name="dca")
    # scale so that final value is comparable: normalise by fraction invested × final
    # Simpler: just return raw (each tranche = 1/N, so at full deployment = sum(1/N) = 1.0)
    return eq


# ---------------------------------------------------------------------------
# Summary statistics for an equity curve vs buy-and-hold
# ---------------------------------------------------------------------------
def summarize_equity(
    eq: pd.Series,
    bh: pd.Series,
    n_years: float | None = None,
) -> dict:
    """Compute CAGR, Sharpe (annualised), max drawdown, and relative performance vs BH.

    Parameters
    ----------
    eq :
        Strategy equity curve (starts at ~1.0).
    bh :
        Buy-and-hold equity curve (starts at 1.0).
    n_years :
        Number of years in the tape (computed from length if None).
    """
    n = len(eq)
    if n_years is None:
        n_years = n / 252.0

    # CAGR
    terminal = float(eq.iloc[-1]) / float(eq.iloc[0]) if float(eq.iloc[0]) > 0 else np.nan
    cagr = terminal ** (1.0 / n_years) - 1.0 if n_years > 0 and terminal > 0 else np.nan

    bh_terminal = float(bh.iloc[-1]) / float(bh.iloc[0])
    bh_cagr = bh_terminal ** (1.0 / n_years) - 1.0 if n_years > 0 else np.nan

    # Daily returns
    daily_ret = eq.pct_change().dropna()
    sharpe = (
        float(daily_ret.mean()) / float(daily_ret.std(ddof=1)) * np.sqrt(252)
        if daily_ret.std() > 0 else np.nan
    )

    bh_daily = bh.pct_change().dropna()
    bh_sharpe = (
        float(bh_daily.mean()) / float(bh_daily.std(ddof=1)) * np.sqrt(252)
        if bh_daily.std() > 0 else np.nan
    )

    # Max drawdown
    roll_max = eq.cummax()
    dd_series = (eq - roll_max) / roll_max
    max_dd = float(dd_series.min())

    return {
        "cagr": cagr,
        "bh_cagr": bh_cagr,
        "cagr_vs_bh": cagr - bh_cagr,
        "sharpe": sharpe,
        "bh_sharpe": bh_sharpe,
        "max_dd": max_dd,
        "terminal": terminal,
        "bh_terminal": bh_terminal,
        "n_years": n_years,
    }


# ---------------------------------------------------------------------------
# Grid sweep over thresholds
# ---------------------------------------------------------------------------
THRESHOLDS = [0.02, 0.05, 0.10, 0.15, 0.20]


def threshold_sweep(
    bars: pd.DataFrame,
    thresholds: list[float] = THRESHOLDS,
    cost_bps: float = COST_BPS,
    cash_rate_annual: float = 0.0,
) -> pd.DataFrame:
    """Run dip-buyer for each threshold and compare to buy-and-hold.

    Returns a DataFrame with one row per threshold and columns:
    threshold, cagr, bh_cagr, cagr_vs_bh, sharpe, bh_sharpe, max_dd,
    n_buys, fraction_invested.
    """
    bh = buy_and_hold_equity(bars)
    rows = []
    for thr in thresholds:
        eq, meta = dip_buyer_equity(
            bars, threshold=thr, cost_bps=cost_bps,
            cash_rate_annual=cash_rate_annual,
        )
        s = summarize_equity(eq, bh)
        rows.append({
            "threshold_pct": int(round(thr * 100)),
            "cagr": s["cagr"],
            "bh_cagr": s["bh_cagr"],
            "cagr_vs_bh": s["cagr_vs_bh"],
            "sharpe": s["sharpe"],
            "bh_sharpe": s["bh_sharpe"],
            "max_dd": s["max_dd"],
            "n_buys": meta["n_buys"],
            "frac_invested": meta["fraction_invested"],
        })
    return pd.DataFrame(rows)
