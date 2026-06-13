"""The Turtle Trader strategy and its honest controls — Study 103 (Turtle-Trader).

The legendary recipe: Richard Dennis's Turtle program used Donchian channel breakouts.
System 1 (short-term): buy when the close exceeds the *20-day high*, exit when it
falls below the *10-day low*; go short on breaks below the 20-day low, exit on breaks
above the 10-day high. System 2 (long-term): the same with 55-day entry / 20-day exit.
Both directions (long and short). N-unit ATR-based position sizing optional.

We implement this as a daily forward-return test and pin it against three honest controls:

- **random-entry control** — same direction (trend-following), but entries on random days
  (matched number of entries), to separate the *timing* signal from the holding-period
  trend premium.
- **buy-and-hold** — the baseline every active strategy must beat.
- **pre/post-2003 split** — the Turtle rules were published in full by Curtis Faith in 2003;
  we test whether the edge decayed post-publication (the McLean-Pontiff hypothesis).

No look-ahead: the channel extremes are computed on closes up to and including day *t*;
entry is executed at day *t+1*'s open.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Indicators
# ---------------------------------------------------------------------------
def donchian_high(close: pd.Series, n: int) -> pd.Series:
    """Rolling n-day high of closes (the Turtle entry channel for longs)."""
    return close.rolling(n, min_periods=n).max()


def donchian_low(close: pd.Series, n: int) -> pd.Series:
    """Rolling n-day low of closes (the Turtle entry channel for shorts)."""
    return close.rolling(n, min_periods=n).min()


def atr(bars: pd.DataFrame, n: int = 20) -> pd.Series:
    """Wilder-style average true range (price units).

    Used as the Turtle risk unit *N* for position sizing and as the reference
    volatility measure for the entry signal sensitivity analysis.
    """
    prev_close = bars["close"].shift(1)
    tr = pd.concat(
        [
            bars["high"] - bars["low"],
            (bars["high"] - prev_close).abs(),
            (bars["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(span=n, adjust=False, min_periods=n).mean()


# ---------------------------------------------------------------------------
# Signal generation — entries and exits
# ---------------------------------------------------------------------------
def donchian_signals(
    bars: pd.DataFrame,
    entry_n: int = 20,
    exit_n: int = 10,
    direction: str = "both",
) -> pd.DataFrame:
    """Donchian channel entry signals for the Turtle breakout rule.

    A long signal fires on the day the close first breaks above the ``entry_n``-day
    high (i.e. ``close > rolling_max_of_prior_entry_n_days``). Symmetrically, a short
    signal fires on a break below the ``entry_n``-day low. Exit signals are generated
    at the ``exit_n``-day low (for longs) or high (for shorts).

    No look-ahead: the channel extreme on day *t* is the max/min over the window
    ending at day *t*; the *entry* fires only when close *strictly exceeds* the
    prior-day's channel extreme, to avoid trading on the day that set the new extreme.

    Parameters
    ----------
    bars : pd.DataFrame with columns open/high/low/close
    entry_n : lookback for entry channel (20 for Sys1, 55 for Sys2)
    exit_n  : lookback for exit channel  (10 for Sys1, 20 for Sys2)
    direction : 'long', 'short', or 'both'

    Returns a DataFrame with columns:
      - ``entry_long``  : bool, long entry bar
      - ``entry_short`` : bool, short entry bar
      - ``exit_long``   : the exit threshold price for a long position (entry_n-day low)
      - ``exit_short``  : the exit threshold price for a short position (entry_n-day high)
      - ``chan_high``   : rolling entry_n-day high (shifted 1: known at start of day t)
      - ``chan_low``    : rolling entry_n-day low  (shifted 1)
    """
    close = bars["close"]

    # Channel extremes known at the *open* of the next day (shifted by 1 bar).
    chan_high = close.rolling(entry_n, min_periods=entry_n).max().shift(1)
    chan_low  = close.rolling(entry_n, min_periods=entry_n).min().shift(1)

    # Entry: close crosses above/below the *prior* channel extreme.
    entry_long  = (close > chan_high) & chan_high.notna()
    entry_short = (close < chan_low)  & chan_low.notna()

    # Exit thresholds (available at the start of each holding day — no look-ahead).
    exit_long  = close.rolling(exit_n, min_periods=exit_n).min()
    exit_short = close.rolling(exit_n, min_periods=exit_n).max()

    sig = pd.DataFrame(
        {
            "entry_long":   entry_long,
            "entry_short":  entry_short,
            "exit_long":    exit_long,
            "exit_short":   exit_short,
            "chan_high":    chan_high,
            "chan_low":     chan_low,
        },
        index=bars.index,
    )
    if direction == "long":
        sig["entry_short"] = False
    elif direction == "short":
        sig["entry_long"] = False
    return sig


# ---------------------------------------------------------------------------
# Forward-return engine — per-trade ledger
# ---------------------------------------------------------------------------
def run_trades(
    bars: pd.DataFrame,
    sig: pd.DataFrame,
    cost_bps: float = 5.0,
    max_hold: int = 252,
    random_entry_dates: np.ndarray | None = None,
    direction_override: np.ndarray | None = None,
) -> pd.DataFrame:
    """Simulate Turtle breakout trades and return a per-trade ledger.

    For each entry signal on day *t*, the position is opened at the *next* day's
    open. The position is held until the close falls below the exit_long channel
    (for longs) or rises above exit_short (for shorts), or until ``max_hold`` days
    are reached (whichever comes first). Exit is at the *next* bar's open after the
    exit trigger fires.

    ``random_entry_dates`` overrides the entry day indices (random-entry control):
    the same number of trades are initiated, but on randomly chosen days instead of
    breakout signals. ``direction_override`` overrides the entry direction (±1 vector).

    No look-ahead: the exit channel seen on day *t* is checked for the position entered
    at most *t+1* and older; the exit is executed at *t+1*'s open.

    Returns a DataFrame with columns:
      entry_date, dir, entry_px, exit_date, exit_px, exit_reason,
      days_held, ret_gross, ret_net.
    """
    close  = bars["close"].to_numpy()
    open_  = bars["open"].to_numpy()
    dates  = bars.index
    n      = len(bars)
    pos    = {d: i for i, d in enumerate(dates)}

    # Collect (bar_idx, direction) pairs
    if random_entry_dates is not None:
        pairs = [(int(i), int(d)) for i, d in random_entry_dates]
    else:
        long_entries  = sig.index[sig["entry_long"]].tolist()
        short_entries = sig.index[sig["entry_short"]].tolist()
        pairs = (
            [(pos[d], +1) for d in long_entries if d in pos]
            + [(pos[d], -1) for d in short_entries if d in pos]
        )
        pairs.sort(key=lambda x: x[0])

    if direction_override is not None:
        dirs_arr = np.asarray(direction_override, dtype=int)
        pairs = [(p[0], int(dirs_arr[k])) for k, p in enumerate(pairs)]

    # Resolve exit-channel arrays (already computed, no shift needed here).
    exit_long_arr  = sig["exit_long"].to_numpy()
    exit_short_arr = sig["exit_short"].to_numpy()

    rows = []
    for (entry_sig_i, d) in pairs:
        enter_i = entry_sig_i + 1   # open of the *next* day
        if enter_i >= n:
            continue
        entry_px = open_[enter_i]
        if not (np.isfinite(entry_px) and entry_px > 0):
            continue

        exit_i = enter_i
        exit_px = close[enter_i]
        exit_reason = "max_hold"

        for j in range(enter_i, min(enter_i + max_hold, n - 1)):
            # Exit signal is checked on day j (channel known from prior closes).
            if d == +1:
                trigger = exit_long_arr[j]
                if np.isfinite(trigger) and close[j] < trigger:
                    exit_i = j + 1
                    exit_px = open_[j + 1]
                    exit_reason = "exit_channel"
                    break
            else:
                trigger = exit_short_arr[j]
                if np.isfinite(trigger) and close[j] > trigger:
                    exit_i = j + 1
                    exit_px = open_[j + 1]
                    exit_reason = "exit_channel"
                    break
            exit_i = j
            exit_px = close[j]
            exit_reason = "max_hold"

        ret_gross = d * (exit_px - entry_px) / entry_px
        rows.append(
            {
                "entry_date":  dates[enter_i],
                "dir":         int(d),
                "entry_px":    float(entry_px),
                "exit_date":   dates[exit_i],
                "exit_px":     float(exit_px),
                "exit_reason": exit_reason,
                "days_held":   int(exit_i - enter_i),
                "ret_gross":   float(ret_gross),
                "ret_net":     float(ret_gross - cost_bps * 1e-4),
            }
        )
    return pd.DataFrame(rows)


def random_entry_pairs(
    n_trades: int,
    n_bars: int,
    entry_n: int = 20,
    seed: int = 0,
) -> np.ndarray:
    """Reproducible random (bar_idx, direction) pairs for the random-entry control.

    Sampled uniformly from the bars that have enough lookback for the channel
    (i.e. bar index >= entry_n). Direction is drawn i.i.d. ±1 (a coin).
    """
    rng = np.random.default_rng(seed)
    valid = np.arange(entry_n, n_bars - 1)
    idxs = rng.choice(valid, size=n_trades, replace=True)
    dirs = rng.choice([-1, 1], size=n_trades)
    return np.column_stack([idxs, dirs])


# ---------------------------------------------------------------------------
# Summarise a ledger
# ---------------------------------------------------------------------------
def summarize(ledger: pd.DataFrame, col: str = "ret_net") -> dict:
    """Headline per-trade statistics for one ledger.

    Returns trade count, win-rate, mean return (bps/trade), per-trade Sharpe,
    P&L skew, and a Newey-West HAC t-stat on the mean — the inference-bar number
    that decides whether the edge is distinguishable from zero.
    """
    r = ledger[col].to_numpy(dtype=float) if len(ledger) else np.array([])
    r = r[np.isfinite(r)]
    n = r.size
    out = {
        "n_trades": int(n),
        "win_rate": float((r > 0).mean()) if n else float("nan"),
        "mean_bps": float(r.mean() * 1e4) if n else float("nan"),
        "sharpe_per_trade": (
            float(r.mean() / r.std(ddof=1)) if n > 1 and r.std() > 0 else float("nan")
        ),
        "skew": float(pd.Series(r).skew()) if n > 2 else float("nan"),
        "tstat": float("nan"),
    }
    if n > 5:
        # Newey-West HAC: the desk's inference standard (no hard dep. on quantlab).
        mu = r.mean()
        e = r - mu
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
        lrv = float(e @ e) / n
        for k in range(1, lags + 1):
            w = 1.0 - k / (lags + 1.0)
            lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
        se = np.sqrt(max(lrv, 0.0) / n)
        out["tstat"] = float(mu / se) if se > 0 else float("nan")
    return out


def equity_curve(ledger: pd.DataFrame, col: str = "ret_net") -> pd.Series:
    """Cumulative compound return from a per-trade ledger, indexed by entry_date."""
    if ledger.empty:
        return pd.Series(dtype=float)
    r = ledger.set_index("entry_date")[col].dropna()
    return (1.0 + r).cumprod()
