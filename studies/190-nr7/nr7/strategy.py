"""The NR7 strategy and its honest controls — Study 190 (NR7).

Tony Crabel (1990, *Day Trading With Short-Term Price Patterns*) described the
**NR7**: a day whose high-minus-low range is the *narrowest* of the last 7 days.
The hypothesis is that volatility contraction precedes volatility expansion — the
NR7 day is a "coiled spring".  Two sub-claims are tested:

1. **Volatility expansion** — the day *after* an NR7 has a wider range than
   unconditional daily ranges (a pure volatility claim, no direction).
2. **Directional breakout edge** — buying above the NR7 high (or selling below the
   low) on the next open gives a positive expectancy relative to an unconditional
   baseline and a random-day control.

Controls:

- **Random-day control** — same number of trading days as there are NR7 signals,
  drawn uniformly at random (seeded), avoiding look-ahead.  If the NR7 "spring"
  does something special, it should beat random days.
- **Unconditional baseline** — the full-sample distribution of next-day returns /
  next-day range ratios (all days, not just NR7 days), as an always-available
  alternative hypothesis.

No look-ahead: the NR7 label is assigned at the *close* of day *t* (signal date);
all forward measurements start on day *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# NR7 detection
# ---------------------------------------------------------------------------
def daily_range(bars: pd.DataFrame) -> pd.Series:
    """High minus Low for each bar."""
    return (bars["high"] - bars["low"]).rename("range")


def nr7_signals(bars: pd.DataFrame, lookback: int = 7) -> pd.Series:
    """Boolean Series: True on days whose range is the narrowest of the last ``lookback`` days.

    The NR7 label for day *t* is assigned using data known at the *close* of day *t*
    — no look-ahead.  The minimum requires at least ``lookback`` complete bars
    (including the current one), so the first ``lookback - 1`` rows are always False.

    Parameters
    ----------
    bars : DataFrame with at least 'high' and 'low' columns.
    lookback : int, default 7 (Tony Crabel's original).

    Returns
    -------
    pd.Series[bool] indexed like ``bars``, True on NR7 days.
    """
    rng = daily_range(bars)
    rolling_min = rng.rolling(lookback, min_periods=lookback).min()
    return (rng == rolling_min).rename("nr7")


# ---------------------------------------------------------------------------
# Volatility-expansion test (no direction)
# ---------------------------------------------------------------------------
def range_expansion_ratio(
    bars: pd.DataFrame,
    signal_mask: pd.Series,
    lookback_range: int = 20,
) -> pd.DataFrame:
    """Next-day range relative to a rolling baseline, for signal days vs the unconditional.

    For each signal day *t*, the *next* day *t+1*'s range is divided by the
    rolling mean range over the prior ``lookback_range`` days (ending at day *t*,
    so no look-ahead).  Returns a DataFrame with columns:

    - ``next_range``    : next-day absolute range
    - ``baseline_range``: rolling mean range on the signal day
    - ``ratio``         : next_range / baseline_range (>1 = expansion)

    Only includes rows where all three are finite.
    """
    rng = daily_range(bars)
    baseline = rng.rolling(lookback_range, min_periods=lookback_range).mean()

    # Align: signal on day t, measurement on day t+1.
    sig_dates = bars.index[signal_mask.reindex(bars.index, fill_value=False)]
    rows = []
    for d in sig_dates:
        loc = bars.index.get_loc(d)
        if loc + 1 >= len(bars):
            continue
        nxt_idx = bars.index[loc + 1]
        nr  = rng.iat[loc + 1]
        bsl = baseline.iat[loc]
        if np.isfinite(nr) and np.isfinite(bsl) and bsl > 0:
            rows.append({"signal_date": d, "next_range": nr,
                         "baseline_range": bsl, "ratio": nr / bsl})
    if not rows:
        return pd.DataFrame(columns=["signal_date", "next_range", "baseline_range", "ratio"])
    return pd.DataFrame(rows).set_index("signal_date")


# ---------------------------------------------------------------------------
# Breakout-trade backtest
# ---------------------------------------------------------------------------
def breakout_returns(
    bars: pd.DataFrame,
    signal_mask: pd.Series,
    cost_bps: float = 5.0,
    hold_days: int = 1,
) -> pd.DataFrame:
    """Per-signal breakout-trade ledger.

    Entry mechanics (Crabel style):
    - At the open of day *t+1* (the day after the NR7), place a buy-stop one
      tick above the NR7 high AND a sell-stop one tick below the NR7 low.
    - The *first* barrier touched intraday (high or low of the entry day) triggers
      the fill at the NR7 high (or low).  If the open gaps past the NR7 high, fill
      at the open.
    - Hold for ``hold_days`` sessions (default 1), exit at the close of that day.
    - If neither barrier is touched on the entry day, the trade is skipped (rare
      on the open-gaps version; here we fill at the open regardless).

    Simplification for daily bars: we simulate the open-trigger as follows:
    - If ``open_{t+1} > high_t``: fill long at ``open_{t+1}`` (gap up above NR7 high).
    - If ``open_{t+1} < low_t``:  fill short at ``open_{t+1}`` (gap down below NR7 low).
    - Otherwise: if ``high_{t+1} > high_t``: fill long at ``high_t``; else if
      ``low_{t+1} < low_t``: fill short at ``low_t``; else no fill (no breakout).

    On hold, exit at the close ``hold_days`` sessions later.

    ``cost_bps`` is a one-way-times-two round-trip cost deducted from the gross return.
    The ledger columns: ``signal_date, dir, entry, exit, ret_gross, ret_net``.
    """
    sig_mask = signal_mask.reindex(bars.index, fill_value=False)
    sig_dates = bars.index[sig_mask]
    rows = []
    n = len(bars)

    for d in sig_dates:
        loc = bars.index.get_loc(d)
        if loc + hold_days >= n:
            continue
        nxt = loc + 1
        exit_loc = loc + hold_days      # exit bar index (close)
        exit_loc = min(exit_loc, n - 1)

        nr7_high  = bars["high"].iat[loc]
        nr7_low   = bars["low"].iat[loc]
        o_nxt     = bars["open"].iat[nxt]
        h_nxt     = bars["high"].iat[nxt]
        l_nxt     = bars["low"].iat[nxt]
        c_exit    = bars["close"].iat[exit_loc]

        if o_nxt > nr7_high:            # gap up → long at open
            fill = o_nxt
            d_trade = +1
        elif o_nxt < nr7_low:           # gap down → short at open
            fill = o_nxt
            d_trade = -1
        elif h_nxt > nr7_high:          # intraday break upward
            fill = nr7_high
            d_trade = +1
        elif l_nxt < nr7_low:           # intraday break downward
            fill = nr7_low
            d_trade = -1
        else:
            continue                    # no breakout; skip

        ret_gross = d_trade * (c_exit - fill) / fill
        rows.append({
            "signal_date": d,
            "dir":         d_trade,
            "entry":       fill,
            "exit":        c_exit,
            "ret_gross":   ret_gross,
            "ret_net":     ret_gross - cost_bps * 1e-4,
        })

    if not rows:
        return pd.DataFrame(columns=["signal_date", "dir", "entry", "exit",
                                     "ret_gross", "ret_net"])
    return pd.DataFrame(rows).set_index("signal_date")


# ---------------------------------------------------------------------------
# Random-day control
# ---------------------------------------------------------------------------
def random_day_control(
    bars: pd.DataFrame,
    n_signals: int,
    hold_days: int = 1,
    cost_bps: float = 5.0,
    seed: int = 0,
) -> pd.DataFrame:
    """Identical trade mechanics, but on uniformly-random days (the control arm).

    The same breakout entry rule applied to ``n_signals`` randomly-drawn days
    (no look-ahead: only days with at least ``hold_days`` subsequent bars).  The
    direction is taken from the random-day breakout trigger, not planted.  This is
    the null: if NR7 adds nothing, the signal and control distributions match.
    """
    rng_gen = np.random.default_rng(seed)
    n = len(bars)
    eligible = [i for i in range(6, n - hold_days)]  # need 6 prior days + exit day
    chosen = rng_gen.choice(eligible, size=min(n_signals, len(eligible)), replace=False)
    chosen = np.sort(chosen)

    rows = []
    for loc in chosen:
        nxt = loc + 1
        exit_loc = min(loc + hold_days, n - 1)

        nr7_high = bars["high"].iat[loc]
        nr7_low  = bars["low"].iat[loc]
        o_nxt    = bars["open"].iat[nxt]
        h_nxt    = bars["high"].iat[nxt]
        l_nxt    = bars["low"].iat[nxt]
        c_exit   = bars["close"].iat[exit_loc]

        if o_nxt > nr7_high:
            fill = o_nxt; d_trade = +1
        elif o_nxt < nr7_low:
            fill = o_nxt; d_trade = -1
        elif h_nxt > nr7_high:
            fill = nr7_high; d_trade = +1
        elif l_nxt < nr7_low:
            fill = nr7_low; d_trade = -1
        else:
            continue

        ret_gross = d_trade * (c_exit - fill) / fill
        rows.append({
            "signal_date": bars.index[loc],
            "dir":         d_trade,
            "entry":       fill,
            "exit":        c_exit,
            "ret_gross":   ret_gross,
            "ret_net":     ret_gross - cost_bps * 1e-4,
        })

    if not rows:
        return pd.DataFrame(columns=["signal_date", "dir", "entry", "exit",
                                     "ret_gross", "ret_net"])
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summarize a ledger (returns-based or range-ratio-based)
# ---------------------------------------------------------------------------
def summarize(ledger: pd.DataFrame, col: str = "ret_net") -> dict:
    """Headline per-trade statistics for one ledger (breakout returns).

    Computes trade count, win-rate, mean return (bps/trade), P&L skew, and a
    Newey-West HAC t-statistic on the mean — the inference-bar number that
    decides whether the edge is distinguishable from zero.
    """
    r = ledger[col].to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    out: dict = {
        "n_trades":        int(n),
        "win_rate":        float((r > 0).mean()) if n else float("nan"),
        "mean_bps":        float(r.mean() * 1e4) if n else float("nan"),
        "sharpe_per_trade": float(r.mean() / r.std(ddof=1))
                            if n > 1 and r.std() > 0 else float("nan"),
        "skew":            float(pd.Series(r).skew()) if n > 2 else float("nan"),
        "tstat":           float("nan"),
    }
    if n > 5:
        mu = r.mean()
        e  = r - mu
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
        lrv  = float(e @ e) / n
        for k in range(1, lags + 1):
            w    = 1.0 - k / (lags + 1.0)
            lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
        se = np.sqrt(max(lrv, 0.0) / n)
        out["tstat"] = float(mu / se) if se > 0 else float("nan")
    return out


def summarize_expansion(exp_df: pd.DataFrame) -> dict:
    """Headline statistics for the range-expansion DataFrame.

    Returns mean ratio, its HAC t-stat (relative to the null ratio = 1), and
    the fraction of days with ratio > 1 (i.e., next day was wider than baseline).
    """
    if exp_df.empty:
        return {"n": 0, "mean_ratio": float("nan"), "frac_above_1": float("nan"),
                "tstat": float("nan")}
    r = exp_df["ratio"].to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    out: dict = {
        "n":           int(n),
        "mean_ratio":  float(r.mean()),
        "frac_above_1": float((r > 1.0).mean()),
        "tstat":       float("nan"),
    }
    if n > 5:
        # HAC t-stat for H0: mean ratio = 1
        e_zero = r - 1.0
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
        lrv   = float(e_zero @ e_zero) / n
        for k in range(1, lags + 1):
            w    = 1.0 - k / (lags + 1.0)
            lrv += 2.0 * w * float(e_zero[k:] @ e_zero[:-k]) / n
        se = np.sqrt(max(lrv, 0.0) / n)
        out["tstat"] = float((r.mean() - 1.0) / se) if se > 0 else float("nan")
    return out
