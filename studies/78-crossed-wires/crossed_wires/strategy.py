"""The strategy and its honest controls — Study 78 (Crossed-Wires).

The folk recipe: watch the MACD(12,26,9) on the daily chart.  When the MACD line
crosses *above* its signal line, go long; when it crosses *below*, go short or flat.
Ride the trend, clip the tail.  We implement it as a forward-return backtest on daily
bars and pin it against the one comparison that settles whether it is anything but a
slow dice roll: the **same entry dates with a random direction**.

Two forward-return horizons share one engine (:func:`run_trades`):

- **symmetric ATR exit** — take-profit and stop at ±1 ATR(20) from entry.  The only
  direction-fair payoff: a coin earns ≈ 0, so the cross must earn more to be real.
- **fixed-day hold** — hold for N calendar days then close.  Measures whether the
  post-cross price drift is positive over a short and medium horizon.

No look-ahead: the MACD signal-line cross is confirmed on closes up to bar *t*;
the position is entered at bar *t+1*'s open; exits are checked from *t+1* onward.

Relationship to Study 72 (Loaded-Dice, SMA(5/10) on 5-minute bars): MACD is a
*lagging* indicator built from two exponential moving averages, so it shares the same
fundamental question — "does an EMA crossover carry directional information?" — but
tested here at the daily horizon with slower EMAs.  The verdict should be compared
explicitly to Study 72's −0.39 bps/trade (HAC t = −1.12) null result.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# MACD indicators
# ---------------------------------------------------------------------------
def ema(close: pd.Series, n: int) -> pd.Series:
    """Exponential moving average with span ``n``, min_periods=n for a clean warmup."""
    return close.ewm(span=n, min_periods=n, adjust=False).mean()


def macd_lines(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Standard MACD lines.

    Returns a DataFrame with columns:
    - ``macd``     — the MACD line: EMA(fast) − EMA(slow).
    - ``signal``   — the signal line: EMA(signal) of the MACD line.
    - ``hist``     — the histogram: macd − signal.

    The first valid bar is at index ``slow + signal − 2`` (the minimum lookback for
    both EMA(slow) to exist and the signal EMA to have ``signal`` prior MACD values).
    """
    f = ema(close, fast)
    s = ema(close, slow)
    mac = f - s
    sig = mac.ewm(span=signal, min_periods=signal, adjust=False).mean()
    return pd.DataFrame(
        {"macd": mac, "signal": sig, "hist": mac - sig},
        index=close.index,
    )


def crossover_entries(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """Bars where the MACD line crosses its signal line.

    ``dir`` = +1 for an up-cross (MACD crosses above signal — long entry),
    ``dir`` = -1 for a down-cross (MACD crosses below signal — short entry).

    The cross is detected from the sign change of (macd − signal) between consecutive
    bars, known at the *close* of the crossover bar.  The trade is taken at the next
    bar's open by :func:`run_trades`.  No look-ahead.
    """
    lines = macd_lines(close, fast=fast, slow=slow, signal=signal)
    diff = lines["macd"] - lines["signal"]
    sign = np.sign(diff)
    crossed = sign.ne(sign.shift(1)) & sign.ne(0) & sign.shift(1).notna()
    out = pd.DataFrame({"dir": sign[crossed].astype(int)})
    out.index.name = close.index.name
    return out


def atr(bars: pd.DataFrame, n: int = 20) -> pd.Series:
    """Wilder-style average true range, in price units."""
    prev_close = bars["close"].shift(1)
    tr = pd.concat(
        [
            bars["high"] - bars["low"],
            (bars["high"] - prev_close).abs(),
            (bars["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def random_directions(n: int, seed: int = 0) -> np.ndarray:
    """A reproducible vector of ±1 — the control arm's coin."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 1], size=n)


# ---------------------------------------------------------------------------
# Barrier backtest (symmetric ATR exits, no overnight on 5m; daily never crosses)
# ---------------------------------------------------------------------------
def run_trades(
    bars: pd.DataFrame,
    entries: pd.DataFrame,
    tp_R: float = 1.0,
    sl_R: float = 1.0,
    atr_n: int = 20,
    cost_bps: float = 1.0,
    directions: np.ndarray | None = None,
    max_hold: int = 60,
) -> pd.DataFrame:
    """Run ATR-barrier trades and return a per-trade ledger.

    For each entry bar the trade is entered at the *next* bar's open; the risk unit is
    ``R = ATR(atr_n)`` at the entry bar.  Take-profit sits ``tp_R * R`` away, stop
    ``sl_R * R`` away.  If neither barrier is hit within ``max_hold`` bars the trade is
    closed at the last available bar's close (a surrogate end-of-period exit).  When a
    single bar straddles both barriers the **stop is assumed first** (conservative fill).

    ``directions`` overrides the entry signs (the random-control arm passes a ±1 vector
    aligned to ``entries``).  ``cost_bps`` is a one-way-times-two round-trip cost charged
    on the net return.

    Columns: ``entry_ts, dir, entry, exit, exit_reason, bars_held, ret_gross, ret_net``.
    """
    close = bars["close"]
    open_ = bars["open"]
    high = bars["high"]
    low = bars["low"]
    r_unit = atr(bars, atr_n)

    pos = {ts: i for i, ts in enumerate(bars.index)}

    dirs = (
        np.asarray(directions, dtype=int)
        if directions is not None
        else entries["dir"].to_numpy(dtype=int)
    )

    rows = []
    n_bars = len(bars)
    for sig_ts, d in zip(entries.index, dirs):
        i = pos.get(sig_ts)
        if i is None or i + 1 >= n_bars:
            continue
        e = i + 1  # enter at the next bar's open
        R = r_unit.iat[i]
        if not np.isfinite(R) or R <= 0:
            continue
        entry_px = open_.iat[e]
        tp = entry_px + d * tp_R * R
        sl = entry_px - d * sl_R * R

        exit_px = exit_reason = None
        last = e
        end = min(e + max_hold, n_bars)
        for j in range(e, end):
            hi, loo = high.iat[j], low.iat[j]
            hit_sl = (loo <= sl) if d > 0 else (hi >= sl)
            hit_tp = (hi >= tp) if d > 0 else (loo <= tp)
            if hit_sl:  # conservative: stop wins a straddling bar
                exit_px, exit_reason = sl, "sl"
                last = j
                break
            if hit_tp:
                exit_px, exit_reason = tp, "tp"
                last = j
                break
            last = j
        if exit_px is None:
            exit_px, exit_reason = close.iat[last], "eod"

        ret_gross = d * (exit_px - entry_px) / entry_px
        rows.append(
            {
                "entry_ts": bars.index[e],
                "dir": int(d),
                "entry": entry_px,
                "exit": exit_px,
                "exit_reason": exit_reason,
                "bars_held": last - e + 1,
                "ret_gross": ret_gross,
                "ret_net": ret_gross - cost_bps * 1e-4,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Fixed-day forward return (alternative to barrier exit)
# ---------------------------------------------------------------------------
def run_forward_returns(
    bars: pd.DataFrame,
    entries: pd.DataFrame,
    hold_days: int = 20,
    cost_bps: float = 1.0,
    directions: np.ndarray | None = None,
) -> pd.DataFrame:
    """Hold for exactly ``hold_days`` bars after entry, close at the bar's close.

    A simpler alternative to barrier exits for calendar-horizon analysis.  Enters at the
    close of ``hold_days``-th bar after the signal (no intrabar precision needed at the
    daily frequency).  Used to check whether the MACD cross predicts a *drifting* return
    path over a horizon that matches its indicator lag.
    """
    close = bars["close"]
    pos = {ts: i for i, ts in enumerate(bars.index)}

    dirs = (
        np.asarray(directions, dtype=int)
        if directions is not None
        else entries["dir"].to_numpy(dtype=int)
    )

    rows = []
    n_bars = len(bars)
    for sig_ts, d in zip(entries.index, dirs):
        i = pos.get(sig_ts)
        if i is None or i + 1 >= n_bars:
            continue
        e = i + 1
        exit_i = min(e + hold_days - 1, n_bars - 1)
        entry_px = close.iat[e]
        exit_px = close.iat[exit_i]
        ret_gross = d * (exit_px - entry_px) / entry_px
        rows.append(
            {
                "entry_ts": bars.index[e],
                "dir": int(d),
                "entry": entry_px,
                "exit": exit_px,
                "exit_reason": f"d{hold_days}",
                "bars_held": exit_i - e + 1,
                "ret_gross": ret_gross,
                "ret_net": ret_gross - cost_bps * 1e-4,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Trade-ledger summary
# ---------------------------------------------------------------------------
def summarize(ledger: pd.DataFrame, col: str = "ret_net") -> dict:
    """Headline per-trade statistics for one ledger.

    Returns trade count, win-rate, mean return (bps/trade), the per-trade Sharpe, the
    P&L skew, and a HAC Newey-West t-stat on the mean — the inference-bar number that
    decides whether the edge is distinguishable from zero.
    """
    if ledger.empty:
        return {k: float("nan") for k in
                ["n_trades", "win_rate", "mean_bps", "sharpe_per_trade", "skew", "tstat"]}
    r = ledger[col].to_numpy(dtype=float)
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
        # Newey-West HAC t-stat — same kernel as Study 72 for apples-to-apples
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
