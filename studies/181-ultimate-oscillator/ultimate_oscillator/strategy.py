"""The strategy and its honest controls — Study 181 (Ultimate-Oscillator).

Larry Williams introduced the Ultimate Oscillator in 1985 (Technical Analysis of Stocks &
Commodities) as a weighted three-period buying-pressure oscillator designed to overcome the
single-look-back problem of simpler momentum indicators.

The oscillator:
    BP(t)  = close(t) - min(low(t), close(t-1))   # buying pressure
    TR(t)  = max(high(t), close(t-1)) - min(low(t), close(t-1))   # true range
    A7     = sum(BP, 7)  / sum(TR, 7)
    A14    = sum(BP, 14) / sum(TR, 14)
    A28    = sum(BP, 28) / sum(TR, 28)
    UO(t)  = 100 * (4*A7 + 2*A14 + A28) / 7

Range: [0, 100].  Two testable rules:

1. **Threshold framing (oversold/overbought):**
   - UO falls below 30 → oversold → buy next open (mean-reversion bet).
   - UO rises above 70 → overbought → short next open (mean-reversion bet).
   Entry on the *first* bar of each extreme episode; exit after a fixed window.

2. **Divergence framing (Williams' original rule):**
   - *Bullish divergence*: price makes a new low but UO fails to confirm (higher low on UO)
     → buy next open.
   - *Bearish divergence*: price makes a new high but UO fails to confirm (lower high on UO)
     → short next open.
   We scan a 28-bar lookback window to detect divergences — the original recipe's horizon.

Both framings are tested against the same honest comparison: the **same entry bars, random
direction** (a fair coin), so the verdict is "does the UO add directional information over a
coin?". No look-ahead: signals computed on closes up to bar *t*; trade entered at *t+1*'s open.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Ultimate Oscillator indicator
# ---------------------------------------------------------------------------
def ultimate_oscillator(
    bars: pd.DataFrame,
    p1: int = 7,
    p2: int = 14,
    p3: int = 28,
) -> pd.Series:
    """Ultimate Oscillator (Williams 1985) with three look-back periods.

    Buying pressure (BP) measures how much of each bar's range is covered by the bullish
    move from the prior close.  True range (TR) anchors the denominator to the prior close
    so overnight gaps are treated correctly.  The three weighted averages give the oscillator
    a broad time-frame, which Williams argued reduced false signals from single-period
    oscillators.

    Returns a Series in [0, 100], with NaN for the first ``p3 - 1`` bars.
    """
    high = bars["high"]
    low = bars["low"]
    close = bars["close"]
    prev_close = close.shift(1)

    # Per-bar quantities (NaN on the first bar due to prev_close shift).
    bp = close - pd.concat([low, prev_close], axis=1).min(axis=1)
    tr = (
        pd.concat([high, prev_close], axis=1).max(axis=1)
        - pd.concat([low, prev_close], axis=1).min(axis=1)
    )

    # Rolling sums (min_periods = full window to avoid edge bias).
    bp1 = bp.rolling(p1, min_periods=p1).sum()
    tr1 = tr.rolling(p1, min_periods=p1).sum()
    bp2 = bp.rolling(p2, min_periods=p2).sum()
    tr2 = tr.rolling(p2, min_periods=p2).sum()
    bp3 = bp.rolling(p3, min_periods=p3).sum()
    tr3 = tr.rolling(p3, min_periods=p3).sum()

    # Guard against zero TR (flat sessions) — treat as neutral (0.5).
    a1 = bp1 / tr1.replace(0, np.nan)
    a2 = bp2 / tr2.replace(0, np.nan)
    a3 = bp3 / tr3.replace(0, np.nan)

    uo = 100.0 * (4.0 * a1 + 2.0 * a2 + a3) / 7.0
    uo.name = "ultimate_oscillator"
    return uo


# ---------------------------------------------------------------------------
# Signal generators — two framings
# ---------------------------------------------------------------------------
def threshold_entries(
    uo: pd.Series,
    oversold: float = 30.0,
    overbought: float = 70.0,
) -> pd.DataFrame:
    """Entry signals on the *first bar* where UO enters an extreme zone.

    A **buy signal** (+1) fires on the first bar where UO drops below ``oversold``
    (i.e. the previous bar was *above* the threshold).  A **sell signal** (-1) fires on
    the first bar where UO rises above ``overbought`` (the previous bar was *below*).
    One entry per episode, not once per day inside the zone.

    Signals are known at the close of bar *t*; trades enter at the *next* bar's open.
    """
    prev = uo.shift(1)
    enter_oversold = (uo < oversold) & (prev >= oversold) & prev.notna()
    enter_overbought = (uo > overbought) & (prev <= overbought) & prev.notna()

    signal = pd.Series(0, index=uo.index, dtype=int)
    signal[enter_oversold] = 1    # oversold → expect a bounce up
    signal[enter_overbought] = -1  # overbought → expect a pullback down

    entries = signal[signal != 0]
    return pd.DataFrame({"dir": entries.astype(int)})


def divergence_entries(
    bars: pd.DataFrame,
    uo: pd.Series,
    lookback: int = 28,
) -> pd.DataFrame:
    """Entry signals based on price/UO divergence — Williams' original rule.

    Scans a rolling ``lookback``-bar window on each bar *t*.

    **Bullish divergence** (+1): price close is below its lookback low (or recent low) but
    UO makes a *higher* low than at the price's lowest bar in the window — UO fails to
    confirm the price weakness → buy signal.

    **Bearish divergence** (-1): price close is above its lookback high but UO makes a
    *lower* high than at the price's highest bar in the window → sell signal.

    This is a simplified but faithful implementation of Williams' original entry trigger;
    the key invariant is *no look-ahead* (signals are known at the close of bar *t*).
    """
    close = bars["close"]
    n = len(close)
    idx = close.index
    close_arr = close.to_numpy(dtype=float)
    uo_arr = uo.to_numpy(dtype=float)

    signals = {}
    for i in range(lookback, n):
        if not np.isfinite(uo_arr[i]):
            continue
        window_close = close_arr[i - lookback : i + 1]
        window_uo = uo_arr[i - lookback : i + 1]
        if not np.all(np.isfinite(window_uo)):
            continue

        price_low_pos = int(np.argmin(window_close))
        price_high_pos = int(np.argmax(window_close))

        # Bullish divergence: price at a recent low, UO above its value at that low.
        if price_low_pos < lookback and window_uo[price_low_pos] < window_uo[-1]:
            # Price lower now; check if this is the bar after a new low was set.
            if window_close[-1] <= window_close[price_low_pos] * 1.01:
                signals[idx[i]] = 1

        # Bearish divergence: price at a recent high, UO below its value at that high.
        if price_high_pos < lookback and window_uo[price_high_pos] > window_uo[-1]:
            if window_close[-1] >= window_close[price_high_pos] * 0.99:
                signals[idx[i]] = -1

    if not signals:
        return pd.DataFrame({"dir": pd.Series(dtype=int)})
    s = pd.Series(signals, name="dir").sort_index()
    # When both signals fire on the same bar, take neither (ambiguous).
    s = s[~s.index.duplicated(keep=False)]
    return pd.DataFrame({"dir": s.astype(int)})


def random_directions(n: int, seed: int = 0) -> np.ndarray:
    """A reproducible vector of ±1 — the control arm's coin."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 1], size=n)


# ---------------------------------------------------------------------------
# Forward-return backtest
# ---------------------------------------------------------------------------
def run_trades(
    bars: pd.DataFrame,
    entries: pd.DataFrame,
    hold_days: int = 5,
    cost_bps: float = 1.0,
    directions: np.ndarray | None = None,
) -> pd.DataFrame:
    """Run fixed-horizon forward-return trades and return a per-trade ledger.

    For each signal bar *t*, the trade is entered at bar *t+1*'s open and exited at the
    close of bar *t + hold_days* (or the last bar, whichever comes first).  This directly
    tests: does the price, after an extreme UO condition, move in the predicted direction
    over the next ``hold_days`` bars?

    ``directions`` overrides the entry sign vector (the random-control arm passes a ±1
    vector aligned to ``entries``).  ``cost_bps`` is a one-way-times-two round-trip cost
    deducted from the gross return.

    Columns: ``entry_ts, dir, entry, exit_price, bars_held, ret_gross, ret_net``.
    """
    open_ = bars["open"]
    close_ = bars["close"]
    idx = bars.index
    n_bars = len(bars)
    pos = {ts: i for i, ts in enumerate(idx)}

    dirs = (
        np.asarray(directions, dtype=int)
        if directions is not None
        else entries["dir"].to_numpy(dtype=int)
    )

    rows = []
    for sig_ts, d in zip(entries.index, dirs):
        i = pos.get(sig_ts)
        if i is None or i + 1 >= n_bars:
            continue
        e = i + 1  # enter at the next bar's open
        entry_px = open_.iat[e]
        exit_idx = min(e + hold_days - 1, n_bars - 1)
        exit_px = close_.iat[exit_idx]
        n_held = exit_idx - e + 1

        ret_gross = d * (exit_px - entry_px) / entry_px
        rows.append(
            {
                "entry_ts": idx[e],
                "dir": int(d),
                "entry": entry_px,
                "exit_price": exit_px,
                "bars_held": n_held,
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

    Returns trade count, win-rate, mean return (bps/trade), per-trade Sharpe, P&L skew,
    and a HAC Newey-West t-stat on the mean — the inference-bar number that decides
    whether the edge is distinguishable from zero.
    """
    if len(ledger) == 0:
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
        # Newey-West HAC t-stat — no hard dependency on quantlab.
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
