"""The strategy and its honest controls — Study 184 (Williams-Fractals).

Bill Williams' fractal is a 5-bar swing-extreme pattern:

- **Bearish fractal**: bar *t* whose high is strictly greater than the highs of bars
  *t-2*, *t-1*, *t+1*, *t+2*  →  a local 5-bar swing high.
- **Bullish fractal**: bar *t* whose low is strictly less than the lows of bars
  *t-2*, *t-1*, *t+1*, *t+2*  →  a local 5-bar swing low.

Because the pattern requires two *right* confirmation bars, the fractal at bar *t* is
*known* only at the close of bar *t+2* and a trade can be entered at bar *t+3*'s open
— two days of unavoidable lag from the extreme bar itself.

Two framings are tested:

1. **Reversal framing** — the original Williams recipe for the Alligator system: *fade*
   the swing extreme.  A bearish fractal (swing high) signals a *short* entry; a
   bullish fractal (swing low) signals a *long* entry.  Theory: price reverts after a
   confirmed local extreme.

2. **Breakout framing** — the more common retail use of fractals: *go with* the next
   breakout.  After a bullish fractal is confirmed, a *buy* signal fires the first time
   the close breaks above the fractal high.  After a bearish fractal, a *sell* signal
   fires when the close breaks below the fractal low.  Theory: the fractal marks support
   or resistance that, once violated, triggers a breakout move.

Both framings are raced against an honest control: **random-direction entries on the
same bars**, so the verdict is "does the fractal add directional information over a
coin?"

No look-ahead: fractals detected on bar *t*, confirmed at close of *t+2*, trade entered
at open of *t+3*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Fractal detection
# ---------------------------------------------------------------------------
def detect_fractals(bars: pd.DataFrame) -> pd.DataFrame:
    """Detect 5-bar Williams fractals on a daily OHLCV tape.

    A bearish fractal at bar *t* (column ``bearish``): bar *t*'s high is strictly
    greater than the highs of bars *t-2*, *t-1*, *t+1*, *t+2*.
    A bullish fractal at bar *t* (column ``bullish``): bar *t*'s low is strictly
    less than the lows of bars *t-2*, *t-1*, *t+1*, *t+2*.

    Only bar *t*'s extremes are required to be *strict* maximums/minimums; ties on
    the surrounding bars do not fire.  Returns a boolean DataFrame indexed like
    ``bars``; the first and last two bars are always False (pattern incomplete).

    Because the pattern uses two *future* bars (t+1, t+2) for confirmation, the
    fractal at *t* can only be *known* at close of *t+2* — no look-ahead here:
    callers must shift by 2 before trading.
    """
    high = bars["high"]
    low = bars["low"]

    n = len(bars)
    bearish = np.zeros(n, dtype=bool)
    bullish = np.zeros(n, dtype=bool)

    h = high.to_numpy()
    l = low.to_numpy()

    for i in range(2, n - 2):
        # Bearish: centre bar has the highest high of the 5-bar window (strict on centre)
        if (h[i] > h[i - 2] and h[i] > h[i - 1] and h[i] > h[i + 1] and h[i] > h[i + 2]):
            bearish[i] = True
        # Bullish: centre bar has the lowest low of the 5-bar window (strict on centre)
        if (l[i] < l[i - 2] and l[i] < l[i - 1] and l[i] < l[i + 1] and l[i] < l[i + 2]):
            bullish[i] = True

    return pd.DataFrame(
        {"bearish": bearish, "bullish": bullish},
        index=bars.index,
    )


# ---------------------------------------------------------------------------
# Signal generators — two framings
# ---------------------------------------------------------------------------
def reversal_entries(fractals: pd.DataFrame) -> pd.DataFrame:
    """Reversal framing: trade *against* the swing extreme.

    A bearish fractal (5-bar swing high) is confirmed at close of bar *t+2*; we
    expect mean reversion → **short** at the open of *t+3* (dir = -1).
    A bullish fractal (5-bar swing low) is confirmed at close of *t+2*; we
    expect mean reversion → **long** at the open of *t+3* (dir = +1).

    ``fractals`` is indexed by the *centre* bar date.  This function shifts the
    signal forward by 2 bars so the index represents the confirmation date (the
    last bar of the pattern), and the caller enters at the *next* bar's open —
    giving exactly 3-bar lag from the extreme to the entry.

    Returns a DataFrame with column ``dir`` (+1 long, -1 short) indexed by the
    confirmation date (bar *t+2* of the pattern).
    """
    # bearish fractal at t → short; bullish fractal at t → long
    # shift(2): fractal_date=t, confirmation_date=t+2
    bear_conf = fractals["bearish"].shift(2, fill_value=False)
    bull_conf = fractals["bullish"].shift(2, fill_value=False)

    signal = pd.Series(0, index=fractals.index, dtype=int)
    signal[bear_conf] = -1  # short after swing high
    signal[bull_conf] = 1   # long after swing low

    # Keep only non-zero (and deduplicate same-bar bear+bull by preferring bull)
    sig = signal[signal != 0]
    return pd.DataFrame({"dir": sig.astype(int)})


def breakout_entries(bars: pd.DataFrame, fractals: pd.DataFrame) -> pd.DataFrame:
    """Breakout framing: enter on the first close that breaks a fractal level.

    After a bullish fractal (swing low) confirmed at *t+2*, watch for the *first*
    subsequent close that exceeds the fractal high of bar *t* → **long** (dir = +1),
    as price escaping the swing low resistance signals upward continuation.

    After a bearish fractal (swing high) confirmed at *t+2*, watch for the *first*
    subsequent close below the fractal low of bar *t* → **short** (dir = -1).

    Each fractal level is active for at most ``MAX_BARS_ACTIVE`` bars.  Only the
    *first* breakout per fractal fires.  Returns a DataFrame with column ``dir``
    indexed by the breakout bar (signal known at close; trade enters next open).
    """
    MAX_BARS_ACTIVE = 20

    close = bars["close"].to_numpy()
    high = bars["high"].to_numpy()
    low = bars["low"].to_numpy()
    idx = bars.index

    # Build lists of (confirmation_bar_pos, fractal_high, fractal_low, kind)
    # fractals at centre bar i, confirmed at i+2 → confirmation_pos = i+2
    pending_bull: list[tuple[int, float]] = []  # (conf_pos, fractal_high_level)
    pending_bear: list[tuple[int, float]] = []  # (conf_pos, fractal_low_level)

    n = len(bars)
    bear_arr = fractals["bearish"].to_numpy()
    bull_arr = fractals["bullish"].to_numpy()

    # Pre-build lists of (centre_bar_pos, level)
    bull_events: list[tuple[int, float]] = []
    bear_events: list[tuple[int, float]] = []
    for i in range(n - 2):
        if bull_arr[i]:
            bull_events.append((i + 2, high[i]))   # buy breakout above fractal high
        if bear_arr[i]:
            bear_events.append((i + 2, low[i]))    # sell breakout below fractal low

    entries: dict[int, int] = {}  # bar_pos → dir (first-wins if multiple)
    bull_ptr = 0
    bear_ptr = 0

    for j in range(n):
        # Activate new fractals confirmed at bar j
        while bull_ptr < len(bull_events) and bull_events[bull_ptr][0] == j:
            pending_bull.append((j, bull_events[bull_ptr][1]))
            bull_ptr += 1
        while bear_ptr < len(bear_events) and bear_events[bear_ptr][0] == j:
            pending_bear.append((j, bear_events[bear_ptr][1]))
            bear_ptr += 1

        # Check bullish breakout (close > fractal high)
        new_bull: list[tuple[int, float]] = []
        for (conf_pos, level) in pending_bull:
            if j <= conf_pos:  # still in the confirmation bar itself
                new_bull.append((conf_pos, level))
                continue
            age = j - conf_pos
            if age > MAX_BARS_ACTIVE:
                continue  # expired
            if close[j] > level:
                if j not in entries:
                    entries[j] = 1  # first breakout fires
                # consumed — drop
            else:
                new_bull.append((conf_pos, level))
        pending_bull = new_bull

        # Check bearish breakout (close < fractal low)
        new_bear: list[tuple[int, float]] = []
        for (conf_pos, level) in pending_bear:
            if j <= conf_pos:
                new_bear.append((conf_pos, level))
                continue
            age = j - conf_pos
            if age > MAX_BARS_ACTIVE:
                continue
            if close[j] < level:
                if j not in entries and j not in entries:
                    entries[j] = -1
            else:
                new_bear.append((conf_pos, level))
        pending_bear = new_bear

    if not entries:
        return pd.DataFrame(columns=["dir"])

    sig_idx = [idx[j] for j in sorted(entries)]
    sig_dir = [entries[j] for j in sorted(entries)]
    return pd.DataFrame({"dir": sig_dir}, index=pd.DatetimeIndex(sig_idx, name="date"))


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

    For each signal bar *t* (the confirmation date / breakout date), the trade is
    entered at bar *t+1*'s open and exited at the close of bar *t + hold_days* (or
    the last bar if it comes first).

    ``directions`` overrides the entry sign vector (the random-control arm passes a
    ±1 vector aligned to ``entries``).  ``cost_bps`` is a round-trip cost charged in
    basis points and deducted from the gross return.

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
