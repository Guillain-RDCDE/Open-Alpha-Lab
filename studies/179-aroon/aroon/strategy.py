"""The strategy and its honest controls — Study 179 (Aroon).

Tushar Chande's Aroon indicator (1995) locates where, within a fixed look-back window,
the most recent highest-high and lowest-low fall:

    Aroon-Up(t)   = 100 * (period - bars_since_highest_high) / period
    Aroon-Down(t) = 100 * (period - bars_since_lowest_low)  / period
    Aroon-Osc(t)  = Aroon-Up(t) - Aroon-Down(t)

All three oscillate on [0, 100] (Up/Down) or [-100, 100] (Oscillator).  The canonical
period is 25 bars.  Two folk rules are tested:

1. **Crossover** — Aroon-Up crosses *above* Aroon-Down → go long next open (strong
   uptrend developing); crosses *below* → go short (downtrend).
2. **Oscillator** — Aroon-Osc crosses *above* 0 → go long; crosses *below* 0 → go short.
   (Equivalent to the crossover because Osc > 0 iff Up > Down.)

Both framings are tested because the community quotes them as distinct, but they
are mathematically identical signals — any difference in results comes from the
implementation's edge-case handling and is a robustness check.

Each signal is pinned against the **same entries with a random direction** (the honest
control), so the verdict is: does the Aroon cross carry directional information beyond
a fair coin?

No look-ahead: all indicators are computed on closes/highs/lows up to bar *t*;
the trade is entered at bar *t+1*'s open.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Aroon indicators
# ---------------------------------------------------------------------------
def aroon(bars: pd.DataFrame, period: int = 25) -> pd.DataFrame:
    """Chande's Aroon-Up, Aroon-Down, and Aroon Oscillator.

    Look-back is ``period`` bars (inclusive of the current bar), matching Chande's
    original definition.  The highest-high and lowest-low positions are found with
    ``idxmax`` / ``idxmin`` on rolling windows, then converted to "bars since" counts
    by walking the index.

    Returns a DataFrame with columns ``up``, ``down``, ``osc`` (all NaN for the first
    ``period`` rows where no full window exists yet).
    """
    # bars_since_highest_high: 0 means the current bar is the highest high (→ Aroon-Up = 100).
    # In a rolling window of (period+1) bars, position 0 is the oldest and position ``period``
    # is the current (most recent) bar.  argmax gives the position of the maximum from the
    # left (oldest end), so bars_since = period - argmax.
    hh_idx = bars["high"].rolling(period + 1, min_periods=period + 1).apply(
        lambda x: period - int(np.argmax(x)), raw=True
    )
    ll_idx = bars["low"].rolling(period + 1, min_periods=period + 1).apply(
        lambda x: period - int(np.argmin(x)), raw=True
    )
    up = 100.0 * (period - hh_idx) / period
    down = 100.0 * (period - ll_idx) / period
    osc = up - down
    return pd.DataFrame({"up": up, "down": down, "osc": osc}, index=bars.index)


# ---------------------------------------------------------------------------
# Signal generators
# ---------------------------------------------------------------------------
def crossover_entries(aroon_df: pd.DataFrame) -> pd.DataFrame:
    """Aroon-Up / Aroon-Down crossover signals.

    A **long signal** (+1) fires when Aroon-Up crosses *above* Aroon-Down (previous
    bar had Up <= Down, current bar has Up > Down).  A **short signal** (-1) fires when
    Aroon-Up crosses *below* Aroon-Down.  One signal per crossover event; no repeated
    signals while the condition persists.

    Signals are stamped at bar *t* (known at close); entries execute at *t+1*'s open.
    """
    up, down = aroon_df["up"], aroon_df["down"]
    diff = up - down
    prev_diff = diff.shift(1)
    valid = diff.notna() & prev_diff.notna()

    long_sig = valid & (prev_diff <= 0) & (diff > 0)
    short_sig = valid & (prev_diff >= 0) & (diff < 0)

    signal = pd.Series(0, index=aroon_df.index, dtype=int)
    signal[long_sig] = 1
    signal[short_sig] = -1

    entries = signal[signal != 0]
    return pd.DataFrame({"dir": entries.astype(int)})


def oscillator_entries(aroon_df: pd.DataFrame) -> pd.DataFrame:
    """Aroon Oscillator zero-line crossover signals.

    Equivalent to the Up/Down crossover because Osc = Up - Down; included to match
    the two folk framings quoted in the literature and to serve as a robustness check.
    A **long signal** (+1) fires when Osc crosses above 0; **short** (-1) below 0.
    """
    osc = aroon_df["osc"]
    prev_osc = osc.shift(1)
    valid = osc.notna() & prev_osc.notna()

    long_sig = valid & (prev_osc <= 0) & (osc > 0)
    short_sig = valid & (prev_osc >= 0) & (osc < 0)

    signal = pd.Series(0, index=aroon_df.index, dtype=int)
    signal[long_sig] = 1
    signal[short_sig] = -1

    entries = signal[signal != 0]
    return pd.DataFrame({"dir": entries.astype(int)})


def random_directions(n: int, seed: int = 0) -> np.ndarray:
    """A reproducible vector of ±1 — the control arm's fair coin."""
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
    close of bar *t + hold_days* (or the last bar, whichever comes first).  This is the
    simplest look-ahead-free measurement: does the price, after an Aroon crossover,
    move in the predicted direction over the next ``hold_days`` bars?

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
