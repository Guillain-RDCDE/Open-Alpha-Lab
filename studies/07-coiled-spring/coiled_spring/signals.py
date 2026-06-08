"""The mechanised rule — turning the book's prose into a function with no discretion.

Jayesh Shah's *Trade the 20 EMA* states the setup in three steps (Chapter 5):

  1. **Form a 20-EMA pivot.** A stock trading *below* its 20-period EMA breaks above it
     and carves a high, then turns down. That high is the "20 EMA pivot".
  2. **The pullback holds the EMA.** Price pulls back from the pivot but **must not close
     below the 20 EMA** — "if it falls below, drop it and look elsewhere" (the hard rule).
  3. **Pivot breakout.** Buy the moment price breaks the pivot high, **on volume at least
     twice the prior month's average** (the volume gate is non-negotiable in the book).

Every word above is mechanisable, so we mechanise it — no eyeballing, no "looks coiled".
The book then buys *on* the breakout day; we buy at the **next open** (an honest fill a
price-taker could actually get) and place the initial stop under the breakout bar's low,
exactly as the book prescribes ("1 point below the breakout candle low"). Backtest in
:mod:`coiled_spring.backtest`.

Pivot confirmation is causal. A pivot high at bar ``p`` is a local maximum of the High
over a symmetric ``±pivot_halfwin`` window, so it can only be *known* ``pivot_halfwin``
bars later. The breakout is therefore always sought strictly after the pivot is confirmed
— there is no look-ahead.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Params:
    """The rule's parameters — all fixed from the book, none fitted."""
    ema_span: int = 20          # "20 EMA" — the whole premise
    pivot_halfwin: int = 3      # a pivot high = local max over +/-3 bars (~weekly swing)
    regime_lookback: int = 20   # the pivot must follow a spell *below* the EMA (it "broke above")
    pullback_max: int = 20      # breakout must land within ~a month of the pivot (short setup)
    vol_window: int = 20        # "prior month" average volume
    vol_mult: float = 2.0       # "at least twice" the average (the book's hard gate)


def ema(close: pd.Series, span: int = 20) -> pd.Series:
    """Standard exponential moving average (the book's '20 EMA')."""
    return close.astype(float).ewm(span=span, adjust=False).mean()


def _confirmed_pivot_highs(high: np.ndarray, halfwin: int) -> np.ndarray:
    """Boolean mask: bar p is a *strict* local High maximum over [p-halfwin, p+halfwin].

    Strict on both sides, so a flat top never registers twice. Vectorised: bar p qualifies
    iff its High exceeds the max of the ``halfwin`` bars on each side. Edge bars (without a
    full window either side) are never pivots. Known only at p+halfwin (causal; the caller
    enforces that the breakout is sought strictly later).
    """
    s = pd.Series(high)
    left = s.rolling(halfwin).max().shift(1)                  # max of the prior halfwin bars
    right = s[::-1].rolling(halfwin).max().shift(1)[::-1]      # max of the next halfwin bars
    is_piv = (s > left) & (s > right)
    return is_piv.fillna(False).to_numpy(bool)


def find_signals(frame: pd.DataFrame, params: Params = Params()) -> pd.DataFrame:
    """Locate every 20-EMA pivot breakout in one name's OHLCV history.

    Returns one row per breakout, columns:
      ``breakout_date``  — the session whose close first clears the pivot (signal day);
      ``entry_date``     — the next session (where a price-taker fills);
      ``pivot_date``     — the pivot high being broken;
      ``pivot_level``    — the pivot's High (the horizontal line being broken);
      ``entry_price``    — next-session Open (NaN if the breakout is the last bar);
      ``stop``           — the breakout bar's Low (the book's initial stop);
      ``vol_ratio``      — breakout volume / trailing ``vol_window`` mean.
    Empty DataFrame (with the right columns) if the name never sets up.
    """
    cols = ["breakout_date", "entry_date", "pivot_date", "pivot_level",
            "entry_price", "stop", "vol_ratio"]
    f = frame.sort_index()
    n = len(f)
    p = params
    if n < max(p.ema_span, p.regime_lookback) + 2 * p.pivot_halfwin + 2:
        return pd.DataFrame(columns=cols)

    close = f["Close"].to_numpy(float)
    high = f["High"].to_numpy(float)
    e = ema(f["Close"], p.ema_span).to_numpy(float)
    have_vol = "Volume" in f.columns
    vol = f["Volume"].to_numpy(float) if have_vol else np.full(n, np.nan)
    vol_ma = (
        pd.Series(vol).rolling(p.vol_window).mean().shift(1).to_numpy(float)
        if have_vol else np.full(n, np.nan)
    )
    is_piv = _confirmed_pivot_highs(high, p.pivot_halfwin)
    dates = f.index

    rows = []
    # The "active" pivot we are waiting to see broken; None when there is nothing armed.
    active_p = None  # bar index of the armed pivot

    for t in range(p.regime_lookback + p.pivot_halfwin, n):
        # 1) Confirm any new pivot that became known at bar t (i.e. formed at t-halfwin).
        cand = t - p.pivot_halfwin
        if cand >= p.regime_lookback and is_piv[cand]:
            above_at_pivot = close[cand] > e[cand]
            was_below = np.nanmin((close - e)[cand - p.regime_lookback : cand]) < 0
            if above_at_pivot and was_below:
                active_p = cand  # arm the most recent qualifying pivot

        if active_p is None:
            continue

        # The setup expires if the breakout doesn't arrive within pullback_max bars,
        # or if any close falls below the EMA during the pullback (the book's hard rule).
        if t - active_p > p.pullback_max + p.pivot_halfwin:
            active_p = None
            continue
        if close[t] < e[t]:                      # pullback broke the EMA -> discard
            active_p = None
            continue

        level = high[active_p]
        # 2) Breakout: first close strictly above the pivot level (t-1 was not yet above).
        broke = close[t] > level and close[t - 1] <= level
        if not broke:
            continue
        # 3) Volume gate: >= vol_mult x trailing-month average (skip if no volume data).
        vr = vol[t] / vol_ma[t] if (have_vol and np.isfinite(vol_ma[t]) and vol_ma[t] > 0) else np.nan
        if have_vol and (not np.isfinite(vr) or vr < p.vol_mult):
            active_p = None         # a breakout happened but failed the gate -> reset
            continue

        entry_price = float(f["Open"].iloc[t + 1]) if t + 1 < n else np.nan
        entry_date = dates[t + 1] if t + 1 < n else pd.NaT
        rows.append({
            "breakout_date": dates[t],
            "entry_date": entry_date,
            "pivot_date": dates[active_p],
            "pivot_level": float(level),
            "entry_price": entry_price,
            "stop": float(f["Low"].iloc[t]),
            "vol_ratio": float(vr) if np.isfinite(vr) else np.nan,
        })
        active_p = None              # this pivot is consumed; hunt for the next one

    return pd.DataFrame(rows, columns=cols)


def find_all_signals(frames: dict[str, pd.DataFrame], params: Params = Params()) -> pd.DataFrame:
    """Run :func:`find_signals` across a universe; one stacked table with a ``ticker`` column."""
    out = []
    for tk, f in frames.items():
        sig = find_signals(f, params)
        if len(sig):
            sig = sig.copy()
            sig.insert(0, "ticker", tk)
            out.append(sig)
    if not out:
        return pd.DataFrame(
            columns=["ticker", "breakout_date", "entry_date", "pivot_date",
                     "pivot_level", "entry_price", "stop", "vol_ratio"]
        )
    return pd.concat(out, ignore_index=True)
