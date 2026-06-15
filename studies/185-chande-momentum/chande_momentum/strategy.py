"""The strategy and its honest controls — Study 185 (Chande-Momentum).

Tushar Chande's Momentum Oscillator (CMO) is a pure-momentum normalised to [-100, +100]:

    Su(n) = sum of up-closes over the past n bars
    Sd(n) = sum of |down-closes| over the past n bars
    CMO(n) = 100 * (Su - Sd) / (Su + Sd)

Unlike RSI (which divides by total average gain) the CMO denominator is total absolute
movement, making it a symmetric measure.  Standard period: n = 14.

Two framings tested head-to-head against a **random-direction control** on the same
entry bars:

1. **Overbought/Oversold framing** — mean-reversion bet.  Signal fires when CMO *enters*
   an extreme zone (|CMO| > threshold, typically 50): oversold (CMO < -50) → buy; overbought
   (CMO > +50) → sell/short.  Hold ``hold_days`` days, exit at close.

2. **Zero-cross framing** — momentum-follow bet.  Signal fires when CMO crosses zero:
   up-cross → buy; down-cross → short.  Hold ``hold_days`` days, exit at close.

Both framings use the same forward-return engine (:func:`run_trades`) and the same
honest comparison: a coin flip on identical entry bars (:func:`random_directions`).
The verdict is "does CMO add directional information over a coin?"

No look-ahead: all indicators computed on closes up to bar *t*; trade entered at *t+1*'s
open.  Costs (``cost_bps``) deducted as a flat round-trip charge per trade.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# CMO indicator
# ---------------------------------------------------------------------------
def cmo(close: pd.Series, period: int = 14) -> pd.Series:
    """Chande Momentum Oscillator.

    CMO(period) = 100 * (Su - Sd) / (Su + Sd)

    where Su is the rolling sum of positive daily changes and Sd the rolling sum of
    absolute negative daily changes, both over ``period`` bars.  Range [-100, +100]:
    +100 all bars closed up; -100 all bars closed down.

    When Su + Sd == 0 (zero movement in the window, rare) the result is 0.  The
    first ``period`` values are NaN (insufficient history).
    """
    delta = close.diff()
    up = delta.clip(lower=0.0)
    dn = (-delta).clip(lower=0.0)

    su = up.rolling(period, min_periods=period).sum()
    sd = dn.rolling(period, min_periods=period).sum()

    total = su + sd
    raw = 100.0 * (su - sd) / total
    # Where total is exactly zero substitute 0 (flat window → no momentum).
    raw[total == 0] = 0.0
    return raw.rename("cmo")


# ---------------------------------------------------------------------------
# Signal generators — two framings
# ---------------------------------------------------------------------------
def ob_os_entries(
    cmo_series: pd.Series,
    threshold: float = 50.0,
) -> pd.DataFrame:
    """Entry signals on the *onset* of an overbought/oversold condition.

    A **buy signal** (+1) fires when CMO first drops *below* ``-threshold``
    (the bar just before was still above it → pure first-entry into oversold
    territory).  A **sell signal** (-1) fires when CMO first rises *above*
    ``+threshold`` (first-entry into overbought territory).

    One entry per episode: once inside a zone, no new signal fires until CMO
    exits the zone and re-enters it.

    Signals are known at the close of bar *t*; trades enter at *t+1*'s open.
    """
    prev = cmo_series.shift(1)
    # Oversold onset: CMO crosses below -threshold from above.
    enter_oversold = (cmo_series < -threshold) & (prev >= -threshold) & prev.notna()
    # Overbought onset: CMO crosses above +threshold from below.
    enter_overbought = (cmo_series > threshold) & (prev <= threshold) & prev.notna()

    signal = pd.Series(0, index=cmo_series.index, dtype=int)
    signal[enter_oversold] = 1    # mean-reversion bet: expect bounce up
    signal[enter_overbought] = -1  # mean-reversion bet: expect pullback

    entries = signal[signal != 0]
    return pd.DataFrame({"dir": entries.astype(int)})


def zero_cross_entries(cmo_series: pd.Series) -> pd.DataFrame:
    """Entry signals when CMO crosses zero (momentum-follow framing).

    An **up-cross** (+1) fires when CMO transitions from negative to positive —
    momentum has turned bullish.  A **down-cross** (-1) fires when CMO transitions
    from positive to negative.

    Signals are known at the close of bar *t*; trades enter at *t+1*'s open.
    """
    prev = cmo_series.shift(1)
    up_cross = (cmo_series > 0) & (prev <= 0) & prev.notna()
    dn_cross = (cmo_series < 0) & (prev >= 0) & prev.notna()

    signal = pd.Series(0, index=cmo_series.index, dtype=int)
    signal[up_cross] = 1
    signal[dn_cross] = -1

    entries = signal[signal != 0]
    return pd.DataFrame({"dir": entries.astype(int)})


def random_directions(n: int, seed: int = 0) -> np.ndarray:
    """A reproducible vector of ±1 — the control arm's coin."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 1], size=n)


# ---------------------------------------------------------------------------
# Forward-return backtest engine
# ---------------------------------------------------------------------------
def run_trades(
    bars: pd.DataFrame,
    entries: pd.DataFrame,
    hold_days: int = 5,
    cost_bps: float = 1.0,
    directions: np.ndarray | None = None,
) -> pd.DataFrame:
    """Run fixed-horizon forward-return trades and return a per-trade ledger.

    For each signal bar *t*, the trade enters at bar *t+1*'s open and exits at the
    close of bar *t + hold_days* (or the last bar, whichever comes first).  This is
    the simplest look-ahead-free measurement: does the price, after a CMO extreme or
    zero-cross, move in the predicted direction over the next ``hold_days`` bars?

    ``directions`` overrides the entry sign vector (the random-control arm passes a ±1
    vector aligned to ``entries``).  ``cost_bps`` is a flat one-way-times-two round-trip
    cost deducted from the gross return.

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
        e = i + 1  # enter at the next bar's open (no look-ahead)
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

    Returns trade count, win-rate, mean return (bps/trade), per-trade Sharpe, P&L
    skew, and a HAC Newey-West t-stat on the mean — the inference-bar number that
    decides whether the edge is distinguishable from zero.
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
