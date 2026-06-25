"""Three-Inside-Up / Three-Inside-Down as a falsifiable mechanical rule — Study 453.

The **three-inside-up** is a classic three-bar candlestick reversal (Nison/Morris):

1. **Bar A** — a clear **down** candle (close < open), the last leg of a short downtrend.
2. **Bar B** — a **harami / inside** bar: its whole range sits inside Bar A's *body*
   (a smaller candle "pregnant" inside the first), hinting the down-move has stalled.
3. **Bar C** — the **confirmation**: it closes back **above Bar A's open** (i.e. past the
   top of the down body) *and* above Bar B's close, "confirming" the reversal up.

The folklore (every candlestick text and chart-pattern site): this triplet *flips the
trend* — a long after the confirming close is a high-probability bullish reversal (the
mirror three-inside-down is the bearish version). The headline thesis we put on trial is
the role of the confirmation candle: **does the third candle add edge** over the bare
harami?

We encode the tightest mechanical version a proponent would accept and test it honestly:

1. **Pattern detection** — Bars A/B/C are read on *closed* bars; the signal is complete on
   the close of the confirming bar *t*, and the position is entered at the **next** close
   (*t+1*, one documented lag). No look-ahead.
2. **Controls.** (a) a **random-entry** baseline (same instrument, same epoch, same hold)
   that captures the tape's drift — the only honest test on an upward-drifting tape; and
   (b) a **harami-only placebo** that fires on the inside bar *without* the confirming
   candle, preserving the marginal "inside-bar after a downtrend" event while removing the
   confirmation. The placebo IS the thesis test: if the confirmation candle adds edge, the
   confirmed entry must beat the harami-only entry.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)


# --------------------------------------------------------------------------- #
# Candle geometry helpers
# --------------------------------------------------------------------------- #
def _body_hi(o, c):
    return np.maximum(o, c)


def _body_lo(o, c):
    return np.minimum(o, c)


def _downtrend(close: pd.Series, lookback: int) -> pd.Series:
    """True where the close is below the close ``lookback`` bars ago (short downtrend)."""
    return close < close.shift(lookback)


def _uptrend(close: pd.Series, lookback: int) -> pd.Series:
    return close > close.shift(lookback)


# --------------------------------------------------------------------------- #
# Pattern detection
# --------------------------------------------------------------------------- #
def three_inside_up(bars: pd.DataFrame, trend_lookback: int = 5,
                    require_confirm: bool = True) -> pd.Series:
    """Boolean Series: True on the **confirming bar** of a three-inside-up.

    Bars A (i-2), B (i-1), C (i): A is a down candle after a downtrend; B is an inside
    (harami) bar contained in A's body; C confirms by closing above A's open and above B's
    close. With ``require_confirm=False`` the confirmation leg is dropped (the harami-only
    placebo): the signal then fires on bar B (the inside bar) itself, one bar earlier, with
    the entry still taken at the *next* close.

    The returned Series is aligned to ``bars.index`` and is True on the bar whose close
    completes the signal (C for confirmed, B for harami-only). Reading is on the close of
    that bar; the trade is entered next close by :func:`forward_returns`.
    """
    o, hgh, low, c = bars["open"], bars["high"], bars["low"], bars["close"]
    bhi, blo = _body_hi(o, c), _body_lo(o, c)

    # Bar A (shift 2 for confirmed, shift 1 for harami-only): a down candle in a downtrend.
    sA = 2 if require_confirm else 1
    sB = 1 if require_confirm else 0
    a_down = (c.shift(sA) < o.shift(sA))
    trend = _downtrend(c, trend_lookback).shift(sA)
    a_bhi = bhi.shift(sA)
    a_blo = blo.shift(sA)

    # Bar B: whole range inside A's body (true harami on the body).
    inside = (hgh.shift(sB) <= a_bhi) & (low.shift(sB) >= a_blo)

    sig = a_down & trend.fillna(False) & inside

    if require_confirm:
        # Bar C (current bar): closes back above A's open and above B's close.
        confirm = (c > o.shift(2)) & (c > c.shift(1))
        sig = sig & confirm

    return sig.fillna(False)


def three_inside_entries(bars: pd.DataFrame, trend_lookback: int = 5,
                         require_confirm: bool = True) -> pd.DatetimeIndex:
    """Dates of three-inside-up signals (confirmed, or harami-only if ``require_confirm`` off).

    Only the *first* bar of a consecutive run is kept (the fresh signal, not every bar the
    condition lingers). Entry is executed at the next close by :func:`forward_returns`.
    """
    sig = three_inside_up(bars, trend_lookback=trend_lookback, require_confirm=require_confirm)
    first = sig & ~sig.shift(1, fill_value=False)
    return bars.index[first.to_numpy()]


def random_entries(close: pd.Series, n: int, warmup: int = 20, seed: int = 0) -> pd.DatetimeIndex:
    """``n`` random entry dates (after the warm-up), the drift-matched baseline."""
    rng = np.random.default_rng(seed)
    valid = close.index[warmup:]
    if len(valid) == 0:
        return pd.DatetimeIndex([])
    chosen = rng.choice(valid, size=min(n, len(valid)), replace=False)
    return pd.DatetimeIndex(sorted(chosen))


# --------------------------------------------------------------------------- #
# Forward-return engine
# --------------------------------------------------------------------------- #
def forward_returns(close: pd.Series, entries, horizon: int, cost_bps: float = 0.0) -> np.ndarray:
    """Forward ``horizon``-day return for each entry, entered at the *next* close (one lag).

    ``cost_bps`` is a one-way cost (charged twice: in + out) subtracted from each trade's
    return. Trades whose window overruns the tape are dropped.
    """
    pos = {d: i for i, d in enumerate(close.index)}
    p = close.to_numpy(dtype=float)
    n = p.size
    out = []
    for d in entries:
        i = pos.get(d)
        if i is None or i + 1 + horizon >= n:
            continue
        e = i + 1                      # enter at next close
        r = p[e + horizon] / p[e] - 1.0
        out.append(r - 2.0 * cost_bps * 1e-4)
    return np.asarray(out, dtype=float)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def hac_t(x: np.ndarray) -> float:
    """Newey-West (HAC) one-sample t-stat of the mean against zero."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 6:
        return float("nan")
    mu = x.mean()
    e = x - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for kk in range(1, lags + 1):
        w = 1.0 - kk / (lags + 1.0)
        lrv += 2.0 * w * float(e[kk:] @ e[:-kk]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def summarize(returns: np.ndarray) -> dict:
    """Headline per-trade stats: count, win-rate, mean (bps), per-trade Sharpe, HAC t."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    return {
        "n": int(n),
        "win": float((r > 0).mean()) if n else float("nan"),
        "mean_bps": float(r.mean() * 1e4) if n else float("nan"),
        "sharpe": float(r.mean() / r.std(ddof=1)) if n > 1 and r.std() > 0 else float("nan"),
        "t": hac_t(r),
    }


def harami_only_placebo(bars: pd.DataFrame, horizon: int, trend_lookback: int = 5) -> dict:
    """Placebo / thesis test: fire on the bare harami WITHOUT the confirmation candle.

    Keeps the "down candle + inside bar after a downtrend" event but removes the confirming
    third candle. Returns the confirmed mean, the harami-only mean, and their difference (the
    confirmation candle's *marginal* contribution). If the confirmation adds edge, the
    confirmed entry should beat the harami-only entry; if the difference is ~0 the third
    candle is decorative.
    """
    c = bars["close"]
    conf = three_inside_entries(bars, trend_lookback=trend_lookback, require_confirm=True)
    har = three_inside_entries(bars, trend_lookback=trend_lookback, require_confirm=False)
    rc = forward_returns(c, conf, horizon)
    rh = forward_returns(c, har, horizon)
    mc = float(rc.mean()) if rc.size else float("nan")
    mh = float(rh.mean()) if rh.size else float("nan")
    return {
        "confirmed_bps": mc * 1e4 if np.isfinite(mc) else float("nan"),
        "harami_bps": mh * 1e4 if np.isfinite(mh) else float("nan"),
        "delta_bps": (mc - mh) * 1e4 if np.isfinite(mc) and np.isfinite(mh) else float("nan"),
        "n_confirmed": int(rc.size), "n_harami": int(rh.size),
    }


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def run_experiment(bars: pd.DataFrame, trend_lookback: int = 5, cost_bps: float = 1.0,
                   random_seed: int = 7) -> dict:
    """Run the full gauntlet on one tape: confirmed three-inside-up vs random baseline.

    Returns a dict keyed by horizon with the confirmed-entry summary (gross + net), the
    drift-matched random-entry baseline, the harami-only placebo, and the deltas.
    """
    c = bars["close"]
    ent = three_inside_entries(bars, trend_lookback=trend_lookback, require_confirm=True)
    res = {"n_entries": int(len(ent)), "by_h": {}}
    for h in HORIZONS:
        g = summarize(forward_returns(c, ent, h, cost_bps=0.0))
        net = summarize(forward_returns(c, ent, h, cost_bps=cost_bps))
        rnd = summarize(forward_returns(
            c, random_entries(c, max(len(ent), 50), seed=random_seed), h))
        plc = harami_only_placebo(bars, h, trend_lookback=trend_lookback)
        res["by_h"][h] = {
            "gross": g, "net": net, "random": rnd, "placebo": plc,
            "delta_bps": (g["mean_bps"] - rnd["mean_bps"])
            if np.isfinite(g["mean_bps"]) and np.isfinite(rnd["mean_bps"]) else float("nan"),
        }
    return res
