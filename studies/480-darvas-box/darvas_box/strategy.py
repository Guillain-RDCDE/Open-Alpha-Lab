"""Darvas Box as a falsifiable mechanical rule — Study 480.

Nicolas Darvas' "box theory" (from *How I Made $2,000,000 in the Stock Market*, 1960) draws a
**box** around a consolidation:

* a stock makes a **new high**;
* it then fails to make a higher high for a few sessions — that recent high becomes the **box top**;
* the subsequent pullback low (that holds) becomes the **box bottom**;
* a **close above the box top** is the breakout — Darvas **buys**, places a stop just below the
  box bottom (here an ATR/box stop), and rides the next box up.

The folklore (Darvas' own teaching, echoed by every momentum-trading write-up): *the breakout
forecasts continuation* — a close above the box top is a high-probability long.

We encode the tightest mechanical version a proponent would accept and test it honestly:

1. **Box top is a confirmed trailing high** — the highest close over a trailing ``lookback``
   window. To call it a *box* (a consolidation, not a runaway) we require the close to have sat
   **below that top for at least ``min_box`` consecutive bars** before the breakout — so the box
   has formed. All quantities are trailing only; no future bars leak in.
2. **Box bottom** — the lowest low over the same trailing window (used for the ATR/box stop and
   the box height, not for entry timing).
3. **Breakout entry** — a long fires when the close pierces **above the box top** for the first
   time after the consolidation. Entry is at the **next** close (one documented lag); we then
   measure the forward H-day return.
4. **Controls.** (a) a **random-entry** baseline (same instrument, same epoch, same hold) that
   captures the tape's drift, and (b) a **shuffled-box placebo** that fires the breakout rule on a
   permutation of the breakout *dates*, destroying the box geometry while keeping the same number
   and marginal of entries — the honest "is the box's geometry doing anything?" null.

No look-ahead: the box top/bottom use only trailing bars, the breakout is read on the close of
*t*, the position is entered at the close of *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)


# --------------------------------------------------------------------------- #
# Box construction + breakout detection
# --------------------------------------------------------------------------- #
def box_levels(bars: pd.DataFrame, lookback: int = 20) -> pd.DataFrame:
    """Trailing box top / box bottom at each bar (uses only bars strictly before ``t``).

    ``box_top`` is the highest close over the prior ``lookback`` bars; ``box_bottom`` is the
    lowest low over the same window. Both are shifted by one bar so the level at ``t`` never
    includes ``t`` itself — the breakout test then compares today's close to a *trailing* top.
    Returns a DataFrame aligned to ``bars.index`` with columns ``box_top``, ``box_bottom``.
    """
    close = bars["close"]
    low = bars["low"]
    box_top = close.rolling(lookback).max().shift(1)
    box_bottom = low.rolling(lookback).min().shift(1)
    return pd.DataFrame({"box_top": box_top, "box_bottom": box_bottom}, index=bars.index)


def atr(bars: pd.DataFrame, n: int = 14) -> pd.Series:
    """Average true range (Wilder), used for the ATR/box stop sizing."""
    h, l, c = bars["high"], bars["low"], bars["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(n).mean()


def breakout_entries(bars: pd.DataFrame, lookback: int = 20,
                     min_box: int = 5) -> pd.DatetimeIndex:
    """Bars whose close breaks **above** the trailing box top — Darvas' 'buy the breakout' rule.

    A breakout is only a *box* breakout if the close had sat **below** the box top for at least
    ``min_box`` consecutive prior bars (the consolidation must have formed). Only the *first* bar
    of each breakout run is kept (the breakout, not every day price stays above the box). Entry is
    executed at the next close by :func:`forward_returns`.
    """
    close = bars["close"]
    lv = box_levels(bars, lookback=lookback)
    top = lv["box_top"]
    above = (close > top) & top.notna()
    # require the close to have sat below the box top for >= min_box consecutive prior bars
    # (the consolidation must have formed before the breakout)
    below_prev = (~above).shift(1, fill_value=True)
    consec = below_prev.rolling(min_box).sum()
    box_formed = consec >= min_box
    first = above & ~above.shift(1, fill_value=False) & box_formed
    return bars.index[first.to_numpy()]


def random_entries(close: pd.Series, n: int, lookback: int = 20, seed: int = 0) -> pd.DatetimeIndex:
    """``n`` random entry dates (after the warm-up), the drift-matched baseline."""
    rng = np.random.default_rng(seed)
    valid = close.index[2 * lookback:]
    if len(valid) == 0:
        return pd.DatetimeIndex([])
    chosen = rng.choice(valid, size=min(n, len(valid)), replace=False)
    return pd.DatetimeIndex(sorted(chosen))


# --------------------------------------------------------------------------- #
# Forward-return engine
# --------------------------------------------------------------------------- #
def forward_returns(close: pd.Series, entries, horizon: int, cost_bps: float = 0.0) -> np.ndarray:
    """Forward ``horizon``-day return for each entry, entered at the *next* close (one lag).

    ``cost_bps`` is a one-way cost (charged twice: in + out) subtracted from each trade's return.
    Trades whose window overruns the tape are dropped.
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


def shuffled_box_placebo(bars: pd.DataFrame, horizon: int, lookback: int = 20,
                         min_box: int = 5, n_draws: int = 1000, seed: int = 480) -> dict:
    """Placebo: fire the same *number* of entries on **random** dates, destroying the box geometry.

    Keeps the marginal (same instrument, epoch, hold) and the same trade *count* as the real
    breakout rule, but scatters the entry dates at random — so the specific box-top breakout
    timing is destroyed while the price marginal is preserved. Returns the share of placebo runs
    whose mean forward return **beats** the real one — the honest "is the box's geometry adding
    anything?" p-value, plus the observed mean. (Identical in spirit to the random-entry baseline,
    but matched trade-for-trade and bootstrapped for a p-value on the geometry.)
    """
    close = bars["close"]
    ent = breakout_entries(bars, lookback=lookback, min_box=min_box)
    obs = float(np.mean(forward_returns(close, ent, horizon))) if len(ent) else float("nan")
    if not np.isfinite(obs) or len(ent) == 0:
        return {"obs": obs, "p_value": float("nan"), "n_draws": 0}
    rng = np.random.default_rng(seed)
    valid = close.index[2 * lookback:]
    beats = 0
    valid_draws = 0
    for _ in range(n_draws):
        chosen = rng.choice(valid, size=min(len(ent), len(valid)), replace=False)
        rr = forward_returns(close, pd.DatetimeIndex(chosen), horizon)
        if rr.size == 0:
            continue
        valid_draws += 1
        if rr.mean() >= obs:
            beats += 1
    p = (beats + 1) / (valid_draws + 1) if valid_draws else float("nan")
    return {"obs": obs, "p_value": float(p), "n_draws": valid_draws}


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def run_experiment(bars: pd.DataFrame, lookback: int = 20, min_box: int = 5,
                   cost_bps: float = 1.0, random_seed: int = 7) -> dict:
    """Run the full gauntlet on one tape: box breakout vs random-entry baseline, all horizons.

    Returns a dict keyed by horizon with the breakout summary (gross + net), the drift-matched
    random-entry baseline, and the breakout-minus-random delta.
    """
    close = bars["close"]
    ent = breakout_entries(bars, lookback=lookback, min_box=min_box)
    res = {"n_entries": int(len(ent)), "by_h": {}}
    for h in HORIZONS:
        g = summarize(forward_returns(close, ent, h, cost_bps=0.0))
        net = summarize(forward_returns(close, ent, h, cost_bps=cost_bps))
        rnd = summarize(forward_returns(
            close, random_entries(close, max(len(ent), 50), lookback=lookback, seed=random_seed), h))
        res["by_h"][h] = {
            "gross": g, "net": net, "random": rnd,
            "delta_bps": (g["mean_bps"] - rnd["mean_bps"])
            if np.isfinite(g["mean_bps"]) and np.isfinite(rnd["mean_bps"]) else float("nan"),
        }
    return res
