"""The Choppiness Index as a falsifiable mechanical rule — Study 477.

E.W. **Dreiss'** Choppiness Index measures how "choppy" (range-bound) vs "directional" the last
``N`` bars were:

    CI_t = 100 * log10( sum_{i=t-N+1..t} TR_i  /  (max(high) - min(low)) ) / log10(N)

where ``TR`` is the per-bar true range and ``max(high)-min(low)`` is the high-low *span* of the
window. It is bounded to roughly 0-100:

* **low CI** (≈ < 38)  → the summed bar-ranges barely exceed the window span ⇒ a near-straight
  **trend** leg;
* **high CI** (≈ > 62) → the bars thrash back and forth (sum of ranges ≫ span) ⇒ **chop**.

The folklore, repeated on every charting site: a **low** CI is a "trending" regime that
**precedes tradable momentum** (and a high CI precedes chop/whipsaw). It is *not* directional on
its own (CI is sign-blind), so the standard long-bias proxy is: on a confirmed low-CI reading,
**go long** and ride the expected continuation of the prevailing trend.

We encode the tightest mechanical version a proponent would accept and test it honestly:

1. **CI on a trailing window** — uses only bars through *t* (no future data). A "low-CI" signal
   fires when ``CI_t`` drops below a fixed threshold and was *not* low on *t-1* (the first bar
   of a low-CI episode — a regime *onset*, not every day it stays low).
2. **Entry.** Long at the **next** close (one documented lag); measure the forward H-day return.
3. **Controls.** (a) a **random-entry** baseline (same instrument, epoch, hold) that captures the
   tape's drift, and (b) a **return-shuffled placebo** that recomputes CI/entries on a surrogate
   price whose daily returns are permuted — destroying the trend-vs-chop *geometry* while keeping
   the marginal return distribution — the honest "is the CI's structure doing anything?" null.

No look-ahead: CI reads only the trailing window, the low-CI onset is read on the close of *t*,
the position is entered at the close of *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)
CI_WINDOW = 14
LOW_CI = 38.2   # the canonical "trending" threshold (Fibonacci-flavoured, as taught)


# --------------------------------------------------------------------------- #
# The Choppiness Index
# --------------------------------------------------------------------------- #
def true_range(bars: pd.DataFrame) -> pd.Series:
    """Wilder true range: max(high-low, |high-prev_close|, |low-prev_close|)."""
    high, low, close = bars["high"], bars["low"], bars["close"]
    prev = close.shift(1)
    tr = pd.concat([(high - low), (high - prev).abs(), (low - prev).abs()], axis=1).max(axis=1)
    return tr


def choppiness_index(bars: pd.DataFrame, window: int = CI_WINDOW) -> pd.Series:
    """The Choppiness Index over a trailing ``window`` of bars (0-100, trailing only).

    ``CI = 100 * log10( sum(TR) / (rolling_max(high) - rolling_min(low)) ) / log10(window)``.
    Uses only bars through ``t`` (rolling, min_periods=window) — no look-ahead.
    """
    tr = true_range(bars)
    sum_tr = tr.rolling(window, min_periods=window).sum()
    hh = bars["high"].rolling(window, min_periods=window).max()
    ll = bars["low"].rolling(window, min_periods=window).min()
    span = (hh - ll).replace(0.0, np.nan)
    ci = 100.0 * np.log10(sum_tr / span) / np.log10(window)
    return ci.clip(lower=0.0, upper=100.0)


# --------------------------------------------------------------------------- #
# Entries
# --------------------------------------------------------------------------- #
def low_ci_entries(bars: pd.DataFrame, window: int = CI_WINDOW,
                   low: float = LOW_CI) -> pd.DatetimeIndex:
    """Onsets of a low-CI ("trending") regime — the folklore 'CI is low, ride the trend' rule.

    A signal fires on the first bar whose ``CI`` drops below ``low`` after having been at or
    above it (the *onset* of a trending regime, not every day CI stays low). Entry is executed
    at the next close by :func:`forward_returns`.
    """
    ci = choppiness_index(bars, window=window)
    is_low = (ci < low) & ci.notna()
    onset = is_low & ~is_low.shift(1, fill_value=False)
    return bars.index[onset.to_numpy()]


def random_entries(close: pd.Series, n: int, window: int = CI_WINDOW,
                   seed: int = 0) -> pd.DatetimeIndex:
    """``n`` random entry dates (after the warm-up), the drift-matched baseline."""
    rng = np.random.default_rng(seed)
    valid = close.index[2 * window:]
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


def _surrogate_bars(bars: pd.DataFrame, rng) -> pd.DataFrame:
    """A surrogate OHLC tape whose daily close-to-close returns are *permuted*.

    Keeps the marginal distribution of daily returns (same set of moves) but destroys the
    serial trend-vs-chop structure the CI reads, then rebuilds a self-consistent OHLC frame
    (open = prev close; symmetric wicks scaled by the original per-bar range) so the CI is
    well-defined on the surrogate.
    """
    c = bars["close"].to_numpy(dtype=float)
    rets = c[1:] / c[:-1] - 1.0
    perm = rng.permutation(rets)
    out = np.empty_like(c)
    out[0] = c[0]
    for i in range(1, c.size):
        out[i] = out[i - 1] * (1.0 + perm[i - 1])
    open_ = np.empty_like(out)
    open_[0] = out[0]
    open_[1:] = out[:-1]
    # preserve the original window's wick scale (per-bar range fraction), reassigned by index
    rng_frac = ((bars["high"] - bars["low"]) / bars["close"]).to_numpy(dtype=float)
    wick = 0.5 * rng_frac * out
    hi = np.maximum(open_, out) + wick
    lo = np.minimum(open_, out) - wick
    return pd.DataFrame({"open": open_, "high": hi, "low": lo, "close": out}, index=bars.index)


def shuffled_returns_placebo(bars: pd.DataFrame, horizon: int, window: int = CI_WINDOW,
                             low: float = LOW_CI, n_draws: int = 500, seed: int = 477) -> dict:
    """Placebo: recompute CI/low-CI entries on a return-shuffled surrogate price.

    Permuting the daily returns keeps the price marginal but destroys the trend-vs-chop
    *geometry* the CI exploits. We read low-CI onsets on the surrogate and bank the *real*
    forward return at those dates. Returns the share of placebo runs whose mean low-CI forward
    return **beats** the real one — the honest "is the CI's structure load-bearing?" p-value,
    plus the observed (real) mean.
    """
    close = bars["close"]
    obs = float(np.mean(forward_returns(close, low_ci_entries(bars, window=window, low=low), horizon)))
    rng = np.random.default_rng(seed)
    beats = 0
    valid = 0
    for _ in range(n_draws):
        sur = _surrogate_bars(bars, rng)
        ent = low_ci_entries(sur, window=window, low=low)
        rr = forward_returns(close, ent, horizon)   # real forward returns at surrogate dates
        if rr.size == 0:
            continue
        valid += 1
        if rr.mean() >= obs:
            beats += 1
    p = (beats + 1) / (valid + 1) if valid else float("nan")
    return {"obs": obs, "p_value": float(p), "n_draws": valid}


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def run_experiment(bars: pd.DataFrame, window: int = CI_WINDOW, low: float = LOW_CI,
                   cost_bps: float = 1.0, random_seed: int = 7) -> dict:
    """Run the full gauntlet on one tape: low-CI onset vs random-entry baseline, all horizons.

    Returns a dict keyed by horizon with the low-CI summary (gross + net), the drift-matched
    random-entry baseline, and the low-CI-minus-random delta.
    """
    close = bars["close"]
    ent = low_ci_entries(bars, window=window, low=low)
    res = {"n_entries": int(len(ent)), "by_h": {}}
    for h in HORIZONS:
        g = summarize(forward_returns(close, ent, h, cost_bps=0.0))
        net = summarize(forward_returns(close, ent, h, cost_bps=cost_bps))
        rnd = summarize(forward_returns(
            close, random_entries(close, max(len(ent), 50), window=window, seed=random_seed), h))
        res["by_h"][h] = {
            "gross": g, "net": net, "random": rnd,
            "delta_bps": (g["mean_bps"] - rnd["mean_bps"])
            if np.isfinite(g["mean_bps"]) and np.isfinite(rnd["mean_bps"]) else float("nan"),
        }
    return res
