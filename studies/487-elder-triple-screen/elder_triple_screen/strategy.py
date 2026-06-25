"""Elder's Triple Screen as a falsifiable mechanical rule — Study 487.

Dr. Alexander Elder's *Triple Screen* trading system (1985, *Trading for a Living*) layers
three filters across two timeframes to take a long only when "the tide, the wave and the
ripple" all line up:

* **Screen 1 — the tide (weekly trend).** The higher-timeframe trend, read as the slope of a
  *weekly* MACD-histogram. Trade longs only when the weekly histogram is **rising** (trend up).
* **Screen 2 — the wave (daily oscillator).** A *daily* oscillator must be **oversold against
  the up-tide** — the pullback inside the trend. We use Elder's own **Force Index** (price
  change times volume; here, with no volume, a 2-day EMA of price change) dipping below zero,
  i.e. a short-term down-move within the larger up-trend.
* **Screen 3 — the ripple (breakout entry).** A trailing buy-stop: a long triggers when the
  close clears the **prior bar's high**, confirming the pullback has turned back up.

The folklore (Elder's own teaching, repeated across every trading-education site): aligning
three screens filters out noise and yields a high-odds long. We encode the tightest mechanical
version a proponent would accept and test it honestly:

1. **Weekly trend, no look-ahead.** The weekly MACD-histogram is computed on resampled weekly
   closes, its slope read on *completed* weeks, then forward-filled to days and **shifted one
   day** so a given day only knows the trend through the prior session.
2. **Daily oscillator.** Force Index proxy (EMA of close-to-close change). Screen 2 asks it to
   be below zero (oversold pullback) while Screen 1 is up.
3. **Breakout trigger.** Screen 3 fires when ``close > prior high`` on the same bar Screens 1
   and 2 are satisfied; we keep only the *first* bar of each alignment run.
4. **Controls.** (a) a **random-entry** baseline (same instrument, epoch, hold) that captures
   the tape's drift, and (b) a **screen-scramble placebo** that circularly *shifts* the weekly
   trend filter relative to price, destroying the multi-timeframe alignment while keeping each
   screen's marginal frequency — the honest "is the timeframe alignment doing anything?" null.

No look-ahead: the weekly trend is shifted one day, the oscillator and breakout are read on the
close of *t*, the position is entered at the close of *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)


# --------------------------------------------------------------------------- #
# Indicator primitives
# --------------------------------------------------------------------------- #
def _ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def macd_hist(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    """Standard MACD histogram = MACD line − signal line."""
    macd = _ema(close, fast) - _ema(close, slow)
    sig = _ema(macd, signal)
    return macd - sig


def force_index(close: pd.Series, span: int = 2) -> pd.Series:
    """Elder's Force Index proxy: an EMA of the close-to-close change.

    The textbook Force Index is ``(close_t − close_{t-1}) * volume``; ETF total-return closes
    carry no clean contemporaneous volume, so we use the price-change EMA (the sign and the
    oversold/overbought reading are what Screen 2 needs).
    """
    return _ema(close.diff(), span)


# --------------------------------------------------------------------------- #
# Screens
# --------------------------------------------------------------------------- #
def weekly_trend_up(close: pd.Series) -> pd.Series:
    """Screen 1: True on each day the *weekly* MACD-histogram is rising (no look-ahead).

    Resample to weekly closes, compute the weekly MACD-histogram, take its slope (week-over-week
    diff > 0), forward-fill to daily, and **shift one day** so day ``t`` only knows the weekly
    trend confirmed through ``t-1``.
    """
    wk = close.resample("W-FRI").last()
    hist = macd_hist(wk)
    rising = (hist.diff() > 0)
    daily = rising.reindex(close.index, method="ffill")
    return daily.shift(1).fillna(False).astype(bool)


def triple_screen_entries(close: pd.Series, high: pd.Series | None = None,
                          lookback: int = 5) -> pd.DatetimeIndex:
    """Bars where all three Elder screens align — the long trigger.

    Screen 1: weekly MACD-histogram rising (shifted, no look-ahead) — the up-tide.
    Screen 2: daily Force-Index proxy dips below zero (an oversold pullback) *recently* — within
              the last ``lookback`` bars, the wave against the tide. This is faithful to Elder:
              the pullback sets up the trade, and the breakout (Screen 3) triggers it on a later
              bar, not necessarily the same one.
    Screen 3: close breaks above the **prior bar's high** (the ripple — a trailing buy-stop).

    A long fires on a bar where Screen 1 is up, a Screen-2 oversold reading occurred within the
    last ``lookback`` bars, and Screen 3 (breakout) confirms. Only the *first* bar of each
    consecutive alignment run is kept. Entry is executed at the next close by
    :func:`forward_returns`.
    """
    if high is None:
        high = close
    s1 = weekly_trend_up(close)
    fi = force_index(close)
    oversold = (fi < 0.0)
    # a recent pullback: an oversold reading somewhere in the last `lookback` bars (shifted so
    # the trigger bar itself need not be oversold — the breakout turns the pullback up)
    recent_pullback = oversold.shift(1).fillna(False).rolling(lookback, min_periods=1).max().astype(bool)
    prior_high = high.shift(1)
    s3 = (close > prior_high)
    align = s1 & recent_pullback & s3 & prior_high.notna()
    first = align & ~align.shift(1, fill_value=False)
    return close.index[first.to_numpy()]


def random_entries(close: pd.Series, n: int, warmup: int = 40, seed: int = 0) -> pd.DatetimeIndex:
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


def scrambled_screen_placebo(close: pd.Series, high: pd.Series, horizon: int,
                             lookback: int = 5, n_draws: int = 500, seed: int = 487) -> dict:
    """Placebo: circularly shift the weekly-trend screen relative to price.

    Keeps each screen's *marginal* (Screen 1 is True the same fraction of the time, Screens 2
    and 3 are untouched) but destroys the **timeframe alignment** by rolling the weekly-trend
    boolean by a random offset. If the triple-screen result survives the scramble, the
    multi-timeframe filter was never load-bearing. Returns the share of placebo runs whose mean
    forward return **beats** the real one, plus the observed mean.
    """
    obs = float(np.mean(forward_returns(close, triple_screen_entries(close, high, lookback), horizon)))
    s1 = weekly_trend_up(close).to_numpy()
    fi = force_index(close)
    oversold = (fi < 0.0)
    recent_pullback = oversold.shift(1).fillna(False).rolling(lookback, min_periods=1).max().astype(bool)
    prior_high = high.shift(1)
    s3 = (close > prior_high) & prior_high.notna()
    base23 = (recent_pullback & s3).to_numpy()
    idx = close.index
    n = len(idx)
    if n == 0:
        return {"obs": obs, "p_value": float("nan"), "n_draws": 0}
    rng = np.random.default_rng(seed)
    beats = 0
    valid = 0
    for _ in range(n_draws):
        shift = int(rng.integers(40, n - 40)) if n > 80 else 1
        s1_perm = np.roll(s1, shift)
        align = s1_perm & base23
        align_s = pd.Series(align, index=idx)
        first = align_s & ~align_s.shift(1, fill_value=False)
        ent = idx[first.to_numpy()]
        rr = forward_returns(close, ent, horizon)
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
def run_experiment(bars: pd.DataFrame, cost_bps: float = 1.0, random_seed: int = 7) -> dict:
    """Run the full gauntlet on one tape: triple-screen vs random-entry baseline, all horizons.

    ``bars`` is an OHLC frame (uses ``close`` and ``high``). Returns a dict keyed by horizon
    with the triple-screen summary (gross + net), the drift-matched random baseline, and the
    triple-minus-random delta.
    """
    close = bars["close"]
    high = bars["high"] if "high" in bars else close
    ent = triple_screen_entries(close, high)
    res = {"n_entries": int(len(ent)), "by_h": {}}
    for h in HORIZONS:
        g = summarize(forward_returns(close, ent, h, cost_bps=0.0))
        net = summarize(forward_returns(close, ent, h, cost_bps=cost_bps))
        rnd = summarize(forward_returns(
            close, random_entries(close, max(len(ent), 50), seed=random_seed), h))
        res["by_h"][h] = {
            "gross": g, "net": net, "random": rnd,
            "delta_bps": (g["mean_bps"] - rnd["mean_bps"])
            if np.isfinite(g["mean_bps"]) and np.isfinite(rnd["mean_bps"]) else float("nan"),
        }
    return res
