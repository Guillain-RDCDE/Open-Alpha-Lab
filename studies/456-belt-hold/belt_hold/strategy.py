"""Belt-Hold (opening marubozu) as a falsifiable mechanical rule — Study 456.

A **bullish belt-hold** (a.k.a. *yorikiri* / opening marubozu) is a single candlestick that

* **opens at (or essentially at) its low** — no lower wick: the open *is* the extreme;
* **closes well up** — a long white real body that eats most of the bar's range;
* arrives **after a downtrend** — so the open-at-the-low is the seller's last gasp.

The folklore (Nison's *Japanese Candlestick Charting Techniques*, and every candlestick site):
the open at the extreme means **buyers seized control from the first tick** and the prior
down-move reverses. So a bullish belt-hold is a **buy**.

We encode the tightest mechanical version a proponent would accept and test it honestly:

1. **Belt-hold flag.** On bar *t*: ``open ≈ low`` (lower wick ≤ a small fraction of the range),
   the body is white and **tall** (close − open is a large fraction of the high − low range),
   and a **prior downtrend** is present (close fell over the last ``trend_lookback`` bars). All
   read from bars **completed by the close of t** — no future data.
2. **Entry.** A long fires on a belt-hold close; entry is at the **next** close (one documented
   lag); we then measure the forward H-day return.
3. **Controls.** (a) a **random-entry** baseline (same instrument, epoch, hold) that captures the
   tape's drift, and (b) a **shape-scramble placebo** that keeps the prior-downtrend filter and
   the *number* of signals but reassigns the belt-hold geometry to random downtrend bars,
   destroying the candle shape while preserving the marginal — the honest "is the open-at-low
   geometry doing anything?" null.

No look-ahead: the belt-hold is read on the close of *t*, the position is entered at the close
of *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)

# Default belt-hold thresholds (the tightest a proponent would accept).
WICK_FRAC = 0.10      # lower wick must be <= 10% of the bar's range (open ~ low)
BODY_FRAC = 0.60      # white body must be >= 60% of the bar's range (a tall marubozu)
TREND_LOOKBACK = 10   # prior downtrend window
TREND_DROP = 0.0      # close must be below close[t-lookback] (a real down-move)


# --------------------------------------------------------------------------- #
# Belt-hold detection
# --------------------------------------------------------------------------- #
def belt_hold_flags(bars: pd.DataFrame,
                    wick_frac: float = WICK_FRAC,
                    body_frac: float = BODY_FRAC,
                    trend_lookback: int = TREND_LOOKBACK,
                    trend_drop: float = TREND_DROP) -> pd.Series:
    """Boolean Series: True on bars that are a **bullish belt-hold** after a downtrend.

    Conditions (all from bars completed by the close of t — no look-ahead):
      * white body: ``close > open``;
      * open at the low: lower wick ``open - low <= wick_frac * (high - low)``;
      * tall body: ``close - open >= body_frac * (high - low)``;
      * prior downtrend: ``close[t] < close[t - trend_lookback] * (1 + trend_drop)``.
    """
    o = bars["open"].to_numpy(dtype=float)
    h = bars["high"].to_numpy(dtype=float)
    low = bars["low"].to_numpy(dtype=float)
    c = bars["close"].to_numpy(dtype=float)
    rng = h - low
    rng = np.where(rng <= 0, np.nan, rng)

    white = c > o
    open_at_low = (o - low) <= wick_frac * rng
    tall = (c - o) >= body_frac * rng
    flag = white & open_at_low & tall

    # prior downtrend
    n = len(c)
    down = np.zeros(n, dtype=bool)
    if n > trend_lookback:
        down[trend_lookback:] = c[trend_lookback:] < c[:-trend_lookback] * (1.0 + trend_drop)
    flag = flag & down
    return pd.Series(np.nan_to_num(flag, nan=False).astype(bool), index=bars.index)


def belt_hold_entries(bars: pd.DataFrame, **kw) -> pd.DatetimeIndex:
    """Entry dates: the bars flagged as a bullish belt-hold. Entry is at the next close."""
    flags = belt_hold_flags(bars, **kw)
    return bars.index[flags.to_numpy()]


def _downtrend_bars(bars: pd.DataFrame, trend_lookback: int = TREND_LOOKBACK,
                    trend_drop: float = TREND_DROP) -> np.ndarray:
    """Boolean array: bars that sit in a prior downtrend (the placebo's matched pool)."""
    c = bars["close"].to_numpy(dtype=float)
    n = len(c)
    down = np.zeros(n, dtype=bool)
    if n > trend_lookback:
        down[trend_lookback:] = c[trend_lookback:] < c[:-trend_lookback] * (1.0 + trend_drop)
    return down


def random_entries(bars: pd.DataFrame, n: int, warmup: int = TREND_LOOKBACK,
                   seed: int = 0) -> pd.DatetimeIndex:
    """``n`` random entry dates (after the warm-up), the drift-matched baseline."""
    rng = np.random.default_rng(seed)
    valid = bars.index[warmup:]
    if len(valid) == 0:
        return pd.DatetimeIndex([])
    chosen = rng.choice(valid, size=min(n, len(valid)), replace=False)
    return pd.DatetimeIndex(sorted(chosen))


# --------------------------------------------------------------------------- #
# Forward-return engine
# --------------------------------------------------------------------------- #
def forward_returns(bars: pd.DataFrame, entries, horizon: int,
                    cost_bps: float = 0.0) -> np.ndarray:
    """Forward ``horizon``-day return for each entry, entered at the *next* close (one lag).

    ``cost_bps`` is a one-way round-trip cost (charged twice: in + out) subtracted from each
    trade's return. Trades whose window overruns the tape are dropped.
    """
    close = bars["close"]
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


def shape_scramble_placebo(bars: pd.DataFrame, horizon: int, n_draws: int = 1000,
                           seed: int = 456, **kw) -> dict:
    """Placebo: keep the prior-downtrend filter and the signal *count*, scramble the candle shape.

    The real rule fires on belt-hold bars that are (a) in a prior downtrend and (b) carry the
    open-at-low / tall-white-body geometry. This placebo keeps requirement (a) — it draws the
    same number of entries from the **prior-downtrend** pool — but throws away requirement (b),
    the candle shape. If the open-at-low geometry carries information, the real result must sit
    far in the right tail of these shape-blind draws. Returns the share of placebo runs whose mean
    forward return **beats** the real one (the honest "is the candle shape load-bearing?" p-value).
    """
    real_ent = belt_hold_entries(bars, **kw)
    obs = float(np.mean(forward_returns(bars, real_ent, horizon))) if len(real_ent) else float("nan")
    k = len(real_ent)
    down = _downtrend_bars(bars,
                           trend_lookback=kw.get("trend_lookback", TREND_LOOKBACK),
                           trend_drop=kw.get("trend_drop", TREND_DROP))
    pool = bars.index[down]
    if k == 0 or len(pool) < k:
        return {"obs": obs, "p_value": float("nan"), "n_draws": 0}
    rng = np.random.default_rng(seed)
    beats = 0
    valid = 0
    for _ in range(n_draws):
        pick = pd.DatetimeIndex(rng.choice(pool, size=k, replace=False))
        rr = forward_returns(bars, pick, horizon)
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
def run_experiment(bars: pd.DataFrame, cost_bps: float = 1.0,
                   random_seed: int = 7, **kw) -> dict:
    """Run the full gauntlet on one tape: belt-hold vs random-entry baseline, all horizons.

    Returns a dict keyed by horizon with the belt-hold summary (gross + net), the drift-matched
    random-entry baseline, and the belt-hold-minus-random delta.
    """
    ent = belt_hold_entries(bars, **kw)
    res = {"n_entries": int(len(ent)), "by_h": {}}
    for h in HORIZONS:
        g = summarize(forward_returns(bars, ent, h, cost_bps=0.0))
        net = summarize(forward_returns(bars, ent, h, cost_bps=cost_bps))
        rnd = summarize(forward_returns(
            bars, random_entries(bars, max(len(ent), 50), seed=random_seed), h))
        res["by_h"][h] = {
            "gross": g, "net": net, "random": rnd,
            "delta_bps": (g["mean_bps"] - rnd["mean_bps"])
            if np.isfinite(g["mean_bps"]) and np.isfinite(rnd["mean_bps"]) else float("nan"),
        }
    return res
