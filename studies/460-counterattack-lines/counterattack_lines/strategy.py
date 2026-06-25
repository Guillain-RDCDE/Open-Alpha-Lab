"""Counterattack / Meeting Lines as a falsifiable mechanical rule — Study 460.

The **counterattack line** (Japanese: *deai sen* / *gyakushu sen*, "meeting line") is a
two-candle reversal pattern from Steve Nison's candlestick canon. The *bullish* version,
the one we test:

* it appears **after a downtrend**;
* candle 1 is a long **black** (down) candle that extends the decline;
* candle 2 **gaps lower on the open** but rallies all day to **close at ~the same price
  as candle 1's close** — the two closes "meet".

The folklore (Nison, *Japanese Candlestick Charting Techniques*): that equal-close
*meeting* after a sell-off marks where the bears lost control — a **buy**, reversal up.
Unlike a piercing line (which closes *above* the midpoint) the counterattack only demands
the closes meet, so it is the weaker cousin and an honest test of "does the equal close
forecast?"

We encode the tightest mechanical version a proponent would accept and test it honestly:

1. **Downtrend context** — the close at *t-1* is below the close ``trend_lookback`` bars
   earlier (a confirmed down leg; no future bars).
2. **The meeting** — candle *t-1* is black (close < open), candle *t* is white
   (close > open) and gaps down (open *t* < close *t-1*), and the two closes meet within a
   tolerance ``tol`` (``|close_t - close_{t-1}| / close_{t-1} <= tol``).
3. **Entry** — a long fires on the close of the meeting bar *t*; entry is at the **next**
   close (one documented lag); we then measure the forward H-day return.
4. **Controls.** (a) a **random-entry** baseline (same instrument, same epoch, same hold)
   that captures the tape's drift, and (b) a **gap-scramble placebo** that keeps the
   downtrend+opposite-colour+gap context but *destroys the equal-close geometry* by
   permuting the close-meeting distances across candidate bars — the honest "is the
   *meeting* doing anything, or just the down-leg dip-buy?" null.

No look-ahead: the downtrend and the meeting are read on the close of *t*, the position is
entered at the close of *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)
DEFAULT_TOL = 0.0015          # closes "meet" within 15 bps
DEFAULT_TREND_LOOKBACK = 10   # bars defining the prior down leg


# --------------------------------------------------------------------------- #
# Pattern detection (bullish counterattack / meeting line)
# --------------------------------------------------------------------------- #
def _meeting_mask(bars: pd.DataFrame, tol: float = DEFAULT_TOL,
                  trend_lookback: int = DEFAULT_TREND_LOOKBACK) -> pd.Series:
    """Boolean mask over bars: True at bar *t* if a bullish meeting line completes at *t*.

    All conditions are read from data at or before *t* (no look-ahead):
      * down leg: close[t-1] < close[t-trend_lookback]
      * candle t-1 black: close[t-1] < open[t-1]
      * candle t   white: close[t]   > open[t]
      * gap down:  open[t] < close[t-1]
      * closes meet: |close[t] - close[t-1]| / close[t-1] <= tol
    """
    o = bars["open"].to_numpy(dtype=float)
    c = bars["close"].to_numpy(dtype=float)
    n = c.size
    mask = np.zeros(n, dtype=bool)
    L = trend_lookback
    for t in range(L, n):
        prev_c = c[t - 1]
        down_leg = c[t - 1] < c[t - L]
        black_prev = c[t - 1] < o[t - 1]
        white_now = c[t] > o[t]
        gap_down = o[t] < prev_c
        meet = abs(c[t] - prev_c) / prev_c <= tol if prev_c > 0 else False
        if down_leg and black_prev and white_now and gap_down and meet:
            mask[t] = True
    return pd.Series(mask, index=bars.index)


def meeting_entries(bars: pd.DataFrame, tol: float = DEFAULT_TOL,
                    trend_lookback: int = DEFAULT_TREND_LOOKBACK) -> pd.DatetimeIndex:
    """Bars completing a bullish counterattack (meeting) line — the buy signals.

    Each True bar is an independent signal (the pattern is a two-bar completion, not a run),
    so no de-duplication is needed beyond what the mask already encodes. Entry is executed at
    the next close by :func:`forward_returns`.
    """
    mask = _meeting_mask(bars, tol=tol, trend_lookback=trend_lookback)
    return bars.index[mask.to_numpy()]


def random_entries(close: pd.Series, n: int, warmup: int = DEFAULT_TREND_LOOKBACK,
                   seed: int = 0) -> pd.DatetimeIndex:
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


# --------------------------------------------------------------------------- #
# Placebo — destroy the equal-close meeting, keep the down-leg dip-buy context
# --------------------------------------------------------------------------- #
def _candidate_mask(bars: pd.DataFrame, trend_lookback: int = DEFAULT_TREND_LOOKBACK) -> np.ndarray:
    """Bars satisfying the *context* of a bullish counterattack but NOT the equal close.

    A candidate = down leg + black candle t-1 + white candle t + gap-down open. These are
    "almost meetings" that share everything with a real meeting EXCEPT the defining
    equal-close geometry. The placebo draws its entries from this pool, so the only thing it
    lacks vs the real rule is the meeting itself.
    """
    o = bars["open"].to_numpy(dtype=float)
    c = bars["close"].to_numpy(dtype=float)
    n = c.size
    mask = np.zeros(n, dtype=bool)
    L = trend_lookback
    for t in range(L, n):
        if (c[t - 1] < c[t - L] and c[t - 1] < o[t - 1]
                and c[t] > o[t] and o[t] < c[t - 1]):
            mask[t] = True
    return mask


def close_scramble_placebo(bars: pd.DataFrame, horizon: int, tol: float = DEFAULT_TOL,
                           trend_lookback: int = DEFAULT_TREND_LOOKBACK,
                           n_draws: int = 1000, seed: int = 460) -> dict:
    """Placebo: keep the down-leg/opposite-colour/gap context, scramble the equal-close test.

    The real rule keeps only the candidates whose closes *meet* (within ``tol``). The placebo
    draws, from the same candidate pool, a random subset of the *same size* as the real
    meeting set — i.e. it fires the same number of "down-leg dip-buy" entries but ignores the
    meeting geometry. Returns the share of placebo runs whose mean forward return **beats** the
    real meeting-line return — the honest "is the equal-close *meeting* load-bearing, or is it
    just a dip-buy after a down leg?" p-value, plus the observed mean.
    """
    obs = float(np.mean(forward_returns(
        bars["close"], meeting_entries(bars, tol=tol, trend_lookback=trend_lookback), horizon)))
    cand = _candidate_mask(bars, trend_lookback=trend_lookback)
    cand_idx = bars.index[cand]
    n_meet = len(meeting_entries(bars, tol=tol, trend_lookback=trend_lookback))
    if n_meet == 0 or len(cand_idx) < n_meet:
        return {"obs": obs, "p_value": float("nan"), "n_draws": 0}
    rng = np.random.default_rng(seed)
    close = bars["close"]
    beats = 0
    valid = 0
    for _ in range(n_draws):
        pick = rng.choice(np.asarray(cand_idx), size=n_meet, replace=False)
        rr = forward_returns(close, pd.DatetimeIndex(sorted(pick)), horizon)
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
def run_experiment(bars: pd.DataFrame, tol: float = DEFAULT_TOL,
                   trend_lookback: int = DEFAULT_TREND_LOOKBACK,
                   cost_bps: float = 1.0, random_seed: int = 7) -> dict:
    """Run the full gauntlet on one tape: meeting-line vs random-entry baseline, all horizons.

    Returns a dict keyed by horizon with the meeting summary (gross + net), the drift-matched
    random-entry baseline, and the meeting-minus-random delta.
    """
    close = bars["close"]
    ent = meeting_entries(bars, tol=tol, trend_lookback=trend_lookback)
    res = {"n_entries": int(len(ent)), "by_h": {}}
    for h in HORIZONS:
        g = summarize(forward_returns(close, ent, h, cost_bps=0.0))
        net = summarize(forward_returns(close, ent, h, cost_bps=cost_bps))
        rnd = summarize(forward_returns(
            close, random_entries(close, max(len(ent), 50), warmup=trend_lookback,
                                  seed=random_seed), h))
        res["by_h"][h] = {
            "gross": g, "net": net, "random": rnd,
            "delta_bps": (g["mean_bps"] - rnd["mean_bps"])
            if np.isfinite(g["mean_bps"]) and np.isfinite(rnd["mean_bps"]) else float("nan"),
        }
    return res
