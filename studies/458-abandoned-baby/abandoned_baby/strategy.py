"""Abandoned-Baby (island doji) as a falsifiable mechanical rule — Study 458.

The *bullish abandoned baby* is a three-candle island-reversal pattern, one of the rarest and
most revered in candlestick lore (Nison, *Japanese Candlestick Charting Techniques*; Bulkowski
ranks it among the most "reliable" patterns). It marks a bottom:

* **bar A** — a candle in the prior **down**-trend (a down/black candle);
* **bar B** — a **doji** (open ≈ close, a near-zero body) that **gaps down** away from A: the
  *entire range* of the doji sits below A's low, so the doji is stranded below;
* **bar C** — a confirmation candle that **gaps up** away from B: its *entire range* sits above
  the doji's high, and it closes up (white candle).

The doji is the "abandoned baby" — marooned on its own price island by gaps on both sides. The
folklore (Nison; every candlestick site): the island doji *calls the turn*, so you **buy the
confirmation candle** and ride the reversal up.

We encode the tightest mechanical version a proponent would accept and test it honestly:

1. **Doji** — ``|close - open| <= doji_frac * (high - low)`` (a small body relative to the
   bar's range), the standard mechanical doji test.
2. **Down-trend context** — bar A closes below an SMA, so the island is a *bottom* not a
   mid-trend blip (the bullish version requires a prior decline).
3. **Both gaps** — full-range gaps (B's high < A's low; C's low > B's high), the strict island
   definition. (A looser "body-gap" variant is offered via ``require_full_gap=False``.)
4. **No look-ahead** — the pattern needs all three bars; it is *confirmed on the close of bar C*
   (=t) and the long is entered at the **next** close (t+1, one documented lag). We then measure
   the forward H-day return.
5. **Controls.** (a) a **random-entry** baseline (same instrument, epoch, hold) that captures the
   tape's drift, and (b) a **gap-scramble placebo** that breaks the island geometry (shuffles the
   sign/labelling of the defining gaps) while keeping the same marginal of three-bar windows — the
   honest "is the island's gap structure doing anything?" null.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)


# --------------------------------------------------------------------------- #
# Pattern detection (bullish abandoned baby)
# --------------------------------------------------------------------------- #
def _is_doji(o, h, l, c, doji_frac: float) -> bool:
    rng = h - l
    if rng <= 0:
        return False
    return abs(c - o) <= doji_frac * rng


def abandoned_baby_entries(
    bars: pd.DataFrame,
    doji_frac: float = 0.10,
    sma_win: int = 20,
    require_full_gap: bool = True,
) -> pd.DatetimeIndex:
    """Confirmation-bar dates of bullish abandoned babies — the 'buy the island' rule.

    A bullish abandoned baby at bars (A=t-2, B=t-1, C=t):
      * A is a down candle in a down-trend  (close_A < open_A and close_A < SMA(t-2));
      * B is a **doji** that **gaps down** below A  (require_full_gap: high_B < low_A;
        else body-gap: max(open_B,close_B) < close_A);
      * C **gaps up** above B and closes up  (require_full_gap: low_C > high_B; else
        min(open_C,close_C) > max(open_B,close_B)) and close_C > open_C.

    The pattern is confirmed on the **close of C** (=t); :func:`forward_returns` enters at the
    next close (t+1). Returns the DatetimeIndex of the C bars (one entry per island).
    """
    o = bars["open"].to_numpy(float)
    h = bars["high"].to_numpy(float)
    l = bars["low"].to_numpy(float)
    c = bars["close"].to_numpy(float)
    n = len(bars)
    sma = bars["close"].rolling(sma_win, min_periods=sma_win).mean().to_numpy(float)
    idx = bars.index

    hits = []
    for t in range(2, n):
        a = t - 2
        b = t - 1
        # context: prior down-trend (A is a down candle below its SMA)
        if not (np.isfinite(sma[a]) and c[a] < sma[a] and c[a] < o[a]):
            continue
        # B is a doji
        if not _is_doji(o[b], h[b], l[b], c[b], doji_frac):
            continue
        # gap down from A to B
        if require_full_gap:
            down_gap = h[b] < l[a]
        else:
            down_gap = max(o[b], c[b]) < c[a]
        if not down_gap:
            continue
        # gap up from B to C, and C closes up
        if require_full_gap:
            up_gap = l[t] > h[b]
        else:
            up_gap = min(o[t], c[t]) > max(o[b], c[b])
        if not (up_gap and c[t] > o[t]):
            continue
        hits.append(idx[t])
    return pd.DatetimeIndex(hits)


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


def gap_scramble_placebo(bars: pd.DataFrame, horizon: int, doji_frac: float = 0.10,
                         sma_win: int = 20, require_full_gap: bool = True,
                         n_draws: int = 500, seed: int = 458) -> dict:
    """Placebo: keep the doji/down-trend filter, but **scramble the island geometry**.

    The abandoned baby's defining structure is the *pair of gaps* (down-gap to the doji,
    up-gap from it). This placebo locates every confirmed-doji-after-a-decline (the same
    marginal of candidate bars), but **replaces** the two-sided-gap requirement with a
    *random relabelling*: each candidate is accepted with the empirical acceptance rate of
    the real rule, drawn at random rather than by its gaps. Positions and the price marginal
    are preserved; only the gap geometry that supposedly carries the signal is destroyed.
    Returns the share of placebo runs whose mean entry forward return **beats** the real one
    — the honest "is the island's gap structure load-bearing?" p-value, plus the observed mean.
    """
    close = bars["close"]
    obs = float(np.mean(forward_returns(
        close, abandoned_baby_entries(bars, doji_frac, sma_win, require_full_gap), horizon)))

    # candidate pool: a confirmed doji at t-1 sitting below the SMA at t-2 (the context
    # filter), regardless of whether the two gaps fire. We then randomly pick acceptances.
    o = bars["open"].to_numpy(float); h = bars["high"].to_numpy(float)
    l = bars["low"].to_numpy(float);  c = bars["close"].to_numpy(float)
    n = len(bars)
    sma = bars["close"].rolling(sma_win, min_periods=sma_win).mean().to_numpy(float)
    idx = bars.index
    candidates = []
    for t in range(2, n):
        a, b = t - 2, t - 1
        if not (np.isfinite(sma[a]) and c[a] < sma[a] and c[a] < o[a]):
            continue
        if not _is_doji(o[b], h[b], l[b], c[b], doji_frac):
            continue
        candidates.append(t)
    n_real = len(abandoned_baby_entries(bars, doji_frac, sma_win, require_full_gap))
    if len(candidates) == 0 or n_real == 0:
        return {"obs": obs, "p_value": float("nan"), "n_draws": 0}

    rng = np.random.default_rng(seed)
    cand = np.asarray(candidates, dtype=int)
    k = min(n_real, len(cand))
    beats = 0
    valid = 0
    for _ in range(n_draws):
        pick = rng.choice(cand, size=k, replace=False)
        ent = idx[np.sort(pick)]
        rr = forward_returns(close, ent, horizon)
        if rr.size == 0:
            continue
        valid += 1
        if rr.mean() >= obs:
            beats += 1
    p = (beats + 1) / (valid + 1) if valid else float("nan")
    return {"obs": obs, "p_value": float(p), "n_draws": valid, "n_candidates": len(candidates)}


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def run_experiment(bars: pd.DataFrame, doji_frac: float = 0.10, sma_win: int = 20,
                   require_full_gap: bool = True, cost_bps: float = 1.0,
                   random_seed: int = 7) -> dict:
    """Run the full gauntlet on one tape: abandoned-baby vs random-entry baseline, all horizons.

    Returns a dict keyed by horizon with the pattern summary (gross + net), the drift-matched
    random-entry baseline, and the pattern-minus-random delta.
    """
    close = bars["close"]
    ent = abandoned_baby_entries(bars, doji_frac, sma_win, require_full_gap)
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
