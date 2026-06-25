"""The Marubozu as a falsifiable mechanical rule — Study 451.

A *marubozu* ("bald / shaven head") is a candle whose real body fills almost the entire
high-low range — it has (essentially) no wicks:

* **bullish marubozu** — ``open ≈ low`` and ``close ≈ high`` (a long up-day with no shadows);
* **bearish marubozu** — ``open ≈ high`` and ``close ≈ low``.

The folklore (Steve Nison's candlestick canon, echoed by every chart-pattern site): a marubozu
shows *decisive, one-way pressure* that **continues** — a bullish marubozu is a high-probability
**buy** (the trend keeps going), a bearish one a sell.

We encode the tightest mechanical version a proponent would accept and test it honestly:

1. **Body-fill rule.** For each bar compute the body fraction ``|close-open| / (high-low)`` and
   the wick fractions. A **bullish marubozu** = an up-bar (``close > open``) whose body fills
   ``>= body_min`` of the range AND whose upper/lower wicks are each ``<= wick_max`` of the range
   (``body_min = 0.95``, ``wick_max = 0.02`` by default — the "no-wick" definition).
2. **Entry.** A bullish marubozu on the close of *t* is a long entered at the **next** close
   (one documented lag); we then measure the forward H-day return. The bar itself is fully known
   at its own close, so there is no look-ahead in the *detection*; the one lag is the execution.
3. **Controls.** (a) a **random-entry** baseline (same instrument, epoch, hold) that captures the
   tape's drift, and (b) a **body-shuffle placebo** that re-assigns the marubozu *label* to a
   random permutation of bars (keeping the same number of "marubozu days" and the price marginal),
   destroying the wickless geometry while preserving the return distribution — the honest
   "is the no-wick body actually load-bearing?" null.

No look-ahead: the marubozu is read on the close of *t*, the position is entered at the close of
*t+1*; nothing here peeks at the future.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)

BODY_MIN = 0.95   # body must fill >= 95% of the high-low range
WICK_MAX = 0.02   # each wick must be <= 2% of the range


# --------------------------------------------------------------------------- #
# Candle geometry
# --------------------------------------------------------------------------- #
def candle_parts(bars: pd.DataFrame) -> pd.DataFrame:
    """Per-bar body / upper-wick / lower-wick fractions of the high-low range.

    Returns a frame aligned to ``bars.index`` with columns ``rng`` (high-low), ``body``
    (|close-open|), ``upper`` (high-max(o,c)), ``lower`` (min(o,c)-low) and their fractions
    ``body_f``, ``upper_f``, ``lower_f`` (each in [0,1], NaN on a zero-range bar). Bullish flag
    ``up`` = close > open.
    """
    o = bars["open"].to_numpy(float)
    h = bars["high"].to_numpy(float)
    l = bars["low"].to_numpy(float)
    c = bars["close"].to_numpy(float)
    rng = h - l
    body = np.abs(c - o)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    with np.errstate(divide="ignore", invalid="ignore"):
        body_f = np.where(rng > 0, body / rng, np.nan)
        upper_f = np.where(rng > 0, upper / rng, np.nan)
        lower_f = np.where(rng > 0, lower / rng, np.nan)
    return pd.DataFrame(
        {"rng": rng, "body": body, "upper": upper, "lower": lower,
         "body_f": body_f, "upper_f": upper_f, "lower_f": lower_f,
         "up": c > o},
        index=bars.index,
    )


def is_bullish_marubozu(bars: pd.DataFrame, body_min: float = BODY_MIN,
                        wick_max: float = WICK_MAX) -> pd.Series:
    """Boolean Series: True on bars that are a *bullish* (wickless up) marubozu.

    Up-bar (close > open) whose body fills ``>= body_min`` of the range and whose upper and lower
    wicks are each ``<= wick_max`` of the range. Computed only from the bar's own OHLC.
    """
    p = candle_parts(bars)
    flag = (p["up"]
            & (p["body_f"] >= body_min)
            & (p["upper_f"] <= wick_max)
            & (p["lower_f"] <= wick_max)
            & p["body_f"].notna())
    return flag.fillna(False)


# --------------------------------------------------------------------------- #
# Entries
# --------------------------------------------------------------------------- #
def marubozu_entries(bars: pd.DataFrame, body_min: float = BODY_MIN,
                     wick_max: float = WICK_MAX) -> pd.DatetimeIndex:
    """Dates of bullish marubozu signals (read on the close of t; entered next close).

    Only the *first* bar of each consecutive run of marubozu days is kept (the signal, not every
    repeat). Entry is executed at the next close by :func:`forward_returns`.
    """
    flag = is_bullish_marubozu(bars, body_min, wick_max)
    first = flag & ~flag.shift(1, fill_value=False)
    return bars.index[first.to_numpy()]


def random_entries(close: pd.Series, n: int, warmup: int = 60, seed: int = 0) -> pd.DatetimeIndex:
    """``n`` random entry dates (after a warm-up), the drift-matched baseline."""
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


def body_shuffle_placebo(bars: pd.DataFrame, horizon: int, body_min: float = BODY_MIN,
                         wick_max: float = WICK_MAX, n_draws: int = 1000,
                         seed: int = 451) -> dict:
    """Placebo: re-assign the marubozu *label* to a random permutation of bars.

    Keeps the price path and the *number* of marubozu signals identical, but scrambles **which**
    bars carry the wickless-body label — so the no-wick geometry is destroyed while the return
    marginal is preserved. Returns the share of placebo runs whose mean forward return **beats**
    the real marubozu one — the honest "is the no-wick body load-bearing?" p-value, plus the
    observed mean.
    """
    close = bars["close"]
    obs = float(np.mean(forward_returns(close, marubozu_entries(bars, body_min, wick_max), horizon)))
    flag = is_bullish_marubozu(bars, body_min, wick_max)
    first = flag & ~flag.shift(1, fill_value=False)
    k = int(first.sum())
    idx = bars.index
    eligible = np.arange(60, len(idx))   # same warm-up as the random baseline
    if k == 0 or eligible.size < k or not np.isfinite(obs):
        return {"obs": obs, "p_value": float("nan"), "n_draws": 0}
    rng = np.random.default_rng(seed)
    beats = 0
    valid = 0
    for _ in range(n_draws):
        pick = rng.choice(eligible, size=k, replace=False)
        ent = idx[np.sort(pick)]
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
def run_experiment(bars: pd.DataFrame, body_min: float = BODY_MIN, wick_max: float = WICK_MAX,
                   cost_bps: float = 1.0, random_seed: int = 7) -> dict:
    """Run the full gauntlet on one tape: marubozu vs random-entry baseline, all horizons.

    Returns a dict keyed by horizon with the marubozu summary (gross + net), the drift-matched
    random-entry baseline, and the marubozu-minus-random delta.
    """
    close = bars["close"]
    ent = marubozu_entries(bars, body_min, wick_max)
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
