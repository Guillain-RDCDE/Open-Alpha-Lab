"""The Spinning-Top candlestick as a falsifiable mechanical rule — Study 452.

A **spinning top** is the textbook "indecision" candle:

* a **small real body** — ``|close - open|`` is a small fraction of the day's full range
  (high - low); we use the canonical ``< 25%`` threshold;
* **two long, comparable wicks** — the upper shadow (high - body_top) and the lower shadow
  (body_bot - low) are both a meaningful fraction of the range *and* roughly balanced (neither
  dwarfs the other).

The folklore (every candlestick primer, after Steve Nison's popularisation of Japanese
candles): the spinning top shows buyers and sellers fighting to a draw — **indecision** — which
then *resolves* into a directional move or a reversal. So a spinning top is a signal that
"something is about to happen".

We encode the tightest mechanical version a proponent would accept and test it honestly:

1. **Mechanical classification** — body, range, and the two wicks are computed straight from
   OHLC; a bar is a spinning top iff body/range < ``body_frac``, both wicks exceed
   ``wick_frac`` of the range, and the wick *balance* (min/max of the two wicks) ≥ ``balance``.
   No discretion, no eyeballing.
2. **No look-ahead** — the pattern is read on the close of *t* (all four prices known that
   bar); the long is entered at the **next close** (one documented lag); we then measure the
   forward H-day return. (Because "indecision" is directionless, we test the *long* leg — the
   "resolves up / reversal-up" reading — which is also the only leg that could rival the tape's
   upward drift; a short leg would lose to drift trivially.)
3. **Controls.** (a) a **random-entry** baseline (same instrument, epoch, hold) that captures
   the tape's drift, and (b) a **wick-scramble placebo** that re-pairs each bar's body with a
   *shuffled* set of wick lengths, destroying the spinning-top geometry while keeping the price
   path and the marginal wick distribution — the honest "is the small-body/balanced-wick shape
   doing anything?" null.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)


# --------------------------------------------------------------------------- #
# Candle geometry
# --------------------------------------------------------------------------- #
def candle_parts(bars: pd.DataFrame) -> pd.DataFrame:
    """Per-bar body / range / upper-wick / lower-wick (all in price units).

    Returns a frame aligned to ``bars.index`` with columns ``body``, ``rng`` (high-low),
    ``up_wick`` (high - max(open,close)), ``dn_wick`` (min(open,close) - low).
    """
    o = bars["open"].to_numpy(dtype=float)
    h = bars["high"].to_numpy(dtype=float)
    lo = bars["low"].to_numpy(dtype=float)
    c = bars["close"].to_numpy(dtype=float)
    body_top = np.maximum(o, c)
    body_bot = np.minimum(o, c)
    out = pd.DataFrame(
        {
            "body": np.abs(c - o),
            "rng": h - lo,
            "up_wick": h - body_top,
            "dn_wick": body_bot - lo,
        },
        index=bars.index,
    )
    return out


def is_spinning_top(
    bars: pd.DataFrame,
    body_frac: float = 0.25,
    wick_frac: float = 0.25,
    balance: float = 0.5,
) -> pd.Series:
    """Boolean Series: True where the bar is a mechanical spinning top.

    * ``body / range < body_frac``  (small real body — canonical 25%);
    * both ``up_wick`` and ``dn_wick`` ≥ ``wick_frac * range`` (two long wicks);
    * ``min(up_wick, dn_wick) / max(up_wick, dn_wick) ≥ balance`` (comparable wicks).

    Bars with a zero range are excluded. No look-ahead: every input is known on the bar's close.
    """
    p = candle_parts(bars)
    rng = p["rng"].to_numpy()
    body = p["body"].to_numpy()
    up = p["up_wick"].to_numpy()
    dn = p["dn_wick"].to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        small_body = (rng > 0) & (body / rng < body_frac)
        long_up = up >= wick_frac * rng
        long_dn = dn >= wick_frac * rng
        wmin = np.minimum(up, dn)
        wmax = np.maximum(up, dn)
        balanced = np.where(wmax > 0, wmin / wmax, 0.0) >= balance
    mask = small_body & long_up & long_dn & balanced
    return pd.Series(mask, index=bars.index)


# --------------------------------------------------------------------------- #
# Entries
# --------------------------------------------------------------------------- #
def spinning_top_entries(
    bars: pd.DataFrame,
    body_frac: float = 0.25,
    wick_frac: float = 0.25,
    balance: float = 0.5,
) -> pd.DatetimeIndex:
    """Bars classified as a spinning top — the 'indecision is about to resolve' rule.

    Entry is executed at the next close by :func:`forward_returns` (one documented lag).
    """
    mask = is_spinning_top(bars, body_frac=body_frac, wick_frac=wick_frac, balance=balance)
    return bars.index[mask.to_numpy()]


def random_entries(bars: pd.DataFrame, n: int, warmup: int = 60, seed: int = 0) -> pd.DatetimeIndex:
    """``n`` random entry dates (after a warm-up), the drift-matched baseline."""
    rng = np.random.default_rng(seed)
    valid = bars.index[warmup:]
    if len(valid) == 0:
        return pd.DatetimeIndex([])
    chosen = rng.choice(valid, size=min(n, len(valid)), replace=False)
    return pd.DatetimeIndex(sorted(chosen))


# --------------------------------------------------------------------------- #
# Forward-return engine
# --------------------------------------------------------------------------- #
def forward_returns(bars: pd.DataFrame, entries, horizon: int, cost_bps: float = 0.0) -> np.ndarray:
    """Forward ``horizon``-day return for each entry, entered at the *next* close (one lag).

    ``cost_bps`` is a one-way cost (charged twice: in + out) subtracted from each trade's
    return. Trades whose window overruns the tape are dropped.
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


def wick_scramble_placebo(bars: pd.DataFrame, horizon: int, body_frac: float = 0.25,
                          wick_frac: float = 0.25, balance: float = 0.5,
                          n_draws: int = 1000, seed: int = 452) -> dict:
    """Placebo: re-pair each bar's body with *shuffled* wick lengths, destroying the geometry.

    For each draw we keep every bar's open/close (so the price path and the body sizes are
    untouched) but permute the two wick lengths across bars — the small-body/balanced-wick
    *shape* of any given bar is now random, while the marginal distribution of wicks is exactly
    preserved. We re-classify spinning tops on the scrambled candles, run the same entry rule,
    and record the share of placebo runs whose mean forward return **beats** the real one — the
    honest "is the spinning-top geometry adding anything?" p-value, plus the observed mean.
    """
    real_ent = spinning_top_entries(bars, body_frac, wick_frac, balance)
    obs = float(np.mean(forward_returns(bars, real_ent, horizon))) if len(real_ent) else float("nan")
    parts = candle_parts(bars)
    up = parts["up_wick"].to_numpy(dtype=float)
    dn = parts["dn_wick"].to_numpy(dtype=float)
    o = bars["open"].to_numpy(dtype=float)
    c = bars["close"].to_numpy(dtype=float)
    body_top = np.maximum(o, c)
    body_bot = np.minimum(o, c)
    idx = bars.index
    rng = np.random.default_rng(seed)
    beats = 0
    valid = 0
    for _ in range(n_draws):
        perm = rng.permutation(len(idx))
        up_s = up[perm]
        dn_s = dn[perm]
        fake = pd.DataFrame(
            {
                "open": o,
                "close": c,
                "high": body_top + up_s,
                "low": body_bot - dn_s,
            },
            index=idx,
        )
        ent = spinning_top_entries(fake, body_frac, wick_frac, balance)
        rr = forward_returns(bars, ent, horizon)   # measure on the REAL price path
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
def run_experiment(bars: pd.DataFrame, body_frac: float = 0.25, wick_frac: float = 0.25,
                   balance: float = 0.5, cost_bps: float = 1.0, random_seed: int = 7) -> dict:
    """Run the full gauntlet on one tape: spinning-top vs random-entry baseline, all horizons.

    Returns a dict keyed by horizon with the spinning-top summary (gross + net), the
    drift-matched random-entry baseline, and the top-minus-random delta.
    """
    ent = spinning_top_entries(bars, body_frac, wick_frac, balance)
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
