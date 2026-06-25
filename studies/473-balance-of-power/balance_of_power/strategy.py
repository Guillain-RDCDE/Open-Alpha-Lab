"""Balance of Power as a falsifiable mechanical rule — Study 473.

Igor Livshin's **Balance of Power** (BOP) reads each bar as a tug-of-war between buyers and
sellers, scaled by the bar's own range:

    BOP_t = (close_t - open_t) / (high_t - low_t)

A green bar that closes near its high gives BOP near +1 (buyers in control); a red bar near
its low gives BOP near -1 (sellers in control). The raw series is noisy, so Livshin smooths it
with a moving average (we use a simple ``smooth``-day MA). The folklore — Livshin's own claim,
echoed across charting sites — is that **smoothed BOP leads price**: when buyers gain the upper
hand (smoothed BOP turns positive, crossing up through zero) a rise follows.

We encode the tightest mechanical version a proponent would accept and test it honestly:

1. **Smoothed BOP** — a trailing ``smooth``-day average of the per-bar body balance. It uses
   only bars up to and including *t*, so it is causal (no look-ahead).
2. **Zero-cross entry** — a long fires when smoothed BOP crosses **up** through zero (negative
   on *t-1*, non-negative on *t*): buyers have just taken control. Entry is at the **next**
   close (one documented lag); we then measure the forward H-day return.
3. **Controls.** (a) a **random-entry** baseline (same instrument, same epoch, same hold) that
   captures the tape's drift, and (b) a **sign-scramble placebo** that permutes the per-bar BOP
   values before smoothing, destroying the temporal structure the cross depends on while keeping
   the marginal distribution — the honest "is the BOP series' ordering load-bearing?" null.

No look-ahead: smoothed BOP is a trailing average, the up-cross is read on the close of *t*, the
position is entered at the close of *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)
SMOOTH = 14


# --------------------------------------------------------------------------- #
# The indicator
# --------------------------------------------------------------------------- #
def raw_bop(bars: pd.DataFrame) -> pd.Series:
    """Per-bar Balance of Power: (close - open) / (high - low), range-guarded.

    Bars with high == low (zero range) map to 0 (no information). Returns a Series aligned to
    ``bars.index``, bounded in [-1, +1].
    """
    rng = (bars["high"] - bars["low"]).to_numpy(dtype=float)
    body = (bars["close"] - bars["open"]).to_numpy(dtype=float)
    out = np.where(rng > 0, body / np.where(rng > 0, rng, 1.0), 0.0)
    return pd.Series(np.clip(out, -1.0, 1.0), index=bars.index)


def smoothed_bop(bars: pd.DataFrame, smooth: int = SMOOTH) -> pd.Series:
    """Smoothed Balance of Power: trailing ``smooth``-day simple MA of the per-bar BOP.

    Causal (uses only bars up to *t*). NaN until ``smooth`` bars exist.
    """
    return raw_bop(bars).rolling(smooth, min_periods=smooth).mean()


# --------------------------------------------------------------------------- #
# Entries
# --------------------------------------------------------------------------- #
def bop_cross_entries(bars: pd.DataFrame, smooth: int = SMOOTH) -> pd.DatetimeIndex:
    """Bars where smoothed BOP crosses **up** through zero — the 'buyers take over' rule.

    Negative on *t-1*, non-negative on *t*. Read on the close of *t*; entry is executed at the
    next close by :func:`forward_returns`.
    """
    sb = smoothed_bop(bars, smooth=smooth)
    prev = sb.shift(1)
    mask = (prev < 0.0) & (sb >= 0.0) & sb.notna() & prev.notna()
    return bars.index[mask.to_numpy()]


def random_entries(bars: pd.DataFrame, n: int, smooth: int = SMOOTH,
                   seed: int = 0) -> pd.DatetimeIndex:
    """``n`` random entry dates (after the smoothing warm-up), the drift-matched baseline."""
    rng = np.random.default_rng(seed)
    valid = bars.index[smooth:]
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


def scramble_placebo(bars: pd.DataFrame, horizon: int, smooth: int = SMOOTH,
                     n_draws: int = 1000, seed: int = 473) -> dict:
    """Placebo: permute the per-bar BOP values before smoothing, destroying the time order.

    Keeps the BOP **marginal** (same set of body-balance values) but scrambles their sequence,
    so the smoothed series and its zero-crosses become temporally meaningless. Returns the share
    of placebo runs whose mean cross-entry forward return **beats** the real one — the honest
    "does the BOP series' ordering carry information?" p-value, plus the observed mean.
    """
    obs = float(np.mean(forward_returns(
        bars["close"], bop_cross_entries(bars, smooth=smooth), horizon)))
    rb = raw_bop(bars).to_numpy(dtype=float)
    idx = bars.index
    close = bars["close"]
    rng = np.random.default_rng(seed)
    beats = 0
    valid = 0
    for _ in range(n_draws):
        perm = rng.permutation(rb)
        sb = pd.Series(perm, index=idx).rolling(smooth, min_periods=smooth).mean()
        prev = sb.shift(1)
        mask = (prev < 0.0) & (sb >= 0.0) & sb.notna() & prev.notna()
        ent = idx[mask.to_numpy()]
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
def run_experiment(bars: pd.DataFrame, smooth: int = SMOOTH, cost_bps: float = 1.0,
                   random_seed: int = 7) -> dict:
    """Run the full gauntlet on one tape: BOP up-cross vs random-entry baseline, all horizons.

    Returns a dict keyed by horizon with the cross-entry summary (gross + net), the
    drift-matched random-entry baseline, and the cross-minus-random delta.
    """
    close = bars["close"]
    ent = bop_cross_entries(bars, smooth=smooth)
    res = {"n_entries": int(len(ent)), "by_h": {}}
    for h in HORIZONS:
        g = summarize(forward_returns(close, ent, h, cost_bps=0.0))
        net = summarize(forward_returns(close, ent, h, cost_bps=cost_bps))
        rnd = summarize(forward_returns(
            close, random_entries(bars, max(len(ent), 50), smooth=smooth, seed=random_seed), h))
        res["by_h"][h] = {
            "gross": g, "net": net, "random": rnd,
            "delta_bps": (g["mean_bps"] - rnd["mean_bps"])
            if np.isfinite(g["mean_bps"]) and np.isfinite(rnd["mean_bps"]) else float("nan"),
        }
    return res
