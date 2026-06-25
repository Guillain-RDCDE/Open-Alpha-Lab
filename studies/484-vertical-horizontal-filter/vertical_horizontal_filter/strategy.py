"""Vertical-Horizontal-Filter as a falsifiable mechanical rule — Study 484.

Adam White's **Vertical Horizontal Filter** (Futures, 1991) measures how *directional* price
has been over a window ``N``:

    VHF_t = |max(close, N) - min(close, N)| / sum_{i} |close_i - close_{i-1}|   (last N bars)

The numerator is the net **vertical** travel (highest minus lowest close in the window); the
denominator is the total **horizontal** path (sum of absolute day-to-day moves). VHF ∈ (0, 1]:
near 1 ⇒ a clean directional move (trend); near 0 ⇒ a lot of back-and-forth with little net
progress (range). The folklore (White's own pitch, echoed on every indicator site): *the VHF
tells you when to switch on a trend-following/momentum system* — only take momentum signals when
VHF is high (trending), skip them when VHF is low (ranging).

We encode the tightest mechanical version a proponent would accept and test it honestly:

1. **The momentum signal** — a long fires when the close is above its ``mom_n``-day moving
   average (the simplest, most-cited trend-following trigger). Read on the close of *t*.
2. **The VHF gate** — keep only the momentum entries whose VHF (window ``vhf_n``) sits in the
   **top tertile** of its own trailing distribution ("VHF says trending"). Both the MA and the
   VHF use only data up to and including *t*.
3. **Entry** — at the **next** close (one documented lag); hold H ∈ {5, 10, 20, 60} days.
4. **Controls.** (a) the **ungated** momentum entry (the question is whether the gate *adds*
   anything over plain momentum); (b) a **drift-matched random-entry** baseline (same epoch,
   same hold); (c) a **shuffled-gate placebo** that permutes the VHF series in time, destroying
   its alignment with the momentum signal while keeping its marginal — the honest "is the gate's
   timing load-bearing?" null.

No look-ahead: both the MA and VHF windows end at *t*; the gate's trailing-tertile threshold uses
only past VHF; the position is entered at the close of *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)


# --------------------------------------------------------------------------- #
# The Vertical-Horizontal-Filter indicator
# --------------------------------------------------------------------------- #
def vhf(close: pd.Series, n: int = 28) -> pd.Series:
    """Vertical-Horizontal-Filter over a rolling window of ``n`` closes.

    VHF = |highest - lowest| / sum(|close diff|) over the last ``n`` bars. Returned as a Series
    aligned to ``close`` (NaN for the first ``n`` bars). Uses only data up to and including each
    bar — no look-ahead.
    """
    c = close.astype(float)
    hi = c.rolling(n).max()
    lo = c.rolling(n).min()
    num = (hi - lo).abs()
    den = c.diff().abs().rolling(n).sum()
    out = num / den.replace(0.0, np.nan)
    return out


def momentum_signal(close: pd.Series, mom_n: int = 50) -> pd.Series:
    """Boolean trend trigger: close above its ``mom_n``-day moving average (read on close of t)."""
    ma = close.rolling(mom_n).mean()
    return (close > ma) & ma.notna()


# --------------------------------------------------------------------------- #
# Entries
# --------------------------------------------------------------------------- #
def _first_of_run(mask: pd.Series) -> pd.DatetimeIndex:
    """Dates that are the *first* bar of each consecutive True run (the fresh signal)."""
    first = mask & ~mask.shift(1, fill_value=False)
    return mask.index[first.to_numpy()]


def momentum_entries(close: pd.Series, mom_n: int = 50) -> pd.DatetimeIndex:
    """Ungated momentum entries: first close that crosses above the moving average."""
    return _first_of_run(momentum_signal(close, mom_n=mom_n))


def gated_entries(close: pd.Series, mom_n: int = 50, vhf_n: int = 28,
                  q: float = 0.667, lookback: int = 252) -> pd.DatetimeIndex:
    """Momentum entries kept only when the VHF is in the **top tertile** of its trailing window.

    The gate "VHF says trending" is: VHF_t exceeds its own ``q``-quantile over the trailing
    ``lookback`` bars (a causal, no-look-ahead threshold). Returns the gated entry dates.
    """
    sig = momentum_signal(close, mom_n=mom_n)
    v = vhf(close, n=vhf_n)
    thr = v.rolling(lookback).quantile(q)
    gate = (v > thr) & thr.notna()
    mask = sig & gate
    return _first_of_run(mask)


def random_entries(close: pd.Series, n: int, warmup: int = 252, seed: int = 0) -> pd.DatetimeIndex:
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


def shuffled_gate_placebo(close: pd.Series, horizon: int, mom_n: int = 50, vhf_n: int = 28,
                          q: float = 0.667, lookback: int = 252,
                          n_draws: int = 1000, seed: int = 484) -> dict:
    """Placebo: permute the VHF series in time, destroying its alignment with momentum.

    Keeps the VHF **marginal** (same set of values, same trailing-tertile machinery) but
    scrambles *which day* each VHF reading lands on, so the gate's timing is meaningless. Returns
    the share of placebo runs whose gated-entry mean forward return **beats** the real gated one
    — the honest "is the gate's timing load-bearing?" p-value, plus the observed gated mean.
    """
    obs = float(np.mean(forward_returns(close, gated_entries(
        close, mom_n=mom_n, vhf_n=vhf_n, q=q, lookback=lookback), horizon)))
    sig = momentum_signal(close, mom_n=mom_n)
    v = vhf(close, n=vhf_n)
    vvals = v.to_numpy(dtype=float)
    finite = np.isfinite(vvals)
    if finite.sum() < lookback + 10:
        return {"obs": obs, "p_value": float("nan"), "n_draws": 0}
    rng = np.random.default_rng(seed)
    idx = close.index
    beats = 0
    valid = 0
    for _ in range(n_draws):
        perm = vvals.copy()
        # permute only the finite VHF readings among themselves (keeps the warm-up NaNs in place)
        fi = np.where(finite)[0]
        perm[fi] = rng.permutation(vvals[fi])
        vp = pd.Series(perm, index=idx)
        thr = vp.rolling(lookback).quantile(q)
        gate = (vp > thr) & thr.notna()
        mask = sig & gate
        ent = _first_of_run(mask)
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
def run_experiment(close: pd.Series, mom_n: int = 50, vhf_n: int = 28, q: float = 0.667,
                   lookback: int = 252, cost_bps: float = 1.0, random_seed: int = 7) -> dict:
    """Run the full gauntlet on one tape: VHF-gated vs ungated momentum vs random, all horizons.

    Returns a dict keyed by horizon with the gated summary (gross + net), the ungated momentum
    summary, the drift-matched random-entry baseline, and the gate-minus-ungated delta (the
    "does the gate add anything?" number) plus gate-minus-random.
    """
    gent = gated_entries(close, mom_n=mom_n, vhf_n=vhf_n, q=q, lookback=lookback)
    uent = momentum_entries(close, mom_n=mom_n)
    res = {"n_gated": int(len(gent)), "n_ungated": int(len(uent)), "by_h": {}}
    for h in HORIZONS:
        g = summarize(forward_returns(close, gent, h, cost_bps=0.0))
        net = summarize(forward_returns(close, gent, h, cost_bps=cost_bps))
        u = summarize(forward_returns(close, uent, h, cost_bps=0.0))
        rnd = summarize(forward_returns(
            close, random_entries(close, max(len(gent), 50), warmup=lookback, seed=random_seed), h))
        res["by_h"][h] = {
            "gated": g, "net": net, "ungated": u, "random": rnd,
            "delta_gate_bps": (g["mean_bps"] - u["mean_bps"])
            if np.isfinite(g["mean_bps"]) and np.isfinite(u["mean_bps"]) else float("nan"),
            "delta_rand_bps": (g["mean_bps"] - rnd["mean_bps"])
            if np.isfinite(g["mean_bps"]) and np.isfinite(rnd["mean_bps"]) else float("nan"),
        }
    return res
