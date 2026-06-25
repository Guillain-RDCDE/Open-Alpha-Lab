"""The McClellan Oscillator as a falsifiable mechanical rule — Study 491.

Sherman & Marian McClellan's oscillator (1969) is a **breadth-momentum** indicator:

* Take the daily **net advances** ``A_t`` = (advancing issues − declining issues) across
  a market basket.
* The oscillator is the difference of two EMAs of that net-breadth series::

      McOsc_t = EMA_19(A)_t − EMA_39(A)_t

  (the classic 10%/5% smoothing-constant pair, equivalently EMA spans 19 and 39).

The folklore (every breadth-trading site, and McClellan's own teaching): the oscillator
**forecasts the index**. The textbook bull trigger is the oscillator **crossing up
through zero from negative territory** — breadth momentum has turned positive, so the
index is "about to" rally. We encode exactly that:

1. **Causal oscillator.** EMAs are computed forward-only (no centering), so ``McOsc_t``
   uses only breadth up to and including day *t*.
2. **The up-cross trigger.** A long fires when ``McOsc_{t-1} <= 0 < McOsc_t`` — the first
   day the oscillator turns positive from a non-positive value.
3. **No look-ahead.** The cross is read on the close of *t*; the position is entered at
   the close of *t+1* (one documented lag); we then measure the forward H-day return.
4. **Controls.** (a) a **random-entry** baseline (same instrument, epoch, hold) that
   captures the tape's drift, and (b) a **shuffled-breadth placebo** that permutes the
   net-advances series in time before building the oscillator — destroying the
   indicator's defining temporal geometry while keeping its marginal distribution. The
   honest "is the breadth-momentum structure load-bearing?" null.

The Signal axis is always **trigger vs the drift-matched random baseline** (a Welch *t*),
never trigger-vs-zero, because an upward-drifting index makes *any* long entry look good.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)
FAST, SLOW = 19, 39  # classic McClellan EMA spans (10% / 5% smoothing constants)


# --------------------------------------------------------------------------- #
# The oscillator
# --------------------------------------------------------------------------- #
def mcclellan(net_adv: pd.Series, fast: int = FAST, slow: int = SLOW) -> pd.Series:
    """The McClellan oscillator = EMA_fast(net_adv) − EMA_slow(net_adv), causal."""
    a = pd.Series(np.asarray(net_adv, dtype=float), index=net_adv.index)
    ef = a.ewm(span=fast, adjust=False).mean()
    es = a.ewm(span=slow, adjust=False).mean()
    osc = ef - es
    osc.name = "mcosc"
    return osc


def _warmup(net_adv: pd.Series, slow: int = SLOW) -> int:
    """Bars to discard so both EMAs are seeded (a generous 3x the slow span)."""
    return min(len(net_adv), 3 * slow)


def up_cross_dates(osc: pd.Series, net_adv: pd.Series | None = None,
                   slow: int = SLOW) -> pd.DatetimeIndex:
    """Dates where the oscillator crosses **up through zero from non-positive**.

    The cross is confirmed on the close of that bar; :func:`forward_returns` enters at
    the next close. A warm-up window is dropped so the EMAs are seeded.
    """
    o = osc.to_numpy(dtype=float)
    cross = np.zeros(o.size, dtype=bool)
    cross[1:] = (o[1:] > 0) & (o[:-1] <= 0)
    w = _warmup(osc, slow)
    cross[:w] = False
    return osc.index[cross]


def osc_up_cross_entries(net_adv: pd.Series, fast: int = FAST, slow: int = SLOW) -> pd.DatetimeIndex:
    """Convenience: the McClellan up-cross entry dates straight from a net-advances series."""
    return up_cross_dates(mcclellan(net_adv, fast, slow), net_adv, slow)


def random_entries(index: pd.DatetimeIndex, n: int, slow: int = SLOW,
                   seed: int = 0) -> pd.DatetimeIndex:
    """``n`` random entry dates (after the warm-up), the drift-matched baseline."""
    rng = np.random.default_rng(seed)
    w = min(len(index), 3 * slow)
    valid = index[w:]
    if len(valid) == 0:
        return pd.DatetimeIndex([])
    chosen = rng.choice(valid, size=min(n, len(valid)), replace=False)
    return pd.DatetimeIndex(sorted(chosen))


# --------------------------------------------------------------------------- #
# Forward-return engine
# --------------------------------------------------------------------------- #
def forward_returns(close: pd.Series, entries, horizon: int, cost_bps: float = 0.0) -> np.ndarray:
    """Forward ``horizon``-day return for each entry, entered at the *next* close (one lag).

    ``cost_bps`` is a one-way cost charged twice (in + out) and subtracted from each
    trade's return. Trades whose window overruns the tape are dropped.
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


def shuffled_breadth_placebo(close: pd.Series, net_adv: pd.Series, horizon: int,
                             fast: int = FAST, slow: int = SLOW,
                             n_draws: int = 1000, seed: int = 491) -> dict:
    """Placebo: time-permute the net-advances series, rebuild the oscillator, re-trigger.

    Shuffling ``net_adv`` in time destroys the breadth-momentum *structure* the McClellan
    EMAs are built to capture (autocorrelation, the slow swings) while preserving its
    marginal distribution exactly. If the real up-cross carries information, the observed
    mean should sit far in the right tail of the shuffled distribution. Returns the share
    of placebo runs whose mean forward return **beats** the real one — the honest "is the
    breadth geometry load-bearing?" p-value, plus the observed mean.
    """
    obs = float(np.mean(forward_returns(
        close, osc_up_cross_entries(net_adv, fast, slow), horizon)))
    rng = np.random.default_rng(seed)
    vals = net_adv.to_numpy(dtype=float)
    idx = net_adv.index
    beats = 0
    valid = 0
    for _ in range(n_draws):
        perm = pd.Series(rng.permutation(vals), index=idx, name="net_adv")
        ent = osc_up_cross_entries(perm, fast, slow)
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
def run_experiment(close: pd.Series, net_adv: pd.Series, fast: int = FAST, slow: int = SLOW,
                   cost_bps: float = 1.0, random_seed: int = 7) -> dict:
    """Run the full gauntlet on one tape: up-cross vs random-entry baseline, all horizons.

    ``close`` is the traded instrument (SPY), ``net_adv`` the breadth series feeding the
    oscillator. Returns a dict keyed by horizon with the up-cross summary (gross + net),
    the drift-matched random-entry baseline, and the trigger-minus-random delta.
    """
    # align breadth onto the traded calendar
    net = net_adv.reindex(close.index).dropna()
    ent = osc_up_cross_entries(net, fast, slow)
    res = {"n_entries": int(len(ent)), "by_h": {}}
    for h in HORIZONS:
        g = summarize(forward_returns(close, ent, h, cost_bps=0.0))
        net_s = summarize(forward_returns(close, ent, h, cost_bps=cost_bps))
        rnd = summarize(forward_returns(
            close, random_entries(close.index, max(len(ent), 50), slow=slow, seed=random_seed), h))
        res["by_h"][h] = {
            "gross": g, "net": net_s, "random": rnd,
            "delta_bps": (g["mean_bps"] - rnd["mean_bps"])
            if np.isfinite(g["mean_bps"]) and np.isfinite(rnd["mean_bps"]) else float("nan"),
        }
    return res
