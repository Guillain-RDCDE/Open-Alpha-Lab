"""DeMark's DeMarker oscillator as a falsifiable mechanical rule — Study 475.

Thomas R. DeMark's **DeMarker** (a.k.a. DeM) is a bounded 0-1 momentum/exhaustion oscillator
built from the *highs and lows*, not the closes:

* **DeMax_t** = max(High_t − High_{t-1}, 0)   — today's "new-high" extension (else 0);
* **DeMin_t** = max(Low_{t-1} − Low_t, 0)     — today's "new-low" extension (else 0);
* over a look-back ``period`` (DeMark's classic 14):

      DeMarker_t = SMA(DeMax, period) / ( SMA(DeMax, period) + SMA(DeMin, period) ).

It sits in [0, 1]: high readings (>0.7) flag **overbought** exhaustion, low readings (<0.3)
flag **oversold** exhaustion. The folklore — DeMark's own teaching, echoed on every indicator
site — is that price **exhausts and reverses** at these extremes, so a DeMarker **rising out of
oversold (<0.3)** is a high-probability **buy** (the down-move is "done").

We encode the tightest mechanical version a proponent would accept and test it honestly:

1. **The oscillator uses only past/current bars** — DeMarker_t needs bars through *t* only.
2. **Oversold-rising trigger** — a long fires when DeMarker was below 0.3 on bar *t-1* and is
   **higher** on bar *t* (rising up *out of* oversold). Only the first bar of each run is kept.
   Entry is at the **next** close (one documented lag); we then measure the forward H-day return.
3. **Controls.** (a) a **random-entry** baseline (same instrument, same epoch, same hold) that
   captures the tape's drift, and (b) a **phase-scramble placebo** that rebuilds the DeMarker
   from a circular *rotation* of the DeMax/DeMin series, destroying the alignment between the
   oscillator and price while keeping the oscillator's exact marginal — the honest "is the
   DeMarker's timing doing anything?" null.

No look-ahead: the oscillator reads only data through *t*, the trigger is read on the close of
*t*, the position is entered at the close of *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)
PERIOD = 14
OVERSOLD = 0.30


# --------------------------------------------------------------------------- #
# The DeMarker oscillator
# --------------------------------------------------------------------------- #
def demarker(high: pd.Series, low: pd.Series, period: int = PERIOD) -> pd.Series:
    """DeMark's DeMarker oscillator (0-1) over ``high``/``low``; NaN until warmed up.

    DeMax = max(ΔHigh, 0); DeMin = max(−ΔLow, 0); DeMarker = SMA(DeMax)/(SMA(DeMax)+SMA(DeMin)).
    Uses only bars through each ``t`` — no look-ahead.
    """
    h = high.to_numpy(dtype=float)
    lo = low.to_numpy(dtype=float)
    n = h.size
    de_max = np.zeros(n)
    de_min = np.zeros(n)
    de_max[1:] = np.maximum(h[1:] - h[:-1], 0.0)
    de_min[1:] = np.maximum(lo[:-1] - lo[1:], 0.0)
    out = np.full(n, np.nan)
    for i in range(period, n):
        sm_max = de_max[i - period + 1:i + 1].sum()
        sm_min = de_min[i - period + 1:i + 1].sum()
        denom = sm_max + sm_min
        out[i] = (sm_max / denom) if denom > 0 else 0.5
    return pd.Series(out, index=high.index)


# --------------------------------------------------------------------------- #
# Entries
# --------------------------------------------------------------------------- #
def oversold_rising_entries(high: pd.Series, low: pd.Series,
                            period: int = PERIOD, oversold: float = OVERSOLD) -> pd.DatetimeIndex:
    """Bars where the DeMarker **rises out of oversold** — DeMark's 'buy the exhaustion' rule.

    A long fires when DeMarker_{t-1} < ``oversold`` and DeMarker_t > DeMarker_{t-1} (turning up
    out of the oversold zone). Only the *first* bar of each consecutive run is kept (the turn,
    not every day the reading keeps rising). Entry is executed at the next close by
    :func:`forward_returns`.
    """
    dem = demarker(high, low, period=period)
    prev = dem.shift(1)
    rising_oos = (prev < oversold) & (dem > prev) & dem.notna() & prev.notna()
    first = rising_oos & ~rising_oos.shift(1, fill_value=False)
    return high.index[first.to_numpy()]


def random_entries(close: pd.Series, n: int, period: int = PERIOD, seed: int = 0) -> pd.DatetimeIndex:
    """``n`` random entry dates (after the warm-up), the drift-matched baseline."""
    rng = np.random.default_rng(seed)
    valid = close.index[period + 1:]
    if len(valid) == 0:
        return pd.DatetimeIndex([])
    chosen = rng.choice(valid, size=min(n, len(valid)), replace=False)
    return pd.DatetimeIndex(sorted(chosen))


# --------------------------------------------------------------------------- #
# Forward-return engine
# --------------------------------------------------------------------------- #
def forward_returns(close: pd.Series, entries, horizon: int, cost_bps: float = 0.0) -> np.ndarray:
    """Forward ``horizon``-day return for each entry, entered at the *next* close (one lag).

    ``cost_bps`` is a one-way round-trip cost (charged twice: in + out) subtracted from each
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


def phase_scramble_placebo(high: pd.Series, low: pd.Series, close: pd.Series, horizon: int,
                           period: int = PERIOD, oversold: float = OVERSOLD,
                           n_draws: int = 1000, seed: int = 475) -> dict:
    """Placebo: rebuild the DeMarker from a circular *rotation* of the DeMax/DeMin streams.

    Keeps the oscillator's exact marginal distribution (every reading is preserved) but breaks
    its *alignment* with price by rolling the DeMax/DeMin series by a random offset before
    forming the oscillator and reading the same oversold-rising trigger against the *true* price.
    If the DeMarker's timing carries information, the real result should sit far in the right tail
    of the scrambled distribution. Returns the share of placebo runs whose mean entry forward
    return **beats** the real one — the honest "is the DeMarker's timing load-bearing?" p-value.
    """
    obs = float(np.mean(forward_returns(close, oversold_rising_entries(high, low, period=period,
                                                                       oversold=oversold), horizon)))
    h = high.to_numpy(dtype=float)
    lo = low.to_numpy(dtype=float)
    n = h.size
    de_max = np.zeros(n)
    de_min = np.zeros(n)
    de_max[1:] = np.maximum(h[1:] - h[:-1], 0.0)
    de_min[1:] = np.maximum(lo[:-1] - lo[1:], 0.0)
    idx = close.index
    rng = np.random.default_rng(seed)
    beats = 0
    valid = 0
    for _ in range(n_draws):
        shift = int(rng.integers(period + 1, n - period - 1))
        rm = np.roll(de_max, shift)
        rn = np.roll(de_min, shift)
        dem = np.full(n, np.nan)
        for i in range(period, n):
            s_max = rm[i - period + 1:i + 1].sum()
            s_min = rn[i - period + 1:i + 1].sum()
            d = s_max + s_min
            dem[i] = (s_max / d) if d > 0 else 0.5
        dser = pd.Series(dem, index=idx)
        prev = dser.shift(1)
        rising = (prev < oversold) & (dser > prev) & dser.notna() & prev.notna()
        first = rising & ~rising.shift(1, fill_value=False)
        rr = forward_returns(close, idx[first.to_numpy()], horizon)
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
def run_experiment(bars: pd.DataFrame, period: int = PERIOD, oversold: float = OVERSOLD,
                   cost_bps: float = 1.0, random_seed: int = 7) -> dict:
    """Run the full gauntlet on one tape: oversold-rising vs random-entry baseline, all horizons.

    Returns a dict keyed by horizon with the entry summary (gross + net), the drift-matched
    random-entry baseline, and the entry-minus-random delta.
    """
    high, low, close = bars["high"], bars["low"], bars["close"]
    ent = oversold_rising_entries(high, low, period=period, oversold=oversold)
    res = {"n_entries": int(len(ent)), "by_h": {}}
    for h in HORIZONS:
        g = summarize(forward_returns(close, ent, h, cost_bps=0.0))
        net = summarize(forward_returns(close, ent, h, cost_bps=cost_bps))
        rnd = summarize(forward_returns(
            close, random_entries(close, max(len(ent), 50), period=period, seed=random_seed), h))
        res["by_h"][h] = {
            "gross": g, "net": net, "random": rnd,
            "delta_bps": (g["mean_bps"] - rnd["mean_bps"])
            if np.isfinite(g["mean_bps"]) and np.isfinite(rnd["mean_bps"]) else float("nan"),
        }
    return res
