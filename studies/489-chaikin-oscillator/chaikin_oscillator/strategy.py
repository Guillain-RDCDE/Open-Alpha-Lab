"""The Chaikin Oscillator as a falsifiable mechanical rule — Study 489.

Marc Chaikin's oscillator measures the *momentum* of the Accumulation/Distribution Line (ADL):

* **Money Flow Multiplier (MFM)** for each bar:
  ``MFM = ((close - low) - (high - close)) / (high - low)``  ∈ [-1, +1]
  (+1 when the bar closes on its high — pure accumulation; -1 on its low — distribution).
* **Money Flow Volume** ``MFV = MFM * volume``.
* **A/D Line** ``ADL_t = ADL_{t-1} + MFV_t`` (a running cumulative sum).
* **Chaikin Oscillator** ``= EMA3(ADL) - EMA10(ADL)``.

The folklore (Chaikin's own teaching, echoed on every charting site): **A/D momentum leads
price.** When the oscillator crosses **above zero**, the short EMA of accumulation has overtaken
the long EMA — buyers are gathering and a price advance is "imminent". So a cross above zero is a
**buy**.

We encode the tightest mechanical version a proponent would accept and test it honestly:

1. **Oscillator** — standard EMA(3) − EMA(10) of the ADL. All EMAs are causal (past-only).
2. **Cross above zero** — a long fires on the first bar whose oscillator turns from ≤ 0 to > 0
   (read on the close of *t*); entry is at the **next** close (one documented lag). We then
   measure the forward H-day return.
3. **Controls.** (a) a **random-entry** baseline (same instrument, same epoch, same hold) that
   captures the tape's drift, and (b) a **scrambled-MFM placebo** that permutes the per-bar Money
   Flow Multipliers across time, destroying the accumulation *information* while keeping the
   marginal MFM distribution and the volume — the honest "is the A/D geometry doing anything?"
   null.

No look-ahead: EMAs use only past bars, the cross is read on the close of *t*, the position is
entered at the close of *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)

FAST = 3
SLOW = 10


# --------------------------------------------------------------------------- #
# Indicator: Accumulation/Distribution line and the Chaikin oscillator
# --------------------------------------------------------------------------- #
def money_flow_multiplier(bars: pd.DataFrame) -> pd.Series:
    """MFM = ((C-L) - (H-C)) / (H-L), in [-1, +1]; 0 when high == low (no range)."""
    hi = bars["high"].astype(float)
    lo = bars["low"].astype(float)
    cl = bars["close"].astype(float)
    rng = (hi - lo)
    mfm = ((cl - lo) - (hi - cl)) / rng.replace(0.0, np.nan)
    return mfm.fillna(0.0)


def adl(bars: pd.DataFrame) -> pd.Series:
    """Accumulation/Distribution Line: cumulative Money Flow Volume (MFM * volume)."""
    mfm = money_flow_multiplier(bars)
    mfv = mfm * bars["volume"].astype(float)
    return mfv.cumsum()


def _ema(s: pd.Series, span: int) -> pd.Series:
    """Causal exponential moving average (past-only; pandas adjust=False)."""
    return s.ewm(span=span, adjust=False).mean()


def chaikin_oscillator(bars: pd.DataFrame, fast: int = FAST, slow: int = SLOW) -> pd.Series:
    """Chaikin Oscillator = EMA(fast, ADL) - EMA(slow, ADL). Causal throughout."""
    line = adl(bars)
    return _ema(line, fast) - _ema(line, slow)


# --------------------------------------------------------------------------- #
# Entries
# --------------------------------------------------------------------------- #
def cross_above_zero_entries(bars: pd.DataFrame, fast: int = FAST, slow: int = SLOW,
                             warmup: int = None) -> pd.DatetimeIndex:
    """Bars whose Chaikin oscillator turns from ≤ 0 to > 0 — the 'A/D momentum turns up' buy.

    The cross is read on the close of *t* (osc[t] > 0 and osc[t-1] <= 0). Entry is executed at
    the next close by :func:`forward_returns`. A warm-up of ``slow`` bars is skipped so the EMAs
    are seeded.
    """
    osc = chaikin_oscillator(bars, fast=fast, slow=slow)
    warm = slow if warmup is None else warmup
    up = (osc > 0) & (osc.shift(1) <= 0)
    up.iloc[:warm] = False
    return bars.index[up.to_numpy()]


def random_entries(bars: pd.DataFrame, n: int, slow: int = SLOW, seed: int = 0) -> pd.DatetimeIndex:
    """``n`` random entry dates (after the warm-up), the drift-matched baseline."""
    rng = np.random.default_rng(seed)
    valid = bars.index[2 * slow:]
    if len(valid) == 0:
        return pd.DatetimeIndex([])
    chosen = rng.choice(valid, size=min(n, len(valid)), replace=False)
    return pd.DatetimeIndex(sorted(chosen))


# --------------------------------------------------------------------------- #
# Forward-return engine
# --------------------------------------------------------------------------- #
def forward_returns(bars: pd.DataFrame, entries, horizon: int, cost_bps: float = 0.0) -> np.ndarray:
    """Forward ``horizon``-day return for each entry, entered at the *next* close (one lag).

    ``cost_bps`` is a one-way cost charged twice (in + out) and subtracted from each trade's
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


def scrambled_mfm_placebo(bars: pd.DataFrame, horizon: int, fast: int = FAST, slow: int = SLOW,
                          n_draws: int = 1000, seed: int = 489) -> dict:
    """Placebo: permute the per-bar Money Flow Multipliers across time, destroying the A/D info.

    Keeps the *marginal* MFM distribution and the volume series intact, but shuffles which day's
    accumulation reading is attached to which bar — so the ADL becomes a cumulated nonsense series
    while its building blocks are unchanged. Returns the share of placebo runs whose mean
    cross-above-zero forward return **beats** the real one — the honest "is the A/D geometry adding
    anything?" p-value, plus the observed mean.
    """
    obs = float(np.mean(forward_returns(bars, cross_above_zero_entries(bars, fast, slow), horizon)))
    mfm = money_flow_multiplier(bars).to_numpy(dtype=float)
    vol = bars["volume"].to_numpy(dtype=float)
    idx = bars.index
    close = bars["close"]
    rng = np.random.default_rng(seed)
    beats = 0
    valid = 0
    for _ in range(n_draws):
        perm = rng.permutation(mfm)
        line = pd.Series(np.cumsum(perm * vol), index=idx)
        osc = _ema(line, fast) - _ema(line, slow)
        up = (osc > 0) & (osc.shift(1) <= 0)
        up.iloc[:slow] = False
        ent = idx[up.to_numpy()]
        rr = forward_returns(bars, ent, horizon)
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
def run_experiment(bars: pd.DataFrame, fast: int = FAST, slow: int = SLOW, cost_bps: float = 1.0,
                   random_seed: int = 7) -> dict:
    """Run the full gauntlet on one tape: cross-above-zero vs random-entry baseline, all horizons.

    Returns a dict keyed by horizon with the cross-above-zero summary (gross + net), the
    drift-matched random-entry baseline, and the cross-minus-random delta.
    """
    ent = cross_above_zero_entries(bars, fast=fast, slow=slow)
    res = {"n_entries": int(len(ent)), "by_h": {}}
    for h in HORIZONS:
        g = summarize(forward_returns(bars, ent, h, cost_bps=0.0))
        net = summarize(forward_returns(bars, ent, h, cost_bps=cost_bps))
        rnd = summarize(forward_returns(
            bars, random_entries(bars, max(len(ent), 50), slow=slow, seed=random_seed), h))
        res["by_h"][h] = {
            "gross": g, "net": net, "random": rnd,
            "delta_bps": (g["mean_bps"] - rnd["mean_bps"])
            if np.isfinite(g["mean_bps"]) and np.isfinite(rnd["mean_bps"]) else float("nan"),
        }
    return res
