"""WaveTrend (LazyBear) as a falsifiable mechanical rule — Study 472.

LazyBear's *WaveTrend Oscillator* (a TradingView staple, itself a re-skin of an older
commodity-channel idea) is built from the **typical price** ``tp = (high+low+close)/3``:

* ``esa  = EMA(tp, n1)``                          — a smoothed centre (the "channel");
* ``d    = EMA(|tp - esa|, n1)``                  — the mean-absolute deviation;
* ``ci   = (tp - esa) / (0.015 * d)``             — the normalised channel index;
* ``WT1  = EMA(ci, n2)``                          — the WaveTrend line;
* ``WT2  = SMA(WT1, signal)``                     — its signal line.

The folklore (LazyBear's own write-up, echoed by every TradingView tutorial): a **WT1 cross
up through WT2 while WT1 is oversold** (deeply negative, below ~−60) is a high-probability
**buy** — the oscillator is turning up from an extreme, so a bounce is "due".

We encode the tightest mechanical version a proponent would accept and test it honestly:

1. **Causal lines** — ``esa``, ``d``, ``ci``, ``WT1``, ``WT2`` are all causal EMAs/SMAs of
   *past* closes; no future data leaks in.
2. **Oversold cross-up** — a long entry fires when WT1 crosses *above* WT2 on the close of
   *t* while WT1 was *below* the oversold band on the prior bar. Entry is at the **next**
   close (one documented lag); we then measure the forward H-day return.
3. **Controls.** (a) a **random-entry** baseline (same instrument, same epoch, same hold)
   that captures the tape's drift, and (b) a **scrambled-signal placebo** that re-runs the
   exact same cross-up-from-oversold rule on a WT1 series whose increments have been
   permuted, destroying the oscillator's wave structure while keeping its marginal — the
   honest "is the WaveTrend geometry doing anything?" null.

No look-ahead: the lines are causal, the cross is read on the close of *t*, the position is
entered at the close of *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)

N1 = 10            # channel EMA length
N2 = 21            # average EMA length
SIGNAL = 4         # signal SMA length
OVERSOLD = -60.0   # oversold band


# --------------------------------------------------------------------------- #
# WaveTrend lines
# --------------------------------------------------------------------------- #
def _ema(s: pd.Series, length: int) -> pd.Series:
    """Causal EMA, TradingView alpha = 2/(length+1)."""
    return s.ewm(span=length, adjust=False).mean()


def typical_price(bars: pd.DataFrame) -> pd.Series:
    """The HLC3 typical price WaveTrend is built on."""
    return (bars["high"] + bars["low"] + bars["close"]) / 3.0


def wavetrend(tp: pd.Series, n1: int = N1, n2: int = N2, signal: int = SIGNAL):
    """Return ``(wt1, wt2)`` — the WaveTrend line and its SMA signal line.

    All EMAs/SMA are causal; the lines at bar ``t`` use only closes up to and including ``t``.
    """
    esa = _ema(tp, n1)
    d = _ema((tp - esa).abs(), n1)
    d = d.replace(0.0, np.nan).ffill().fillna(1e-12)
    ci = (tp - esa) / (0.015 * d)
    wt1 = _ema(ci, n2)
    wt2 = wt1.rolling(signal).mean()
    return wt1, wt2


# --------------------------------------------------------------------------- #
# Entries
# --------------------------------------------------------------------------- #
def cross_up_entries(bars: pd.DataFrame, n1: int = N1, n2: int = N2, signal: int = SIGNAL,
                     oversold: float = OVERSOLD) -> pd.DatetimeIndex:
    """Bars where WT1 crosses *up* through WT2 while oversold — the WaveTrend 'buy' rule.

    Cross on close of *t* (WT1[t] > WT2[t] and WT1[t-1] <= WT2[t-1]) with WT1[t-1] below the
    oversold band. Entry is executed at the next close by :func:`forward_returns`.
    """
    tp = typical_price(bars)
    wt1, wt2 = wavetrend(tp, n1=n1, n2=n2, signal=signal)
    prev_below = wt1.shift(1) <= wt2.shift(1)
    now_above = wt1 > wt2
    cross = prev_below & now_above & wt1.notna() & wt2.notna()
    over = wt1.shift(1) < oversold
    fire = cross & over
    return bars.index[fire.fillna(False).to_numpy()]


def random_entries(close: pd.Series, n: int, warm: int = 40, seed: int = 0) -> pd.DatetimeIndex:
    """``n`` random entry dates (after the warm-up), the drift-matched baseline."""
    rng = np.random.default_rng(seed)
    valid = close.index[warm:]
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


def scrambled_signal_placebo(bars: pd.DataFrame, horizon: int, n1: int = N1, n2: int = N2,
                             signal: int = SIGNAL, oversold: float = OVERSOLD,
                             n_draws: int = 1000, seed: int = 472) -> dict:
    """Placebo: re-run the exact cross-up-from-oversold rule on a WT1 with permuted increments.

    We keep WT1's marginal (its values, hence the oversold band still bites the same fraction
    of the time) but **permute its first differences** before re-cumulating, destroying the
    wave structure that makes a "cross up from oversold" meaningful. WT2 is recomputed as the
    SMA of the scrambled WT1, and the same rule fires. Returns the share of placebo runs whose
    mean cross-up forward return **beats** the real one — the honest "is the WaveTrend geometry
    adding anything?" p-value, plus the observed mean.
    """
    obs = float(np.mean(forward_returns(
        bars["close"], cross_up_entries(bars, n1=n1, n2=n2, signal=signal, oversold=oversold),
        horizon)))
    tp = typical_price(bars)
    wt1, _ = wavetrend(tp, n1=n1, n2=n2, signal=signal)
    idx = bars.index
    close = bars["close"]
    v = wt1.to_numpy(dtype=float)
    finite = np.isfinite(v)
    base0 = v[finite][0] if finite.any() else 0.0
    diffs = np.diff(v[finite])
    rng = np.random.default_rng(seed)
    beats = 0
    valid = 0
    for _ in range(n_draws):
        perm = rng.permutation(diffs)
        scr = np.concatenate([[base0], base0 + np.cumsum(perm)])
        s = pd.Series(np.nan, index=idx)
        s.iloc[np.flatnonzero(finite)] = scr
        s2 = s.rolling(signal).mean()
        prev_below = s.shift(1) <= s2.shift(1)
        now_above = s > s2
        over = s.shift(1) < oversold
        fire = (prev_below & now_above & over & s.notna() & s2.notna()).fillna(False)
        ent = idx[fire.to_numpy()]
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
def run_experiment(bars: pd.DataFrame, n1: int = N1, n2: int = N2, signal: int = SIGNAL,
                   oversold: float = OVERSOLD, cost_bps: float = 1.0,
                   random_seed: int = 7) -> dict:
    """Run the full gauntlet on one tape: WaveTrend cross-up vs random-entry baseline.

    Returns a dict keyed by horizon with the cross-up summary (gross + net), the
    drift-matched random-entry baseline, and the cross-minus-random delta.
    """
    close = bars["close"]
    ent = cross_up_entries(bars, n1=n1, n2=n2, signal=signal, oversold=oversold)
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
