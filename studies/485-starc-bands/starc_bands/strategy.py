"""STARC Bands as a falsifiable mechanical rule — Study 485.

Manning Stoller's **STARC bands** (Stoller Average Range Channel) wrap a short simple moving
average of the close in an Average-True-Range envelope:

* the **center** is an ``sma_n``-bar SMA of the close;
* the **bands** are ``center ± k·ATR(atr_n)`` where ATR is Wilder's Average True Range.

The folklore (Stoller's own teaching, echoed by every chart-pattern site): the bands are a
**volatility-scaled channel** and a close *outside* a band is an over-extension that snaps
back — so a close **below the lower band** is a high-probability **buy** (reversion toward the
SMA), the mirror of a close above the upper band being a sell.

We encode the tightest mechanical version a proponent would accept and test it honestly:

1. **Causal bands** — the SMA and ATR at bar ``t`` use only closes/ranges through ``t`` (ATR
   is a Wilder EMA of the true range; the SMA a trailing mean). No future bars touch the band.
2. **Lower-band pierce** — a long entry fires on the first close *below* ``SMA − k·ATR`` (the
   "buy the lower band" rule). Entry is at the **next** close (one documented lag); we then
   measure the forward H-day return.
3. **Controls.** (a) a **random-entry** baseline (same instrument, same epoch, same hold) that
   captures the tape's drift, and (b) a **shuffled-ATR placebo** that rebuilds the bands from a
   permutation of the ATR series, destroying the volatility geometry while keeping the SMA and
   the price marginal — the honest "is the ATR band doing anything?" null.

No look-ahead: the bands are causal, the pierce is read on the close of *t*, the position is
entered at the close of *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)


# --------------------------------------------------------------------------- #
# Indicator: SMA, ATR, STARC bands
# --------------------------------------------------------------------------- #
def true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """Wilder's true range: max(H−L, |H−Cprev|, |L−Cprev|)."""
    prev = close.shift(1)
    tr = pd.concat([(high - low).abs(),
                    (high - prev).abs(),
                    (low - prev).abs()], axis=1).max(axis=1)
    return tr


def atr(high: pd.Series, low: pd.Series, close: pd.Series, n: int = 15) -> pd.Series:
    """Wilder's Average True Range (causal EMA of the true range, alpha = 1/n)."""
    tr = true_range(high, low, close)
    return tr.ewm(alpha=1.0 / n, adjust=False, min_periods=n).mean()


def starc_bands(bars: pd.DataFrame, sma_n: int = 6, atr_n: int = 15, k: float = 2.0):
    """STARC center (SMA) and the upper/lower ATR bands; all causal (trailing only).

    Returns three aligned Series (center, lower, upper) over ``bars.index``; NaN during the
    warm-up until both the SMA and the ATR are defined.
    """
    close = bars["close"]
    center = close.rolling(sma_n).mean()
    a = atr(bars["high"], bars["low"], close, n=atr_n)
    lower = center - k * a
    upper = center + k * a
    return center, lower, upper


# --------------------------------------------------------------------------- #
# Entries
# --------------------------------------------------------------------------- #
def lower_band_entries(bars: pd.DataFrame, sma_n: int = 6, atr_n: int = 15,
                       k: float = 2.0) -> pd.DatetimeIndex:
    """Bars whose close pierces *below* the lower STARC band — the 'buy the lower band' rule.

    Only the *first* bar of each consecutive run is kept (the pierce, not every day price stays
    outside the channel). Entry is executed at the next close by :func:`forward_returns`.
    """
    close = bars["close"]
    _, lower, _ = starc_bands(bars, sma_n=sma_n, atr_n=atr_n, k=k)
    mask = (close < lower) & lower.notna()
    first = mask & ~mask.shift(1, fill_value=False)
    return bars.index[first.to_numpy()]


def random_entries(bars: pd.DataFrame, n: int, warmup: int = 30,
                   seed: int = 0) -> pd.DatetimeIndex:
    """``n`` random entry dates (after the warm-up), the drift-matched baseline."""
    rng = np.random.default_rng(seed)
    valid = bars.index[warmup:]
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


def shuffled_atr_placebo(bars: pd.DataFrame, horizon: int, sma_n: int = 6, atr_n: int = 15,
                         k: float = 2.0, n_draws: int = 1000, seed: int = 485) -> dict:
    """Placebo: rebuild the bands from a permuted ATR series, destroying the volatility geometry.

    Keeps the SMA center and the price marginal but scrambles which ATR (band half-width) sits
    at which date, so the bands become volatility-meaningless while the typical width is
    preserved. Returns the share of placebo runs whose mean lower-pierce forward return
    **beats** the real one — the honest "is the ATR band adding anything?" p-value, plus the
    observed mean.
    """
    close = bars["close"]
    obs = float(np.mean(forward_returns(close, lower_band_entries(bars, sma_n, atr_n, k), horizon)))
    center = close.rolling(sma_n).mean()
    a = atr(bars["high"], bars["low"], close, n=atr_n)
    valid_mask = center.notna() & a.notna()
    a_vals = a[valid_mask].to_numpy(dtype=float)
    if a_vals.size < 5:
        return {"obs": obs, "p_value": float("nan"), "n_draws": 0}
    rng = np.random.default_rng(seed)
    idx = bars.index
    beats = 0
    valid = 0
    for _ in range(n_draws):
        perm = rng.permutation(a_vals)
        a_shuf = a.copy()
        a_shuf[valid_mask] = perm
        lower = center - k * a_shuf
        mask = (close < lower) & lower.notna()
        first = mask & ~mask.shift(1, fill_value=False)
        ent = idx[first.to_numpy()]
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
def run_experiment(bars: pd.DataFrame, sma_n: int = 6, atr_n: int = 15, k: float = 2.0,
                   cost_bps: float = 1.0, random_seed: int = 7) -> dict:
    """Run the full gauntlet on one tape: lower-band pierce vs random-entry baseline, all horizons.

    Returns a dict keyed by horizon with the pierce summary (gross + net), the drift-matched
    random-entry baseline, and the pierce-minus-random delta.
    """
    close = bars["close"]
    ent = lower_band_entries(bars, sma_n=sma_n, atr_n=atr_n, k=k)
    res = {"n_entries": int(len(ent)), "by_h": {}}
    for h in HORIZONS:
        g = summarize(forward_returns(close, ent, h, cost_bps=0.0))
        net = summarize(forward_returns(close, ent, h, cost_bps=cost_bps))
        rnd = summarize(forward_returns(
            close, random_entries(bars, max(len(ent), 50), warmup=max(sma_n, atr_n) + 5,
                                   seed=random_seed), h))
        res["by_h"][h] = {
            "gross": g, "net": net, "random": rnd,
            "delta_bps": (g["mean_bps"] - rnd["mean_bps"])
            if np.isfinite(g["mean_bps"]) and np.isfinite(rnd["mean_bps"]) else float("nan"),
        }
    return res
