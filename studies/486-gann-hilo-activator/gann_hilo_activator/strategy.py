"""Gann Hi-Lo Activator as a falsifiable mechanical rule — Study 486.

The **Gann Hi-Lo Activator** (popularised in Robert Krausz's *New Gann Swing Chartist* work and
built into MetaTrader/TradingView as "Gann HiLo Activator") is a trailing stop-and-reverse line:

* compute a simple moving average of the last ``period`` **highs** and of the last ``period``
  **lows** (both *shifted one bar*, so today's line uses only bars strictly before today);
* the activator is a single line that **flips**: while the regime is *long* it tracks the SMA of
  **lows** (a trailing stop below price); the moment a close prints **below** that line the
  regime flips to *short* and the activator jumps to the SMA of **highs** (a trailing stop above
  price); a close back **above** the high-line flips it long again.

The folklore (every Gann-tool write-up): the **flip forecasts trend** — a flip up is a high-odds
**buy** (the new up-leg is beginning), a flip down a sell. We encode the tightest mechanical
version a proponent would accept and test it honestly:

1. **No look-ahead.** The SMAs are shifted one bar; the regime/flip at bar ``t`` is read on the
   *close of t* (using highs/lows through ``t-1``), and the position is entered at the **close of
   t+1** (one documented lag).
2. **Flip-up entry** — a long fires on the first bar whose close flips the activator from short
   to long (price closes above the high-line). Only the first bar of each flip is kept.
3. **Controls.** (a) a **random-entry** baseline (same instrument, epoch, hold) that captures the
   tape's drift, and (b) a **shuffled-flip placebo** that keeps the *number* of flips and the
   price marginal but moves the flip dates to random bars — the honest "is the flip's *timing*
   load-bearing, or would any set of buy-days do as well?" null.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)


# --------------------------------------------------------------------------- #
# Gann Hi-Lo Activator
# --------------------------------------------------------------------------- #
def hilo_activator(bars: pd.DataFrame, period: int = 10):
    """The flipping Gann Hi-Lo Activator line and its regime over ``bars.index``.

    Returns ``(activator, regime)`` aligned Series. ``regime`` is +1 (long; activator = SMA of
    lows, a trailing stop below) or -1 (short; activator = SMA of highs, above). Both SMAs are
    **shifted one bar** so the line at bar ``t`` uses only highs/lows through ``t-1`` — the flip
    at ``t`` is therefore knowable on the close of ``t`` with no look-ahead. NaN during warm-up.
    """
    high = bars["high"].astype(float)
    low = bars["low"].astype(float)
    close = bars["close"].astype(float)
    sma_hi = high.rolling(period).mean().shift(1)
    sma_lo = low.rolling(period).mean().shift(1)

    n = len(bars)
    act = np.full(n, np.nan)
    reg = np.zeros(n, dtype=int)
    cl = close.to_numpy()
    shi = sma_hi.to_numpy()
    slo = sma_lo.to_numpy()
    regime = 1
    for t in range(n):
        if np.isnan(shi[t]) or np.isnan(slo[t]):
            reg[t] = 0
            continue
        # current activator under the standing regime
        if regime == 1:
            line = slo[t]
            if cl[t] < line:               # close below the long-stop -> flip short
                regime = -1
                line = shi[t]
        else:
            line = shi[t]
            if cl[t] > line:               # close above the short-stop -> flip long
                regime = +1
                line = slo[t]
        act[t] = line
        reg[t] = regime
    idx = bars.index
    return pd.Series(act, index=idx), pd.Series(reg, index=idx)


# --------------------------------------------------------------------------- #
# Entries
# --------------------------------------------------------------------------- #
def flip_up_entries(bars: pd.DataFrame, period: int = 10) -> pd.DatetimeIndex:
    """Bars whose close **flips the activator long** (short->long) — the Gann 'buy the flip' rule.

    Only the first bar of each flip is kept. Entry is executed at the next close by
    :func:`forward_returns`.
    """
    _, reg = hilo_activator(bars, period=period)
    r = reg.to_numpy()
    prev = np.concatenate([[0], r[:-1]])
    flip = (r == 1) & (prev == -1)
    return bars.index[flip]


def random_entries(bars: pd.DataFrame, n: int, period: int = 10, seed: int = 0) -> pd.DatetimeIndex:
    """``n`` random entry dates (after the warm-up), the drift-matched baseline."""
    rng = np.random.default_rng(seed)
    valid = bars.index[2 * period:]
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


def shuffled_flip_placebo(bars: pd.DataFrame, horizon: int, period: int = 10,
                          n_draws: int = 1000, seed: int = 486) -> dict:
    """Placebo: keep the *number* of flips but move the flip dates to random bars.

    The Gann claim is that the flip's **timing** forecasts trend. This placebo destroys exactly
    that timing — it preserves the count of flip-up entries and the price marginal, but draws the
    entry dates uniformly at random from the valid window. Returns the share of placebo runs
    whose mean forward return **beats** the real flip — the honest "is the flip's timing
    load-bearing?" p-value, plus the observed mean.
    """
    close = bars["close"]
    ent = flip_up_entries(bars, period=period)
    obs = float(np.mean(forward_returns(close, ent, horizon))) if len(ent) else float("nan")
    k = len(ent)
    if k == 0 or not np.isfinite(obs):
        return {"obs": obs, "p_value": float("nan"), "n_draws": 0}
    rng = np.random.default_rng(seed)
    valid = bars.index[2 * period:]
    beats = 0
    valid_draws = 0
    for _ in range(n_draws):
        chosen = pd.DatetimeIndex(sorted(rng.choice(valid, size=min(k, len(valid)), replace=False)))
        rr = forward_returns(close, chosen, horizon)
        if rr.size == 0:
            continue
        valid_draws += 1
        if rr.mean() >= obs:
            beats += 1
    p = (beats + 1) / (valid_draws + 1) if valid_draws else float("nan")
    return {"obs": obs, "p_value": float(p), "n_draws": valid_draws}


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def run_experiment(bars: pd.DataFrame, period: int = 10, cost_bps: float = 1.0,
                   random_seed: int = 7) -> dict:
    """Run the full gauntlet on one tape: flip-up vs random-entry baseline, all horizons.

    Returns a dict keyed by horizon with the flip summary (gross + net), the drift-matched
    random-entry baseline, and the flip-minus-random delta.
    """
    close = bars["close"]
    ent = flip_up_entries(bars, period=period)
    res = {"n_entries": int(len(ent)), "by_h": {}}
    for h in HORIZONS:
        g = summarize(forward_returns(close, ent, h, cost_bps=0.0))
        net = summarize(forward_returns(close, ent, h, cost_bps=cost_bps))
        rnd = summarize(forward_returns(
            close, random_entries(bars, max(len(ent), 50), period=period, seed=random_seed), h))
        res["by_h"][h] = {
            "gross": g, "net": net, "random": rnd,
            "delta_bps": (g["mean_bps"] - rnd["mean_bps"])
            if np.isfinite(g["mean_bps"]) and np.isfinite(rnd["mean_bps"]) else float("nan"),
        }
    return res
