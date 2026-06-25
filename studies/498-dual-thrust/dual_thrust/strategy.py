"""Dual Thrust as a falsifiable mechanical rule — Study 498.

Michael Chalek's **Dual Thrust** (popularised on the original Robbins / TradeStation circuit
in the 1980s–90s and a fixture of the open-source quant world) is an *opening-range breakout*:

* Over a trailing ``N``-day window, compute four extremes — the highest high (HH), lowest low
  (LL), highest close (HC) and lowest close (LC) — and form the **Range**

  $$\\text{Range} = \\max(\\text{HH}-\\text{LC},\\ \\text{HC}-\\text{LL}).$$

* Around **today's open** ``O`` draw two trigger bands:

  $$\\text{buy\\_line} = O + k_1\\,\\text{Range},\\qquad \\text{sell\\_line} = O - k_2\\,\\text{Range}.$$

* Go **long** when price trades above ``buy_line``; go **short** when it trades below
  ``sell_line``. The folklore: a clean break of the band "catches the day's trend."

We test the long side (the proponents' headline, and the only direction that survives on an
upward-drifting tape) and steelman it honestly:

1. **Range uses only the prior ``N`` bars** — HH/LL/HC/LC computed on bars strictly before
   today, so the bands are known at today's open. No look-ahead.
2. **Breakout read on the close of ``t``** — a long fires the first bar the close exceeds
   ``buy_line``; entry is at the **next** close (one documented lag); we then measure the
   forward H-day return.
3. **Controls.** (a) a **random-entry** baseline (same instrument, epoch, hold) that captures
   the tape's drift, and (b) a **scrambled-Range placebo** that permutes the Range series
   across dates, destroying the *which-day-is-volatile* structure while keeping the Range
   marginal and the ``k`` coefficients — the honest "is the Dual-Thrust geometry doing
   anything?" null.

No look-ahead: the Range is trailing, the breakout is read on the close of *t*, the position is
entered at the close of *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)
DEFAULT_N = 5      # Chalek's classic lookback
DEFAULT_K1 = 0.5   # upper trigger coefficient
DEFAULT_K2 = 0.5   # lower trigger coefficient


# --------------------------------------------------------------------------- #
# Dual-Thrust Range + trigger bands
# --------------------------------------------------------------------------- #
def dual_thrust_lines(bars: pd.DataFrame, n: int = DEFAULT_N,
                      k1: float = DEFAULT_K1, k2: float = DEFAULT_K2):
    """Trailing Dual-Thrust Range and the buy/sell trigger lines around today's open.

    Returns three aligned Series over ``bars.index``: ``(range_, buy_line, sell_line)``. The
    Range is built from the **prior** ``n`` bars (shifted by one), so the bands are known at
    today's open — no look-ahead. NaN until ``n`` bars exist.
    """
    high, low, close, open_ = bars["high"], bars["low"], bars["close"], bars["open"]
    # extremes over the trailing window ending YESTERDAY (shift(1) excludes today)
    hh = high.rolling(n).max().shift(1)
    ll = low.rolling(n).min().shift(1)
    hc = close.rolling(n).max().shift(1)
    lc = close.rolling(n).min().shift(1)
    range_ = np.maximum(hh - lc, hc - ll)
    buy_line = open_ + k1 * range_
    sell_line = open_ - k2 * range_
    return range_, buy_line, sell_line


# --------------------------------------------------------------------------- #
# Entries
# --------------------------------------------------------------------------- #
def breakout_entries(bars: pd.DataFrame, n: int = DEFAULT_N,
                     k1: float = DEFAULT_K1, k2: float = DEFAULT_K2) -> pd.DatetimeIndex:
    """Bars whose close breaks *above* the buy line — the Dual-Thrust long breakout.

    Only the *first* bar of each consecutive run is kept (the breakout, not every day price
    stays above the band). Entry is executed at the next close by :func:`forward_returns`.
    """
    _, buy_line, _ = dual_thrust_lines(bars, n=n, k1=k1, k2=k2)
    close = bars["close"]
    mask = (close > buy_line) & buy_line.notna()
    first = mask & ~mask.shift(1, fill_value=False)
    return bars.index[first.to_numpy()]


def random_entries(bars: pd.DataFrame, n_entries: int, n: int = DEFAULT_N,
                   seed: int = 0) -> pd.DatetimeIndex:
    """``n_entries`` random entry dates (after the warm-up), the drift-matched baseline."""
    rng = np.random.default_rng(seed)
    valid = bars.index[2 * n:]
    if len(valid) == 0:
        return pd.DatetimeIndex([])
    chosen = rng.choice(valid, size=min(n_entries, len(valid)), replace=False)
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


def scrambled_range_placebo(bars: pd.DataFrame, horizon: int, n: int = DEFAULT_N,
                            k1: float = DEFAULT_K1, k2: float = DEFAULT_K2,
                            n_draws: int = 1000, seed: int = 498) -> dict:
    """Placebo: permute the Range series across dates, destroying the volatility geometry.

    Keeps the Range *marginal* (the same set of band-widths) and the ``k`` coefficients, but
    shuffles which day each Range belongs to — so the buy line at any date is built from a
    Range that no longer corresponds to that date's trailing window. If the real result
    survives the scramble, the Dual-Thrust geometry (the *right-day* Range) was never
    load-bearing. Returns the share of placebo runs whose mean breakout forward return
    **beats** the real one — the honest "is the geometry adding anything?" p-value.
    """
    obs = float(np.mean(forward_returns(bars["close"], breakout_entries(bars, n=n, k1=k1, k2=k2), horizon)))
    range_, _, _ = dual_thrust_lines(bars, n=n, k1=k1, k2=k2)
    open_ = bars["open"]
    close = bars["close"]
    valid_mask = range_.notna().to_numpy()
    valid_idx = np.where(valid_mask)[0]
    range_vals = range_.to_numpy()[valid_idx]
    if valid_idx.size < 10 or not np.isfinite(obs):
        return {"obs": obs, "p_value": float("nan"), "n_draws": 0}
    rng = np.random.default_rng(seed)
    o = open_.to_numpy(dtype=float)
    idx = bars.index
    beats = 0
    valid = 0
    for _ in range(n_draws):
        perm = rng.permutation(range_vals)
        rng_full = np.full(len(idx), np.nan)
        rng_full[valid_idx] = perm
        buy = o + k1 * rng_full
        buyser = pd.Series(buy, index=idx)
        mask = (close > buyser) & buyser.notna()
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
def run_experiment(bars: pd.DataFrame, n: int = DEFAULT_N, k1: float = DEFAULT_K1,
                   k2: float = DEFAULT_K2, cost_bps: float = 1.0,
                   random_seed: int = 7) -> dict:
    """Run the full gauntlet on one tape: breakout vs random-entry baseline, all horizons.

    Returns a dict keyed by horizon with the breakout summary (gross + net), the
    drift-matched random-entry baseline, and the breakout-minus-random delta.
    """
    close = bars["close"]
    ent = breakout_entries(bars, n=n, k1=k1, k2=k2)
    res = {"n_entries": int(len(ent)), "by_h": {}}
    for h in HORIZONS:
        g = summarize(forward_returns(close, ent, h, cost_bps=0.0))
        net = summarize(forward_returns(close, ent, h, cost_bps=cost_bps))
        rnd = summarize(forward_returns(
            close, random_entries(bars, max(len(ent), 50), n=n, seed=random_seed), h))
        res["by_h"][h] = {
            "gross": g, "net": net, "random": rnd,
            "delta_bps": (g["mean_bps"] - rnd["mean_bps"])
            if np.isfinite(g["mean_bps"]) and np.isfinite(rnd["mean_bps"]) else float("nan"),
        }
    return res
