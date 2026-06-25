"""Three-Outside-Up / Three-Outside-Down as a falsifiable mechanical rule — Study 454.

The *three-outside* is a three-bar candlestick pattern (Nison / Bulkowski):

* **Bar 1 (engulfed)** — a candle with a body of one colour.
* **Bar 2 (engulfing)** — an *opposite-colour* candle whose real body **completely engulfs**
  bar 1's body (open below/above, close above/below). This two-bar piece is the classic
  bullish/bearish **engulfing** pattern.
* **Bar 3 (confirming)** — a candle that closes **further in the engulf direction** (higher for
  a bullish engulf, lower for a bearish one), "confirming" the reversal.

The folklore (every candlestick primer): the confirmation upgrades the engulfing into a
high-probability reversal — go **long** on a three-outside-**up**, short on a
three-outside-**down**.

We encode the tightest mechanical version a proponent would accept and test it honestly:

1. **Pattern read on close of t.** The engulfing is bars (t-2, t-1); the confirming bar is t.
   Everything is known by the close of *t* — no future bars.
2. **Entry next close.** A three-outside-up long is entered at the close of *t+1* (one
   documented lag); we then measure the forward H-day return.
3. **Controls.** (a) a **random-entry** baseline (same instrument, epoch, hold) that captures
   the tape's drift, and (b) a **body-shuffle placebo** that re-labels each engulfing as
   confirmed/unconfirmed by a coin flip while keeping the *count* of entries — the honest "is
   the confirmation doing anything beyond the raw engulf?" null.

No look-ahead: the pattern is read on the close of *t*, the position is entered at the close of
*t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)


# --------------------------------------------------------------------------- #
# Pattern detection (three-outside-up / -down)
# --------------------------------------------------------------------------- #
def _engulf_flags(bars: pd.DataFrame):
    """Vectorised bullish/bearish ENGULFING flags on the two-bar piece ending at each bar.

    A bullish engulf at bar ``t`` means bar ``t-1`` is bearish, bar ``t`` is bullish, and bar
    ``t``'s real body fully engulfs bar ``t-1``'s body. Returns two boolean Series aligned to
    ``bars.index`` (the engulfing bar is the second of the pair).
    """
    o = bars["open"].to_numpy(dtype=float)
    c = bars["close"].to_numpy(dtype=float)
    n = o.size
    bull = np.zeros(n, dtype=bool)
    bear = np.zeros(n, dtype=bool)
    for t in range(1, n):
        po, pc = o[t - 1], c[t - 1]
        co, cc = o[t], c[t]
        # bullish engulfing: prior bearish, current bullish, current body engulfs prior body
        if pc < po and cc > co and cc >= po and co <= pc:
            bull[t] = True
        # bearish engulfing: prior bullish, current bearish, current body engulfs prior body
        elif pc > po and cc < co and cc <= po and co >= pc:
            bear[t] = True
    return (pd.Series(bull, index=bars.index), pd.Series(bear, index=bars.index))


def three_outside(bars: pd.DataFrame) -> pd.DataFrame:
    """For each bar ``t``, flag a confirmed three-outside-up / -down with the close of ``t``.

    The engulfing is bars (t-2, t-1); the confirming bar is ``t``. A three-outside-**up** needs a
    bullish engulf at ``t-1`` and ``close[t] > close[t-1]`` (confirmation in the up direction); a
    three-outside-**down** needs a bearish engulf at ``t-1`` and ``close[t] < close[t-1]``.
    Returns a DataFrame with boolean columns ``up`` and ``down`` aligned to ``bars.index``.
    """
    bull, bear = _engulf_flags(bars)
    c = bars["close"]
    bull_prev = bull.shift(1, fill_value=False)   # engulf at t-1
    bear_prev = bear.shift(1, fill_value=False)
    confirm_up = c > c.shift(1)
    confirm_dn = c < c.shift(1)
    up = bull_prev & confirm_up.fillna(False)
    down = bear_prev & confirm_dn.fillna(False)
    return pd.DataFrame({"up": up, "down": down}, index=bars.index)


# --------------------------------------------------------------------------- #
# Entries
# --------------------------------------------------------------------------- #
def three_outside_entries(bars: pd.DataFrame, side: str = "up") -> pd.DatetimeIndex:
    """Bars whose close completes a confirmed three-outside on the given ``side`` ('up'/'down').

    The confirming bar is the signal bar (close of *t*); :func:`forward_returns` enters the
    position at the close of *t+1*.
    """
    pat = three_outside(bars)
    col = pat[side]
    return bars.index[col.to_numpy()]


def random_entries(close: pd.Series, n: int, warmup: int = 3, seed: int = 0) -> pd.DatetimeIndex:
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

    ``cost_bps`` is a one-way cost (charged twice: in + out) subtracted from each trade's return.
    Trades whose window overruns the tape are dropped.
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


def confirm_shuffle_placebo(bars: pd.DataFrame, horizon: int, side: str = "up",
                            n_draws: int = 1000, seed: int = 454) -> dict:
    """Placebo: keep every engulfing, but re-decide 'confirmed?' by a coin flip.

    The defining geometry of the *three-outside* is the **third (confirming) bar**: the claim is
    that a confirmed engulf forecasts better than a bare engulf. This placebo destroys exactly
    that structure — it takes the set of all engulfing bars (the marginal of "an engulf
    happened") and labels a random subset (matched in size to the real confirmed-up count) as the
    entries, ignoring whether the third bar actually confirmed. If price genuinely respects the
    *confirmation*, the real confirmed entries should beat these confirmation-blind draws.

    Returns the share of placebo runs whose mean forward return **beats** the real confirmed one
    — the honest "is the confirmation load-bearing?" p-value, plus the observed mean.
    """
    ent = three_outside_entries(bars, side=side)
    close = bars["close"]
    obs = float(np.mean(forward_returns(close, ent, horizon))) if len(ent) else float("nan")
    bull, bear = _engulf_flags(bars)
    engulf = bull if side == "up" else bear
    # the engulf is at t-1 relative to the confirming/signal bar t; align to signal bars
    engulf_signal = engulf.shift(1, fill_value=False)
    pool = bars.index[engulf_signal.to_numpy()]      # all bars after a same-direction engulf
    k = len(ent)
    if k == 0 or len(pool) <= k:
        return {"obs": obs, "p_value": float("nan"), "n_draws": 0}
    rng = np.random.default_rng(seed)
    pool_arr = np.asarray(pool)
    beats = 0
    valid = 0
    for _ in range(n_draws):
        pick = rng.choice(len(pool_arr), size=k, replace=False)
        draw_ent = pd.DatetimeIndex(pool_arr[pick])
        rr = forward_returns(close, draw_ent, horizon)
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
def run_experiment(bars: pd.DataFrame, side: str = "up", cost_bps: float = 1.0,
                   random_seed: int = 7) -> dict:
    """Run the full gauntlet on one tape: three-outside vs random-entry baseline, all horizons.

    Returns a dict keyed by horizon with the three-outside summary (gross + net), the
    drift-matched random-entry baseline, and the pattern-minus-random delta.
    """
    close = bars["close"]
    ent = three_outside_entries(bars, side=side)
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
