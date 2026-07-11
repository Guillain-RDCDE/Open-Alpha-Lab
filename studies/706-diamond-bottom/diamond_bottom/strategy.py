"""Diamond Bottom as a falsifiable mechanical rule — Study 706.

A **diamond bottom** is the bullish mirror of the diamond top: after a decline, the swing
range first **broadens** (a megaphone — successively higher highs and lower lows) and then
**narrows** (a symmetrical triangle — lower highs and higher lows), tracing a diamond. The
folklore (every chart-pattern site, Bulkowski, StockCharts) says the diamond marks
**accumulation** at a low: when price finally breaks **up** out of the narrowing apex, a real
advance is starting, so you **buy the breakout**.

We encode the tightest mechanical version a proponent would accept and test it honestly:

1. **Pivots are confirmed fractals** — a local extremum with ``k`` strictly-lower (higher)
   bars on each side. A pivot at bar ``t`` is only *usable* from bar ``t+k`` onward (the
   confirmation lag), so the pattern is never drawn with future bars.
2. **Diamond geometry** — over a window of the most-recent confirmed alternating pivots we
   require a broadening leg (swing amplitudes increasing) followed by a narrowing leg
   (amplitudes decreasing), i.e. the range expands then contracts — a diamond. This part of
   the geometry is direction-agnostic; it's the same test as the diamond-top study.
3. **Upside break** — once a diamond is confirmed **after a decline** (the sequence starts
   away from its low, i.e. the trough is reached *during* the diamond, not before it), a long
   fires on the first close **above** the upper edge of the narrowing apex (the breakout).
   Entry is at the **next** close (one documented lag); we then measure the forward H-day
   return of the long.
4. **Controls.** (a) a **random-entry** baseline (same instrument, same epoch, same hold,
   long) that captures the tape's drift, and (b) a **shuffled-pivot placebo** that rebuilds
   diamonds from a permutation of the pivot *prices*, destroying the broaden-then-narrow
   geometry while keeping the marginal distribution — the honest "is the diamond shape doing
   anything?" null.

No look-ahead: pivots carry a ``k``-bar confirmation lag, the breakout is read on the close
of *t*, the position is entered at the close of *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)


# --------------------------------------------------------------------------- #
# Pivot detection (confirmed fractals) — identical to the diamond-top engine
# --------------------------------------------------------------------------- #
def find_pivots(close: pd.Series, k: int = 5) -> pd.DataFrame:
    """Confirmed swing pivots: local extrema with ``k`` strictly-beaten bars on each side.

    Returns a DataFrame indexed by the pivot bar with columns ``price`` and ``kind``
    (+1 = swing high, -1 = swing low). A pivot at position ``i`` is only *confirmed* (knowable)
    at bar ``i + k`` — the diamond builder enforces that lag, so no future data leaks in.
    """
    p = close.to_numpy(dtype=float)
    n = p.size
    rows = []
    for i in range(k, n - k):
        win = p[i - k:i + k + 1]
        c = p[i]
        if c == win.max() and (win[:k] < c).all() and (win[k + 1:] < c).all():
            rows.append((i, c, +1))
        elif c == win.min() and (win[:k] > c).all() and (win[k + 1:] > c).all():
            rows.append((i, c, -1))
    if not rows:
        return pd.DataFrame(columns=["pos", "price", "kind"]).set_index("pos")
    df = pd.DataFrame(rows, columns=["pos", "price", "kind"]).set_index("pos")
    return df


def _alternating(pivots: pd.DataFrame) -> pd.DataFrame:
    """Collapse consecutive same-kind pivots, keeping the most extreme, so kinds alternate."""
    if pivots.empty:
        return pivots
    keep = []
    for pos, row in pivots.iterrows():
        if keep and keep[-1][2] == row["kind"]:
            prev = keep[-1]
            better = (row["price"] > prev[1]) if row["kind"] > 0 else (row["price"] < prev[1])
            if better:
                keep[-1] = (pos, row["price"], row["kind"])
        else:
            keep.append((pos, row["price"], row["kind"]))
    return pd.DataFrame(keep, columns=["pos", "price", "kind"]).set_index("pos")


# --------------------------------------------------------------------------- #
# Diamond geometry (shared shape test; direction-agnostic)
# --------------------------------------------------------------------------- #
def _swing_amplitudes(prices: np.ndarray) -> np.ndarray:
    """Absolute leg sizes between consecutive alternating pivots."""
    return np.abs(np.diff(prices))


def is_diamond(prices: np.ndarray, tol: float = 0.0) -> bool:
    """True if the alternating-pivot price sequence broadens then narrows (a diamond).

    Given the prices of a run of alternating pivots (high, low, high, low, ...), the leg
    amplitudes must first *increase* (broadening / megaphone) to a peak and then *decrease*
    (narrowing / symmetrical triangle). We require at least two legs on each side of the
    widest leg, with a small ``tol`` slack so a single tie doesn't disqualify a clean shape.
    """
    amp = _swing_amplitudes(prices)
    m = amp.size
    if m < 4:                      # need >=2 broadening + >=2 narrowing legs
        return False
    peak = int(np.argmax(amp))
    if peak < 1 or peak > m - 2:   # widest leg can't be at the very ends
        return False
    # broadening: amplitudes up to the peak are (weakly) increasing
    rising = all(amp[j + 1] >= amp[j] * (1.0 - tol) for j in range(peak))
    # narrowing: amplitudes after the peak are (weakly) decreasing
    falling = all(amp[j + 1] <= amp[j] * (1.0 + tol) for j in range(peak, m - 1))
    return rising and falling


def diamond_breakouts(close: pd.Series, k: int = 5, n_piv: int = 6,
                      tol: float = 0.08) -> pd.DatetimeIndex:
    """Bars whose close breaks **up** out of a confirmed diamond bottom.

    At each bar we take the most-recent ``n_piv`` confirmed, alternating pivots. If their
    price sequence forms a diamond (broaden then narrow, :func:`is_diamond`) **and** the last
    pivot sits in a declining context (a *bottom* — the diamond formed after a sell-off, so
    the trough is reached *during* the run, not at its very start) we arm a long. The long
    fires on the first close **above** the highest of the narrowing-apex pivots (the upside
    break). Only the first bar of each consecutive run is kept.

    Entry is executed at the next close by :func:`forward_returns` (with ``short=False``).
    """
    piv = _alternating(find_pivots(close, k=k))
    n = len(close)
    cl = close.to_numpy(dtype=float)
    idx = close.index
    armed_ceiling = np.full(n, np.nan)   # the breakout level active at each bar
    if len(piv) >= n_piv:
        positions = [int(p) for p in piv.index]
        prices = piv["price"].to_numpy(dtype=float)
        confirm = [p + k for p in positions]   # bar at which each pivot becomes known
        for t in range(n):
            # indices of pivots confirmed strictly by bar t
            avail = [j for j in range(len(positions)) if confirm[j] <= t]
            if len(avail) < n_piv:
                continue
            seg = avail[-n_piv:]
            seg_prices = prices[seg]
            if not is_diamond(seg_prices, tol=tol):
                continue
            # a *bottom*: the diamond should follow a decline -> the start pivot is above the
            # trough (the low is reached during the diamond, not already sitting at it)
            trough_low = seg_prices.min()
            if seg_prices[0] <= trough_low:     # already at the low -> not a clean bottom context
                continue
            # breakout level = highest pivot of the narrowing apex (last 3 pivots)
            armed_ceiling[t] = float(seg_prices[-3:].max())
    ceiling = pd.Series(armed_ceiling, index=idx)
    mask = (close > ceiling) & ceiling.notna()
    first = mask & ~mask.shift(1, fill_value=False)
    return idx[first.to_numpy()]


def random_entries(close: pd.Series, n: int, k: int = 5, seed: int = 0) -> pd.DatetimeIndex:
    """``n`` random entry dates (after the warm-up), the drift-matched baseline."""
    rng = np.random.default_rng(seed)
    valid = close.index[3 * k:]
    if len(valid) == 0:
        return pd.DatetimeIndex([])
    chosen = rng.choice(valid, size=min(n, len(valid)), replace=False)
    return pd.DatetimeIndex(sorted(chosen))


# --------------------------------------------------------------------------- #
# Forward-return engine
# --------------------------------------------------------------------------- #
def forward_returns(close: pd.Series, entries, horizon: int, cost_bps: float = 0.0,
                    short: bool = False) -> np.ndarray:
    """Forward ``horizon``-day return for each entry, entered at the *next* close (one lag).

    The diamond-bottom rule is a **long** (``short=False``): the trade return is simply the
    price move, so a post-breakout advance is a profit. ``cost_bps`` is a one-way cost
    charged twice (in + out). Trades whose window overruns the tape are dropped.
    """
    pos = {d: i for i, d in enumerate(close.index)}
    p = close.to_numpy(dtype=float)
    n = p.size
    sign = -1.0 if short else 1.0
    out = []
    for d in entries:
        i = pos.get(d)
        if i is None or i + 1 + horizon >= n:
            continue
        e = i + 1                      # enter at next close
        r = sign * (p[e + horizon] / p[e] - 1.0)
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


def shuffled_pivot_placebo(close: pd.Series, horizon: int, k: int = 5, n_piv: int = 6,
                           tol: float = 0.08, n_draws: int = 1000, seed: int = 706) -> dict:
    """Placebo: rebuild diamonds from permuted pivot *prices*, destroying the geometry.

    Keeps the pivot *positions* and the marginal price distribution but scrambles which price
    sits at which pivot, so the broaden-then-narrow shape becomes meaningless. Returns the
    share of placebo runs whose mean breakout forward (long) return **beats** the real one —
    the honest "is the diamond shape adding anything?" p-value, plus the observed mean.
    """
    obs = float(np.mean(forward_returns(close, diamond_breakouts(close, k=k, n_piv=n_piv,
                                                                 tol=tol), horizon)))
    piv = _alternating(find_pivots(close, k=k))
    if len(piv) < n_piv:
        return {"obs": obs, "p_value": float("nan"), "n_draws": 0}
    rng = np.random.default_rng(seed)
    prices = piv["price"].to_numpy(dtype=float)
    positions = [int(p) for p in piv.index]
    confirm = [p + k for p in positions]
    idx = close.index
    n = len(close)
    beats = 0
    valid = 0
    for _ in range(n_draws):
        perm = rng.permutation(prices)
        armed = np.full(n, np.nan)
        for t in range(n):
            avail = [j for j in range(len(positions)) if confirm[j] <= t]
            if len(avail) < n_piv:
                continue
            seg = avail[-n_piv:]
            sp = perm[seg]
            if not is_diamond(sp, tol=tol):
                continue
            if sp[0] <= sp.min():
                continue
            armed[t] = float(sp[-3:].max())
        ceiling = pd.Series(armed, index=idx)
        mask = (close > ceiling) & ceiling.notna()
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
def run_experiment(close: pd.Series, k: int = 5, n_piv: int = 6, tol: float = 0.08,
                   cost_bps: float = 1.0, random_seed: int = 7) -> dict:
    """Run the full gauntlet on one tape: breakout long vs random-entry baseline, all H.

    Returns a dict keyed by horizon with the breakout summary (gross + net), the
    drift-matched random-entry baseline (same long sign), and the breakout-minus-random
    delta.
    """
    ent = diamond_breakouts(close, k=k, n_piv=n_piv, tol=tol)
    res = {"n_entries": int(len(ent)), "by_h": {}}
    for h in HORIZONS:
        g = summarize(forward_returns(close, ent, h, cost_bps=0.0))
        net = summarize(forward_returns(close, ent, h, cost_bps=cost_bps))
        rnd = summarize(forward_returns(
            close, random_entries(close, max(len(ent), 50), k=k, seed=random_seed), h))
        res["by_h"][h] = {
            "gross": g, "net": net, "random": rnd,
            "delta_bps": (g["mean_bps"] - rnd["mean_bps"])
            if np.isfinite(g["mean_bps"]) and np.isfinite(rnd["mean_bps"]) else float("nan"),
        }
    return res
