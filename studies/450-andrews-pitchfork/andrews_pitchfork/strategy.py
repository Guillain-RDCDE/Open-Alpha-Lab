"""Andrews' Pitchfork as a falsifiable mechanical rule — Study 450.

Alan Andrews' pitchfork (a.k.a. the *median-line* method) is drawn from three swing pivots:

* **P0** — a confirmed swing (high or low) that anchors the handle;
* **P1, P2** — the next two confirmed swings of alternating direction.

From these three points:

* the **median line (ML)** runs from P0 through the *midpoint* M = (P1 + P2)/2;
* the two **tines** are lines through P1 and through P2, both parallel to the ML.

The folklore (Andrews' own teaching, echoed by every chart-pattern site): *price respects the
fork* — it oscillates between the tines and is "drawn back" to the median line, so a touch of
the **lower tine** is a high-probability **buy** (reversion up toward the ML), and a touch of
the upper tine a sell.

We encode the tightest mechanical version a proponent would accept and test it honestly:

1. **Pivots are confirmed fractals** — a local extremum with ``k`` strictly-lower (higher)
   bars on each side. A pivot at bar ``t`` is only *usable* from bar ``t+k`` onward (the
   confirmation lag), so the fork is never drawn with future bars.
2. **Rolling fork** — at each bar we anchor the fork on the three most-recent *confirmed*
   pivots (P0 oldest, then P1, P2), requiring P1/P2 to alternate so the fork is well-formed.
3. **Lower-tine touch** — a long entry fires when the close pierces *below* the lower tine
   (the channel's lower edge), the Andrews "buy the lower line" rule. Entry is at the **next**
   close (one documented lag); we then measure the forward H-day return.
4. **Controls.** (a) a **random-entry** baseline (same instrument, same epoch, same hold) that
   captures the tape's drift, and (b) a **shuffled-pivot placebo** that rebuilds forks from a
   permutation of the pivot *prices*, destroying the geometry while keeping the marginal
   distribution — the honest "is the fork's geometry doing anything?" null.

No look-ahead: pivots carry a ``k``-bar confirmation lag, the touch is read on the close of
*t*, the position is entered at the close of *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)


# --------------------------------------------------------------------------- #
# Pivot detection (confirmed fractals)
# --------------------------------------------------------------------------- #
def find_pivots(close: pd.Series, k: int = 10) -> pd.DataFrame:
    """Confirmed swing pivots: local extrema with ``k`` strictly-beaten bars on each side.

    Returns a DataFrame indexed by the pivot bar with columns ``price`` and ``kind``
    (+1 = swing high, -1 = swing low). A pivot at position ``i`` is only *confirmed* (knowable)
    at bar ``i + k`` — :func:`build_forks` enforces that lag, so no future data leaks in.
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
            # same kind in a row: replace if more extreme in that direction
            prev = keep[-1]
            better = (row["price"] > prev[1]) if row["kind"] > 0 else (row["price"] < prev[1])
            if better:
                keep[-1] = (pos, row["price"], row["kind"])
        else:
            keep.append((pos, row["price"], row["kind"]))
    return pd.DataFrame(keep, columns=["pos", "price", "kind"]).set_index("pos")


# --------------------------------------------------------------------------- #
# Pitchfork geometry
# --------------------------------------------------------------------------- #
def fork_lines(p0, p1, p2):
    """Given (pos, price) for the three pivots, return slope + intercepts of (ML, lower, upper).

    The median line runs from P0 through M = midpoint(P1, P2). The tines are parallel lines
    through P1 and P2. Lines are in (bar-index, price) space: ``price = slope*pos + intercept``.
    Returns ``(slope, ml_b, lo_b, up_b)`` where ``*_b`` are the intercepts and the lower/upper
    are ordered by their value at P0's bar (lower tine first).
    """
    x0, y0 = p0
    x1, y1 = p1
    x2, y2 = p2
    mx, my = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    if mx == x0:
        return None
    slope = (my - y0) / (mx - x0)
    ml_b = y0 - slope * x0
    b1 = y1 - slope * x1
    b2 = y2 - slope * x2
    lo_b, up_b = (b1, b2) if b1 <= b2 else (b2, b1)
    return slope, ml_b, lo_b, up_b


def build_forks(close: pd.Series, k: int = 10):
    """For each bar, the *currently usable* fork built from the 3 latest confirmed pivots.

    Returns three aligned Series (median, lower tine, upper tine) over ``close.index``; NaN
    until three confirmed, alternating pivots exist. A pivot at bar ``i`` becomes usable at
    ``i + k`` (confirmation lag) — so the fork at bar ``t`` only uses pivots confirmed by ``t``.
    """
    piv = _alternating(find_pivots(close, k=k))
    n = len(close)
    med = np.full(n, np.nan)
    low = np.full(n, np.nan)
    up = np.full(n, np.nan)
    if len(piv) >= 3:
        confirm_pos = [int(pos) + k for pos in piv.index]   # bar at which each pivot is known
        pts = [(int(pos), float(pr)) for pos, pr in zip(piv.index, piv["price"])]
        for t in range(n):
            # pivots confirmed strictly by bar t
            avail = [pt for pt, cp in zip(pts, confirm_pos) if cp <= t]
            if len(avail) < 3:
                continue
            p0, p1, p2 = avail[-3], avail[-2], avail[-1]
            ln = fork_lines(p0, p1, p2)
            if ln is None:
                continue
            slope, ml_b, lo_b, up_b = ln
            med[t] = slope * t + ml_b
            low[t] = slope * t + lo_b
            up[t] = slope * t + up_b
    idx = close.index
    return (pd.Series(med, index=idx), pd.Series(low, index=idx), pd.Series(up, index=idx))


# --------------------------------------------------------------------------- #
# Entries
# --------------------------------------------------------------------------- #
def tine_touch_entries(close: pd.Series, k: int = 10) -> pd.DatetimeIndex:
    """Bars whose close pierces *below* the lower tine — the Andrews 'buy the lower line' rule.

    Only the *first* bar of each consecutive run is kept (the touch, not every day price stays
    outside the channel). Entry is executed at the next close by :func:`forward_returns`.
    """
    med, low, up = build_forks(close, k=k)
    mask = (close < low) & low.notna()
    first = mask & ~mask.shift(1, fill_value=False)
    return close.index[first.to_numpy()]


def random_entries(close: pd.Series, n: int, k: int = 10, seed: int = 0) -> pd.DatetimeIndex:
    """``n`` random entry dates (after the warm-up), the drift-matched baseline."""
    rng = np.random.default_rng(seed)
    valid = close.index[2 * k:]
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


def shuffled_pivot_placebo(close: pd.Series, horizon: int, k: int = 10,
                           n_draws: int = 1000, seed: int = 450) -> dict:
    """Placebo: rebuild forks from permuted pivot *prices*, destroying the geometry.

    Keeps the pivot *positions* and the marginal price distribution but scrambles which price
    sits at which pivot, so the fork lines become geometrically meaningless. Returns the share
    of placebo runs whose mean tine-touch forward return **beats** the real one — the honest
    "is the fork's geometry adding anything?" p-value, plus the observed mean.
    """
    obs = float(np.mean(forward_returns(close, tine_touch_entries(close, k=k), horizon)))
    piv = _alternating(find_pivots(close, k=k))
    if len(piv) < 3:
        return {"obs": obs, "p_value": float("nan"), "n_draws": 0}
    rng = np.random.default_rng(seed)
    prices = piv["price"].to_numpy(dtype=float)
    positions = list(piv.index)
    confirm = [int(pos) + k for pos in positions]
    cl = close.to_numpy(dtype=float)
    n = cl.size
    idx = close.index
    beats = 0
    valid = 0
    for _ in range(n_draws):
        perm = rng.permutation(prices)
        pts = list(zip([int(p) for p in positions], [float(v) for v in perm]))
        low = np.full(n, np.nan)
        for t in range(n):
            avail = [pt for pt, cp in zip(pts, confirm) if cp <= t]
            if len(avail) < 3:
                continue
            ln = fork_lines(avail[-3], avail[-2], avail[-1])
            if ln is None:
                continue
            slope, ml_b, lo_b, up_b = ln
            low[t] = slope * t + lo_b
        lowser = pd.Series(low, index=idx)
        mask = (close < lowser) & lowser.notna()
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
def run_experiment(close: pd.Series, k: int = 10, cost_bps: float = 1.0,
                   random_seed: int = 7) -> dict:
    """Run the full gauntlet on one tape: tine-touch vs random-entry baseline, all horizons.

    Returns a dict keyed by horizon with the tine-touch summary (gross + net), the
    drift-matched random-entry baseline, and the touch-minus-random delta.
    """
    ent = tine_touch_entries(close, k=k)
    res = {"n_entries": int(len(ent)), "by_h": {}}
    for h in HORIZONS:
        g = summarize(forward_returns(close, ent, h, cost_bps=0.0))
        net = summarize(forward_returns(close, ent, h, cost_bps=cost_bps))
        # drift-matched random baseline: same number of trades, same hold
        rnd = summarize(forward_returns(
            close, random_entries(close, max(len(ent), 50), k=k, seed=random_seed), h))
        res["by_h"][h] = {
            "gross": g, "net": net, "random": rnd,
            "delta_bps": (g["mean_bps"] - rnd["mean_bps"])
            if np.isfinite(g["mean_bps"]) and np.isfinite(rnd["mean_bps"]) else float("nan"),
        }
    return res
