"""Broadening Formation (megaphone top) as a falsifiable mechanical rule — Study 465.

A *broadening formation* (a.k.a. megaphone, broadening top, expanding triangle) is the
classic chart pattern of **diverging** swing highs and swing lows: the highs make *higher
highs*, the lows make *lower lows*, and the trading range fans out like a megaphone. The
folklore (Schabacker, Edwards & Magee, and every chart-pattern site) is that this expanding
range marks an **exhausted, over-excited top** that **reverses down** — a short fires when
price breaks the lower boundary of the megaphone.

We encode the tightest mechanical version a proponent would accept and test it honestly:

1. **Pivots are confirmed fractals** — a local extremum with ``k`` strictly-lower (higher)
   bars on each side. A pivot at bar ``t`` is only *usable* from bar ``t+k`` onward (the
   confirmation lag), so the pattern is never drawn with future bars.
2. **Megaphone test.** From the most recent confirmed pivots we take the last two swing highs
   and last two swing lows; the formation is a broadening top iff the highs are **rising**
   and the lows are **falling** (the two boundary lines *diverge*). The upper boundary line
   runs through the two highs, the lower boundary line through the two lows.
3. **Lower-boundary break.** A **short** entry fires when the close pierces *below* the lower
   boundary of a confirmed megaphone — the "broadening top reverses" rule. Entry is at the
   **next** close (one documented lag); we then measure the forward H-day return of the short
   (i.e. minus the price change).
4. **Controls.** (a) a **random-entry** short baseline (same instrument, epoch, hold) that
   captures the tape's drift, and (b) a **shuffled-pivot placebo** that rebuilds boundaries
   from a permutation of the pivot *prices*, destroying the diverging geometry while keeping
   the marginal — the honest "is the megaphone's geometry doing anything?" null.

No look-ahead: pivots carry a ``k``-bar confirmation lag, the break is read on the close of
*t*, the position is entered at the close of *t+1*. Returns are signed for a **short**.
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
    at bar ``i + k`` — :func:`build_megaphones` enforces that lag, so no future data leaks in.
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
# Megaphone geometry
# --------------------------------------------------------------------------- #
def _line(x1, y1, x2, y2):
    """Slope + intercept of the line through (x1,y1),(x2,y2). None if vertical."""
    if x2 == x1:
        return None
    s = (y2 - y1) / (x2 - x1)
    b = y1 - s * x1
    return s, b


def megaphone_boundaries(highs, lows):
    """Given the two latest swing highs and lows, return (upper, lower) boundary lines.

    ``highs`` and ``lows`` are each a list of two (pos, price) tuples (oldest first). Returns
    ``(upper_line, lower_line)`` where each is ``(slope, intercept)``, **only if** the highs
    are rising (upper slope > 0) and the lows are falling (lower slope < 0) — the diverging
    "megaphone". Returns ``None`` if the formation is not broadening.
    """
    (hx1, hy1), (hx2, hy2) = highs
    (lx1, ly1), (lx2, ly2) = lows
    up = _line(hx1, hy1, hx2, hy2)
    lo = _line(lx1, ly1, lx2, ly2)
    if up is None or lo is None:
        return None
    if up[0] > 0.0 and lo[0] < 0.0:          # highs rising AND lows falling -> diverging
        return up, lo
    return None


def build_megaphones(close: pd.Series, k: int = 10):
    """For each bar, the *currently usable* megaphone lower/upper boundary (NaN if none).

    Returns two aligned Series (lower boundary, upper boundary) over ``close.index``; NaN
    until two confirmed swing highs and two confirmed swing lows form a diverging megaphone.
    A pivot at bar ``i`` becomes usable at ``i + k`` (confirmation lag) — so the boundaries at
    bar ``t`` only use pivots confirmed by ``t``.
    """
    piv = _alternating(find_pivots(close, k=k))
    n = len(close)
    low_b = np.full(n, np.nan)
    up_b = np.full(n, np.nan)
    if len(piv) >= 4:
        confirm_pos = [int(pos) + k for pos in piv.index]
        kinds = [int(kk) for kk in piv["kind"]]
        pts = [(int(pos), float(pr)) for pos, pr in zip(piv.index, piv["price"])]
        for t in range(n):
            avail_idx = [j for j, cp in enumerate(confirm_pos) if cp <= t]
            if len(avail_idx) < 4:
                continue
            highs = [pts[j] for j in avail_idx if kinds[j] > 0]
            lows = [pts[j] for j in avail_idx if kinds[j] < 0]
            if len(highs) < 2 or len(lows) < 2:
                continue
            bnd = megaphone_boundaries(highs[-2:], lows[-2:])
            if bnd is None:
                continue
            (us, ub), (ls, lb) = bnd
            up_b[t] = us * t + ub
            low_b[t] = ls * t + lb
    idx = close.index
    return pd.Series(low_b, index=idx), pd.Series(up_b, index=idx)


# --------------------------------------------------------------------------- #
# Entries
# --------------------------------------------------------------------------- #
def lower_break_entries(close: pd.Series, k: int = 10) -> pd.DatetimeIndex:
    """Bars whose close pierces *below* the megaphone lower boundary — the SHORT signal.

    Only the *first* bar of each consecutive run is kept (the break, not every day price
    stays outside). Entry (a short) is executed at the next close by :func:`forward_returns`.
    """
    low_b, up_b = build_megaphones(close, k=k)
    mask = (close < low_b) & low_b.notna()
    first = mask & ~mask.shift(1, fill_value=False)
    return close.index[first.to_numpy()]


def random_entries(close: pd.Series, n: int, k: int = 10, seed: int = 0) -> pd.DatetimeIndex:
    """``n`` random entry dates (after the warm-up), the drift-matched short baseline."""
    rng = np.random.default_rng(seed)
    valid = close.index[2 * k:]
    if len(valid) == 0:
        return pd.DatetimeIndex([])
    chosen = rng.choice(valid, size=min(n, len(valid)), replace=False)
    return pd.DatetimeIndex(sorted(chosen))


# --------------------------------------------------------------------------- #
# Forward-return engine (SHORT)
# --------------------------------------------------------------------------- #
def forward_returns(close: pd.Series, entries, horizon: int, cost_bps: float = 0.0) -> np.ndarray:
    """Forward ``horizon``-day **short** return for each entry, entered at the *next* close.

    The pattern is a *reversal-down* short, so the trade return is ``-(P[e+h]/P[e]-1)``. One
    documented lag (enter next close). ``cost_bps`` is a one-way cost charged twice (in + out).
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
        r = -(p[e + horizon] / p[e] - 1.0)     # SHORT
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
                           n_draws: int = 1000, seed: int = 465) -> dict:
    """Placebo: rebuild megaphones from permuted pivot *prices*, destroying the geometry.

    Keeps the pivot *positions* and *kinds* and the marginal price distribution but permutes
    which price sits at which pivot, so the diverging-boundary structure becomes meaningless.
    Returns the share of placebo runs whose mean lower-break short return **beats** the real
    one — the honest "is the megaphone geometry adding anything?" p-value, plus the observed
    mean.
    """
    obs = float(np.mean(forward_returns(close, lower_break_entries(close, k=k), horizon)))
    piv = _alternating(find_pivots(close, k=k))
    if len(piv) < 4:
        return {"obs": obs, "p_value": float("nan"), "n_draws": 0}
    rng = np.random.default_rng(seed)
    prices = piv["price"].to_numpy(dtype=float)
    positions = [int(p) for p in piv.index]
    kinds = [int(kk) for kk in piv["kind"]]
    confirm = [p + k for p in positions]
    n = len(close)
    idx = close.index
    beats = 0
    valid = 0
    for _ in range(n_draws):
        perm = rng.permutation(prices)
        pts = list(zip(positions, [float(v) for v in perm]))
        low_b = np.full(n, np.nan)
        for t in range(n):
            avail_idx = [j for j, cp in enumerate(confirm) if cp <= t]
            if len(avail_idx) < 4:
                continue
            highs = [pts[j] for j in avail_idx if kinds[j] > 0]
            lows = [pts[j] for j in avail_idx if kinds[j] < 0]
            if len(highs) < 2 or len(lows) < 2:
                continue
            bnd = megaphone_boundaries(highs[-2:], lows[-2:])
            if bnd is None:
                continue
            (_us, _ub), (ls, lb) = bnd
            low_b[t] = ls * t + lb
        lowser = pd.Series(low_b, index=idx)
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
    """Run the full gauntlet on one tape: lower-break short vs random-entry baseline.

    Returns a dict keyed by horizon with the lower-break summary (gross + net), the
    drift-matched random-entry baseline, and the break-minus-random delta.
    """
    ent = lower_break_entries(close, k=k)
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
