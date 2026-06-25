"""Descending Triangle as a falsifiable mechanical rule — Study 461.

The *descending triangle* is one of the canonical chart-pattern continuation figures:

* a horizontal **support** — a run of swing **lows at (roughly) the same price** (a flat floor);
* a falling **resistance** — a run of swing **highs that descend** (a downward-sloping ceiling);
* the two converge toward an **apex**.

The folklore (Edwards & Magee, every chart-pattern site, TradingView/StockCharts ChartSchool):
the descending triangle is a **bearish continuation** — price coils, then *breaks down through the
flat support and keeps falling*. The textbook trade is to **short the support break**.

We encode the tightest mechanical version a proponent would accept and test it honestly:

1. **Pivots are confirmed fractals** — a local extremum with ``k`` strictly-lower (higher) bars
   on each side. A pivot at bar ``t`` is only *usable* from bar ``t+k`` onward (the confirmation
   lag), so the triangle is never drawn with future bars.
2. **Rolling triangle test** — at each bar we look back over a window of confirmed pivots and ask:
   are the last few swing **highs descending** and the last few swing **lows flat** (a horizontal
   support within a tolerance band)? If so, a descending triangle is "live" with that support.
3. **Support break** — a SHORT entry fires when the close first pierces *below* the flat support
   of a live triangle (the textbook break-down). Entry is at the **next** close (one documented
   lag); we then measure the forward H-day return of the **short** (= minus the price return).
4. **Controls.** (a) a **random-entry** baseline (same instrument, epoch, hold; also booked
   short) that captures the tape's drift, and (b) a **scrambled-geometry placebo** that keeps the
   pivot timing/support level but *shuffles the swing-high prices* so the "descending highs"
   constraint is destroyed while the marginal is preserved — the honest "is the triangle's
   geometry doing anything beyond a plain support break?" null.

No look-ahead: pivots carry a ``k``-bar confirmation lag, the break is read on the close of
*t*, the position is entered at the close of *t+1*. Because the trade is a SHORT, ``forward_returns``
returns the short P&L, so a textbook break-DOWN shows as a POSITIVE number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)


# --------------------------------------------------------------------------- #
# Pivot detection (confirmed fractals)
# --------------------------------------------------------------------------- #
def find_pivots(close: pd.Series, k: int = 5) -> pd.DataFrame:
    """Confirmed swing pivots: local extrema with ``k`` strictly-beaten bars on each side.

    Returns a DataFrame indexed by the pivot bar (integer position) with columns ``price`` and
    ``kind`` (+1 = swing high, -1 = swing low). A pivot at position ``i`` is only *confirmed*
    (knowable) at bar ``i + k`` — the entry/detection routines enforce that lag, so no future
    data leaks in.
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
    return pd.DataFrame(rows, columns=["pos", "price", "kind"]).set_index("pos")


# --------------------------------------------------------------------------- #
# Descending-triangle geometry
# --------------------------------------------------------------------------- #
def _is_descending_triangle(highs, lows, flat_tol=0.02, min_drop=0.01):
    """Given recent swing-high prices and swing-low prices, is this a descending triangle?

    ``highs`` / ``lows`` are lists of (pos, price), oldest-first. Rule:

    * at least 2 swing highs and 2 swing lows;
    * the swing **highs descend** — the last high is below the first by ``min_drop`` (fractional)
      and the sequence is (weakly) non-increasing;
    * the swing **lows are flat** — their spread is within ``flat_tol`` of the support level.

    Returns the **support level** (mean of the flat lows) if it qualifies, else ``None``.
    """
    if len(highs) < 2 or len(lows) < 2:
        return None
    hp = np.array([pr for _, pr in highs], dtype=float)
    lp = np.array([pr for _, pr in lows], dtype=float)
    # descending highs: last well below first, and monotone-ish (no higher high)
    if not (hp[-1] < hp[0] * (1.0 - min_drop)):
        return None
    if np.any(np.diff(hp) > hp[:-1] * 0.005):  # allow tiny noise, but no real higher-high
        return None
    # flat lows: tight band
    support = float(lp.mean())
    if support <= 0:
        return None
    if (lp.max() - lp.min()) / support > flat_tol:
        return None
    return support


def build_support(close: pd.Series, k: int = 5, lookback: int = 6,
                  flat_tol: float = 0.02, min_drop: float = 0.01) -> pd.Series:
    """For each bar, the flat-support level of the *currently live* descending triangle (else NaN).

    At bar ``t`` we collect the confirmed pivots usable by ``t`` (each pivot at bar ``i`` becomes
    usable at ``i+k``), take the most-recent ``lookback`` of them, split into swing highs/lows,
    and run :func:`_is_descending_triangle`. The returned Series is the support price when a
    descending triangle is live, NaN otherwise — strictly causal.
    """
    piv = find_pivots(close, k=k)
    return _support_from_pivots(piv, len(close), close.index, k=k, lookback=lookback,
                               flat_tol=flat_tol, min_drop=min_drop)


def _support_from_pivots(piv: pd.DataFrame, n: int, index, k: int = 5, lookback: int = 6,
                         flat_tol: float = 0.02, min_drop: float = 0.01) -> pd.Series:
    """Causal support Series from a pivot table — the hot loop shared by real + placebo.

    Pivots become *confirmed* (usable) at ``pos + k``; both ``pos`` and the confirm times are
    sorted, so we sweep a pointer instead of re-filtering every bar (O(n_bars + n_pivots) rather
    than O(n_bars * n_pivots)). At each bar we run :func:`_is_descending_triangle` on the most
    recent ``lookback`` confirmed pivots.
    """
    sup = np.full(n, np.nan)
    if len(piv) >= 4:
        pts = [(int(pos), float(pr), int(kd))
               for pos, pr, kd in zip(piv.index, piv["price"], piv["kind"])]
        confirm = sorted(pos + k for pos, _, _ in pts)
        # pivots are emitted in position order; confirm = pos+k preserves that order
        avail = []
        ptr = 0
        m = len(pts)
        for t in range(n):
            while ptr < m and confirm[ptr] <= t:
                avail.append(pts[ptr])
                ptr += 1
            if len(avail) < 4:
                continue
            recent = avail[-lookback:]
            highs = [(pos, pr) for pos, pr, kd in recent if kd > 0]
            lows = [(pos, pr) for pos, pr, kd in recent if kd < 0]
            s = _is_descending_triangle(highs, lows, flat_tol=flat_tol, min_drop=min_drop)
            if s is not None:
                sup[t] = s
    return pd.Series(sup, index=index)


# --------------------------------------------------------------------------- #
# Entries
# --------------------------------------------------------------------------- #
def support_break_entries(close: pd.Series, k: int = 5, lookback: int = 6,
                          flat_tol: float = 0.02, min_drop: float = 0.01) -> pd.DatetimeIndex:
    """Bars whose close first pierces *below* a live descending-triangle support (the break-down).

    Only the *first* bar of each consecutive run below support is kept (the break, not every day
    price stays below). Entry (a SHORT) is executed at the next close by :func:`forward_returns`.
    """
    sup = build_support(close, k=k, lookback=lookback, flat_tol=flat_tol, min_drop=min_drop)
    mask = (close < sup) & sup.notna()
    first = mask & ~mask.shift(1, fill_value=False)
    return close.index[first.to_numpy()]


def random_entries(close: pd.Series, n: int, k: int = 5, seed: int = 0) -> pd.DatetimeIndex:
    """``n`` random entry dates (after the warm-up), the drift-matched (short) baseline."""
    rng = np.random.default_rng(seed)
    valid = close.index[2 * k:]
    if len(valid) == 0:
        return pd.DatetimeIndex([])
    chosen = rng.choice(valid, size=min(n, len(valid)), replace=False)
    return pd.DatetimeIndex(sorted(chosen))


# --------------------------------------------------------------------------- #
# Forward-return engine (SHORT P&L)
# --------------------------------------------------------------------------- #
def forward_returns(close: pd.Series, entries, horizon: int, cost_bps: float = 0.0) -> np.ndarray:
    """Forward ``horizon``-day SHORT return for each entry, entered at the *next* close (one lag).

    The trade is a short on the break-down, so the booked return is **minus** the price return:
    a textbook continuation (price falls) shows as a POSITIVE number. ``cost_bps`` is a one-way
    cost charged twice (in + out). Trades whose window overruns the tape are dropped.
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
        price_ret = p[e + horizon] / p[e] - 1.0
        short_ret = -price_ret         # short P&L
        out.append(short_ret - 2.0 * cost_bps * 1e-4)
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


def scrambled_highs_placebo(close: pd.Series, horizon: int, k: int = 5, lookback: int = 6,
                            flat_tol: float = 0.02, min_drop: float = 0.01,
                            n_draws: int = 1000, seed: int = 461) -> dict:
    """Placebo: destroy the *descending-highs* constraint, keep the flat-support break.

    The descending triangle's distinguishing geometry is the **falling ceiling** above a flat
    floor. This placebo permutes the swing-**high** prices (positions kept, marginal kept) so the
    "highs descend" test becomes meaningless, while the flat-support detection and break logic are
    unchanged. If the real break-down result survives the scramble, the triangle's defining
    descending-highs geometry was never load-bearing — it was just a support break. Returns the
    share of placebo runs whose mean short return **beats** the real one, plus the observed mean.
    """
    obs = float(np.mean(forward_returns(
        close, support_break_entries(close, k=k, lookback=lookback,
                                     flat_tol=flat_tol, min_drop=min_drop), horizon)))
    piv = find_pivots(close, k=k)
    if len(piv) < 4:
        return {"obs": obs, "p_value": float("nan"), "n_draws": 0}
    rng = np.random.default_rng(seed)
    high_idx = [j for j, kd in enumerate(piv["kind"].to_numpy()) if kd > 0]
    high_prices = piv["price"].to_numpy(dtype=float)[high_idx]
    idx = close.index
    n = len(close)
    beats = 0
    valid = 0
    for _ in range(n_draws):
        perm = rng.permutation(high_prices)
        sp = piv.copy()
        prices = sp["price"].to_numpy(dtype=float).copy()
        for slot, j in enumerate(high_idx):
            prices[j] = perm[slot]
        sp["price"] = prices
        supser = _support_from_pivots(sp, n, idx, k=k, lookback=lookback,
                                      flat_tol=flat_tol, min_drop=min_drop)
        mask = (close < supser) & supser.notna()
        first = mask & ~mask.shift(1, fill_value=False)
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
def run_experiment(close: pd.Series, k: int = 5, lookback: int = 6, cost_bps: float = 1.0,
                   random_seed: int = 7) -> dict:
    """Run the full gauntlet on one tape: support-break SHORT vs random-entry baseline, all H.

    Returns a dict keyed by horizon with the break-down short summary (gross + net), the
    drift-matched random-entry baseline, and the break-minus-random delta.
    """
    ent = support_break_entries(close, k=k, lookback=lookback)
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
