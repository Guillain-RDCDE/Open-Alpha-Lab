"""Gartley / AB=CD harmonic patterns as a falsifiable mechanical rule — Study 468.

Harmonic-pattern trading (H.M. Gartley's 1935 figure, codified into ratios by Scott Carney,
Larry Pesavento and others) reads a five-point zig-zag **X-A-B-C-D** of swing pivots and
declares point **D** a reversal *iff* the retracement ratios of the legs land on the Fibonacci
grid:

* **B** retraces ``XA`` by ≈ **0.618**;
* **C** retraces ``AB`` by 0.382–0.886;
* **D** retraces ``XA`` by ≈ **0.786** (and ``CD`` ≈ ``AB`` — the "AB=CD" symmetry),
  with the deeper ``BC`` projection (1.272–1.618).

The folklore (every harmonic-trading course, and the indicators baked into TradingView and
MetaTrader): when the ratios are satisfied, **D is a high-probability reversal** — a bullish
Gartley completing at a swing low is a buy.

We encode the tightest mechanical version a proponent would accept and test it honestly:

1. **Pivots are confirmed fractals** — a local extremum with ``k`` strictly-lower/higher bars
   on each side. A pivot at bar ``t`` is only *usable* from bar ``t+k`` onward (the confirmation
   lag), so a pattern is never recognised with future bars.
2. **XABCD zig-zag** — alternating swings X(low) A(high) B(low) C(high) D(low) for a *bullish*
   Gartley; the four legs' ratios are checked against the Fibonacci grid with a tolerance band.
3. **D-point long** — when a valid bullish XABCD completes (its D pivot just confirmed), a long
   fires; entry is at the **next** close (one documented lag); we then measure the forward
   H-day return.
4. **Controls.** (a) a **random-entry** baseline (same instrument, epoch, hold) that captures
   the tape's drift, and (b) a **ratio-grid placebo** that replaces the Fibonacci grid with a
   *random* ratio grid (same tolerance, same zig-zag machinery), destroying the harmonic
   structure while keeping the pivots — the honest "are the Fibonacci ratios load-bearing?" null.

No look-ahead: pivots carry a ``k``-bar confirmation lag, completion is read on the close of
*t*, the position is entered at the close of *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)

# The Gartley/AB=CD Fibonacci grid (bullish): (target ratio, tolerance) for each leg constraint.
# B = 0.618 of XA ; C = 0.382..0.886 of AB ; D = 0.786 of XA.
GARTLEY = {
    "B_of_XA": (0.618, 0.10),
    "C_of_AB_lo": 0.382,
    "C_of_AB_hi": 0.886,
    "D_of_XA": (0.786, 0.12),
}


# --------------------------------------------------------------------------- #
# Pivot detection (confirmed fractals)
# --------------------------------------------------------------------------- #
def find_pivots(close: pd.Series, k: int = 4) -> pd.DataFrame:
    """Confirmed swing pivots: local extrema with ``k`` strictly-beaten bars on each side.

    Returns a DataFrame indexed by the pivot bar with columns ``price`` and ``kind``
    (+1 = swing high, -1 = swing low). A pivot at position ``i`` is only *confirmed* (knowable)
    at bar ``i + k``; downstream code enforces that lag, so no future data leaks in.
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
# Harmonic-pattern geometry
# --------------------------------------------------------------------------- #
def _is_bullish_xabcd(X, A, B, C, D, grid=GARTLEY) -> bool:
    """True iff prices (X,A,B,C,D) form a *bullish Gartley* under the Fibonacci ``grid``.

    Bullish geometry: X is a low, A a high, B a low, C a high, D a low, with
    X < A, B < A, B > X (B above X), C < A, C > B, D < C, D > X is NOT required (D may undercut).
    Ratios checked: AB/XA ≈ B_of_XA; BC/AB in [C_lo, C_hi]; AD/XA ≈ D_of_XA.
    """
    xa = A - X
    if xa <= 0:
        return False
    ab = A - B
    if ab <= 0:
        return False
    bc = C - B
    if bc <= 0:
        return False
    cd = C - D
    if cd <= 0:
        return False
    r_ab = ab / xa                      # B retraces XA
    r_bc = bc / ab                      # C retraces AB
    r_ad = (A - D) / xa                 # D retraces XA (measured from A)
    tb, tolb = grid["B_of_XA"]
    td, told = grid["D_of_XA"]
    if abs(r_ab - tb) > tolb:
        return False
    if not (grid["C_of_AB_lo"] <= r_bc <= grid["C_of_AB_hi"]):
        return False
    if abs(r_ad - td) > told:
        return False
    return True


def detect_completions(close: pd.Series, k: int = 4, grid=GARTLEY,
                       lookback: int = 8) -> pd.DatetimeIndex:
    """Bars at which a bullish Gartley XABCD *completes* (its D pivot just got confirmed).

    Walk the confirmed, alternating pivots. At every swing-**low** pivot ``D``, search the recent
    pivots for an X(low)-A(high)-B(low)-C(high) prefix (within ``lookback`` pivots) whose
    retracement ratios — together with ``D`` — satisfy the Fibonacci ``grid``. This windowed
    search is how harmonic scanners work in practice (the five points need not be immediately
    adjacent, only correctly ordered and alternating). The pattern's signal date is the bar at
    which D is **confirmed** (D's bar + k) — never earlier, so no look-ahead. Entry is executed at
    the next close by :func:`forward_returns`.
    """
    piv = _alternating(find_pivots(close, k=k))
    if len(piv) < 5:
        return pd.DatetimeIndex([])
    idx = close.index
    positions = list(piv.index)
    prices = piv["price"].to_numpy(dtype=float)
    kinds = piv["kind"].to_numpy(dtype=int)
    sig_bars = []
    for jd in range(4, len(piv)):
        if kinds[jd] != -1:               # D must be a swing low
            continue
        D = prices[jd]
        matched = False
        # candidate C = a swing high before D; B = a low before C; A = a high before B; X = low before A
        for jc in range(jd - 1, max(jd - 1 - lookback, 0), -1):
            if kinds[jc] != +1:
                continue
            for jb in range(jc - 1, max(jc - 1 - lookback, 0), -1):
                if kinds[jb] != -1:
                    continue
                for ja in range(jb - 1, max(jb - 1 - lookback, 0), -1):
                    if kinds[ja] != +1:
                        continue
                    for jx in range(ja - 1, max(ja - 1 - lookback, 0), -1):
                        if kinds[jx] != -1:
                            continue
                        X, A, B, C = prices[jx], prices[ja], prices[jb], prices[jc]
                        if _is_bullish_xabcd(X, A, B, C, D, grid=grid):
                            matched = True
                            break
                    if matched:
                        break
                if matched:
                    break
            if matched:
                break
        if matched:
            d_pos = int(positions[jd])
            confirm_pos = d_pos + k
            if confirm_pos < len(idx):
                sig_bars.append(confirm_pos)
    if not sig_bars:
        return pd.DatetimeIndex([])
    sig_bars = sorted(set(sig_bars))
    return pd.DatetimeIndex([idx[b] for b in sig_bars])


def random_entries(close: pd.Series, n: int, k: int = 4, seed: int = 0) -> pd.DatetimeIndex:
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


def ratio_grid_placebo(close: pd.Series, horizon: int, k: int = 4,
                       n_draws: int = 500, seed: int = 468) -> dict:
    """Placebo: replace the Fibonacci grid with a *random* ratio grid, keeping all machinery.

    The harmonic claim is that the **specific Fibonacci ratios** (0.618 / 0.786) forecast the
    D-turn. The placebo swaps those targets for random ratios drawn from the same plausible band
    (0.3-0.9), with the same tolerance and the same XABCD zig-zag detector. If price truly turns
    at *Fibonacci* D-points, the real grid should beat almost every random grid. Returns the share
    of random grids whose mean D-point forward return **beats** the real one — the honest
    "are the Fibonacci ratios load-bearing?" p-value, plus the observed mean.
    """
    real_ent = detect_completions(close, k=k, grid=GARTLEY)
    obs = float(np.mean(forward_returns(close, real_ent, horizon))) if len(real_ent) else float("nan")
    if not np.isfinite(obs):
        return {"obs": obs, "p_value": float("nan"), "n_draws": 0}
    rng = np.random.default_rng(seed)
    beats = 0
    valid = 0
    for _ in range(n_draws):
        b_t = rng.uniform(0.35, 0.85)
        d_t = rng.uniform(0.45, 0.95)
        c_lo = rng.uniform(0.2, 0.5)
        c_hi = c_lo + rng.uniform(0.2, 0.5)
        grid = {"B_of_XA": (b_t, 0.075), "D_of_XA": (d_t, 0.075),
                "C_of_AB_lo": c_lo, "C_of_AB_hi": c_hi}
        ent = detect_completions(close, k=k, grid=grid)
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
def run_experiment(close: pd.Series, k: int = 4, cost_bps: float = 1.0,
                   random_seed: int = 7) -> dict:
    """Run the full gauntlet on one tape: D-point long vs random-entry baseline, all horizons.

    Returns a dict keyed by horizon with the D-point summary (gross + net), the drift-matched
    random-entry baseline, and the D-minus-random delta.
    """
    ent = detect_completions(close, k=k)
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
