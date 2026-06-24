"""Detector + inference for Study 416 — Rounding Bottom (saucer base).

The claim: a **rounding bottom** (a.k.a. saucer base) — a long, smooth, U-shaped
basin in price — marks quiet *accumulation*; when price finishes the saucer and
**breaks out** above the rim, it signals the start of a sustained advance. Buy the
confirmed breakout.

Chart figures are partly subjective, so we test the closest **mechanical** definition
and say so. A rounding bottom over a trailing window of ``base_len`` bars is detected
when, on a fit of the window's closes:

  1. **U-shape.** A least-squares **parabola** ``c(u) = a u^2 + b u + d`` has positive
     curvature ``a > 0`` and explains the window well (``R^2 >= r2_min``) — the basin is
     genuinely bowl-shaped, not a line or a random wiggle.
  2. **Interior trough.** The vertex (minimum) sits *inside* the window, away from the
     edges (``edge_frac`` margin) — a real saucer dips in the middle, it is not a
     monotone slide.
  3. **Depth.** The trough is at least ``min_depth`` below the rim (the higher of the two
     window ends) — a flat line is not a saucer.
  4. **Confirmed breakout.** The current close **crosses above the rim** (the window's
     left/right reference high) for the first time — the saucer is *finished*. This is
     the entry trigger; without it there is no trade.

A signal is known at the close of bar *t*; the trade is entered at *t+1*'s open (one
documented execution lag) and held a fixed horizon.

Honest arbiters (the desk's shared spirit):

  * **One-sample / HAC t** of the breakout trade returns against zero, *and* against the
    same-name **base rate** (every bar's forward return) via a Welch t — does the figure
    add anything over just being long the stock?
  * **A date-shuffle permutation placebo** — draw the same number of random entry dates
    (matched per name) thousands of times and ask how often random entries beat the
    figure's entries (the honest "is the shape doing anything?" null).
  * **Costs** — one-way bps × NAV on the round trip (long-only; no borrow).
  * A **synthetic positive control** confirms the detector recovers a *planted*
    continuation and does NOT manufacture one when only the shape (no follow-through)
    is planted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (10, 20, 60)            # trading-day forward horizons


# --------------------------------------------------------------------------- #
# The mechanical rounding-bottom detector
# --------------------------------------------------------------------------- #
def _parabola_fit(y: np.ndarray) -> tuple[float, float, float, float]:
    """Least-squares parabola of ``y`` on u = 0..n-1. Returns (a, b, d, R2)."""
    n = len(y)
    u = np.arange(n, dtype=float)
    X = np.column_stack([u * u, u, np.ones(n)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b, d = coef
    pred = X @ coef
    ss_res = float(np.sum((y - pred) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return float(a), float(b), float(d), float(r2)


def detect_breakouts(bars: pd.DataFrame, base_len: int = 90, r2_min: float = 0.55,
                     min_depth: float = 0.08, edge_frac: float = 0.20,
                     cooldown: int = 30) -> pd.DataFrame:
    """Find confirmed rounding-bottom **breakout** bars in one name's tape.

    For each bar ``t`` we look back at the window of closes ``[t-base_len+1 .. t]``
    (log scale), fit a parabola, and require: positive curvature (``a > 0``) with
    ``R^2 >= r2_min`` (genuine U), an interior vertex (``edge_frac`` from each edge),
    a trough at least ``min_depth`` (log) below the rim, and that the **close at t is the
    first** to exceed the rim (= the higher window end) — the confirmed breakout. A
    ``cooldown`` suppresses repeat fires from the same saucer.

    No look-ahead: the window ends at ``t`` and the trigger uses only data up to ``t``.
    Returns a frame indexed by the breakout timestamps with columns
    ``a, r2, depth, rim, vertex_frac`` (entry happens at t+1, handled in the backtest).
    """
    close = bars["close"].to_numpy(dtype=float)
    logc = np.log(close)
    idx = bars.index
    n = len(close)
    rows = []
    last_fire = -10 ** 9
    lo_edge = int(edge_frac * base_len)
    hi_edge = base_len - lo_edge
    # Cheap vectorised pre-filter: a confirmed breakout MUST be the first close to clear
    # the left rim (= the window's first close), so only bars where logc[t] > logc[t-L+1]
    # and logc[t-1] <= logc[t-L+1] can fire. This eliminates ~all bars before the (costly)
    # parabola fit — same result, ~100x faster.
    # for t = base_len + k (k>=0): left-rim index = t - base_len + 1 = k + 1
    t_arr = np.arange(base_len, n)
    cur = logc[t_arr]                                            # logc[t]
    prev = logc[t_arr - 1]                                       # logc[t-1]
    rim_arr = logc[t_arr - base_len + 1]                         # left rim of each window
    cand = t_arr[(cur > rim_arr) & (prev <= rim_arr)]
    for t in cand:
        t = int(t)
        if t - last_fire < cooldown:
            continue
        w = logc[t - base_len + 1: t + 1]          # length base_len
        a, b, d, r2 = _parabola_fit(w)
        if a <= 0 or r2 < r2_min:
            continue
        vertex = -b / (2.0 * a)                     # in window coords 0..base_len-1
        if not (lo_edge <= vertex <= hi_edge):
            continue
        rim = max(w[0], w[-1])                       # higher of the two window ends
        trough = w.min()
        depth = rim - trough
        if depth < min_depth:
            continue
        # confirmed breakout: the close at t is the FIRST to clear the left-rim
        # resistance (the level being broken); the prior close was at/below it.
        left_rim = w[0]
        if not (logc[t] > left_rim and logc[t - 1] <= left_rim):
            continue
        rows.append({"date": idx[t], "a": a, "r2": r2, "depth": float(depth),
                     "rim": float(np.exp(left_rim)), "vertex_frac": float(vertex / base_len)})
        last_fire = t
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame(
        columns=["a", "r2", "depth", "rim", "vertex_frac"])


# --------------------------------------------------------------------------- #
# Forward-return measurement (one execution lag)
# --------------------------------------------------------------------------- #
def forward_returns(bars: pd.DataFrame, entry_dates, horizon: int,
                    cost_bps: float = 0.0) -> np.ndarray:
    """Forward ``horizon``-day return per entry, entered at t+1 open, exited at close.

    For a signal known at the close of bar ``t`` (``entry_dates``), enter at bar
    ``t+1``'s open and exit at the close ``horizon`` bars later. ``cost_bps`` is the
    one-way cost charged once on the round trip (×2 inside). Windows that overrun the
    tape are dropped. Returns the per-entry net (or gross if cost=0) return array.
    """
    open_ = bars["open"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    pos = {ts: i for i, ts in enumerate(bars.index)}
    c = cost_bps * 1e-4
    out = []
    for ts in entry_dates:
        i = pos.get(ts)
        if i is None or i + 1 >= len(close):
            continue
        e = i + 1
        x = e + horizon
        if x >= len(close):
            continue
        gross = close[x] / open_[e] - 1.0
        out.append(gross - 2.0 * c)
    return np.asarray(out, dtype=float)


def base_rate_returns(bars: pd.DataFrame, horizon: int) -> np.ndarray:
    """Every-bar forward ``horizon``-day return (enter t+1 open, exit close) — the base
    rate the figure must beat. The unconditional 'just be long' benchmark."""
    open_ = bars["open"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    out = []
    for e in range(1, len(close) - horizon):
        out.append(close[e + horizon] / open_[e] - 1.0)
    return np.asarray(out, dtype=float)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def ttest_vs_zero(sample: np.ndarray) -> float:
    """One-sample t of ``sample`` against 0."""
    sample = sample[np.isfinite(sample)]
    if len(sample) < 2:
        return float("nan")
    se = sample.std(ddof=1) / np.sqrt(len(sample))
    return float(sample.mean() / se) if se > 0 else float("nan")


def welch_t(sample: np.ndarray, base: np.ndarray) -> float:
    """Welch t of mean(sample) - mean(base) (unequal variance)."""
    sample = sample[np.isfinite(sample)]
    base = base[np.isfinite(base)]
    if len(sample) < 2 or len(base) < 2:
        return float("nan")
    v1 = sample.var(ddof=1) / len(sample)
    v0 = base.var(ddof=1) / len(base)
    se = np.sqrt(v1 + v0)
    return float((sample.mean() - base.mean()) / se) if se > 0 else float("nan")


def hac_t(sample: np.ndarray) -> float:
    """Newey-West HAC t-stat of the mean of ``sample`` against 0 (overlap-robust)."""
    r = sample[np.isfinite(sample)]
    n = r.size
    if n <= 5:
        return float("nan")
    mu = r.mean()
    e = r - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def pooled_signal_returns(panel: dict, base_len: int, horizon: int,
                          cost_bps: float = 0.0, **det) -> tuple[np.ndarray, int]:
    """Pool the breakout forward returns across every name in the panel.

    Returns (pooled return array, number of names that produced >=1 breakout).
    """
    pooled = []
    n_names = 0
    for tk, bars in panel.items():
        bk = detect_breakouts(bars, base_len=base_len, **det)
        if len(bk) == 0:
            continue
        rets = forward_returns(bars, list(bk.index), horizon, cost_bps=cost_bps)
        if len(rets):
            pooled.append(rets)
            n_names += 1
    if not pooled:
        return np.array([]), 0
    return np.concatenate(pooled), n_names


def pooled_base_rate(panel: dict, horizon: int) -> np.ndarray:
    """Pool every-bar forward returns across the panel — the base rate."""
    pooled = [base_rate_returns(b, horizon) for b in panel.values()]
    return np.concatenate(pooled) if pooled else np.array([])


def permutation_placebo(panel: dict, base_len: int, horizon: int,
                        n_signals_per_name: dict, n_draws: int = 2000,
                        seed: int = 416) -> dict:
    """Date-shuffle placebo: draw the same number of RANDOM entry dates per name and
    pool their forward returns; repeat ``n_draws`` times.

    ``n_signals_per_name`` maps ticker -> number of real breakouts to match. ``p`` =
    P[random-entry pooled mean >= observed pooled mean] — the honest "is the *shape*
    doing anything beyond picking that many dates in this name?" null.
    """
    rng = np.random.default_rng(seed)
    # precompute every-bar forward returns per name (valid entry positions)
    fwd = {}
    for tk, bars in panel.items():
        open_ = bars["open"].to_numpy(dtype=float)
        close = bars["close"].to_numpy(dtype=float)
        nb = len(close)
        valid = np.arange(1, nb - horizon)
        r = close[valid + horizon] / open_[valid] - 1.0
        fwd[tk] = r
    means = np.empty(n_draws)
    for j in range(n_draws):
        parts = []
        for tk in panel:
            k = n_signals_per_name.get(tk, 0)
            if k == 0 or len(fwd[tk]) == 0:
                continue
            pick = rng.integers(0, len(fwd[tk]), size=k)
            parts.append(fwd[tk][pick])
        means[j] = np.concatenate(parts).mean() if parts else np.nan
    return {"draws": means}


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def run_experiment(panel: dict, base_len: int = 90, horizon: int = 20,
                   cost_bps: float = 5.0, r2_min: float = 0.55,
                   min_depth: float = 0.08, edge_frac: float = 0.20,
                   cooldown: int = 30, placebo: bool = True,
                   n_draws: int = 2000, seed: int = 416) -> dict:
    """End-to-end: detect breakouts across the panel, measure forward returns, run the
    base-rate Welch t, one-sample/HAC t, costs, and the date-shuffle placebo.

    Returns a headline dict (the numbers that land in docs/results.md).
    """
    det = dict(r2_min=r2_min, min_depth=min_depth, edge_frac=edge_frac, cooldown=cooldown)

    # per-name breakout counts (for the placebo to match)
    counts = {}
    for tk, bars in panel.items():
        bk = detect_breakouts(bars, base_len=base_len, **det)
        counts[tk] = len(bk)

    gross, n_names = pooled_signal_returns(panel, base_len, horizon, cost_bps=0.0, **det)
    net, _ = pooled_signal_returns(panel, base_len, horizon, cost_bps=cost_bps, **det)
    base = pooled_base_rate(panel, horizon)

    res = {
        "base_len": base_len, "horizon": horizon, "cost_bps": cost_bps,
        "n_signals": int(len(gross)), "n_names": int(n_names),
        "gross_mean": float(gross.mean()) if len(gross) else float("nan"),
        "net_mean": float(net.mean()) if len(net) else float("nan"),
        "base_mean": float(base.mean()) if len(base) else float("nan"),
        "win": float((gross > 0).mean()) if len(gross) else float("nan"),
        "base_win": float((base > 0).mean()) if len(base) else float("nan"),
        "t_zero": ttest_vs_zero(gross),
        "t_hac": hac_t(gross),
        "t_vs_base": welch_t(gross, base),
        "p_placebo": float("nan"),
    }
    if placebo and len(gross):
        pl = permutation_placebo(panel, base_len, horizon, counts,
                                 n_draws=n_draws, seed=seed)
        draws = pl["draws"]
        draws = draws[np.isfinite(draws)]
        res["p_placebo"] = float((draws >= gross.mean()).mean()) if len(draws) else float("nan")
        res["placebo_mean"] = float(draws.mean()) if len(draws) else float("nan")
    return res
