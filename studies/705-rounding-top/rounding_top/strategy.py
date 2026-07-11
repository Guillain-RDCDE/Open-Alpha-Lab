"""Detector + inference for Study 705 — Rounding Top (dome distribution).

The claim: a **rounding top** (a.k.a. dome distribution) — a long, smooth,
inverted-U arc where price slowly rolls over on a declining-slope moving average —
marks quiet *distribution*; when price finishes the dome and **breaks down** below
the rim (the support level established at the start of the arc), it signals the
start of a sustained decline. Short the confirmed breakdown.

Chart figures are partly subjective, so we test the closest **mechanical**
definition and say so. A rounding top over a trailing window of ``base_len`` bars
is detected when, on a fit of the window's log closes:

  1. **Dome shape.** A least-squares **parabola** ``c(u) = a u^2 + b u + d`` has
     *negative* curvature ``a < 0`` (the slow-roll-over, declining-slope arc) and
     explains the window well (``R^2 >= r2_min``) — the top is genuinely
     dome-shaped, not a line or a random wiggle.
  2. **Interior peak.** The vertex (maximum) sits *inside* the window, away from
     the edges (``edge_frac`` margin) — a real dome peaks in the middle, it is not
     a monotone climb.
  3. **Height.** The peak is at least ``min_height`` above the support (the lower
     of the two window ends) — a flat line is not a dome.
  4. **Confirmed breakdown.** The current close **crosses below the support**
     (the window's left/starting reference level) for the first time — the dome
     is *finished*. This is the entry trigger; without it there is no trade.

A signal is known at the close of bar *t*; the trade is a **short** entered at
*t+1*'s open (one documented execution lag) and held a fixed horizon, charged
one-way costs per leg *and* an annualized borrow rate on the short (a house-rule
requirement absent from the long-only saucer study).

Honest arbiters (the desk's shared spirit):

  * **One-sample / HAC t** of the breakdown short-trade returns against zero, *and*
    against the same-name **base rate** (every bar's forward SHORT return) via a
    Welch t — does the figure add anything over just shorting on a random day
    (which, given the equity drift premium, loses money on average)?
  * **A date-shuffle permutation placebo** — draw the same number of random entry
    dates (matched per name), short them the same way, thousands of times, and ask
    how often random entries beat the figure's entries (the honest "is the shape
    doing anything?" null).
  * **Costs + borrow** — one-way bps x NAV on the round trip, plus an annualized
    borrow rate pro-rated over the holding horizon (short-only; no leverage).
  * A **synthetic positive control** confirms the detector recovers a *planted*
    decline and does NOT manufacture one when only the shape (no follow-through)
    is planted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (10, 20, 60)            # trading-day forward horizons


# --------------------------------------------------------------------------- #
# The mechanical rounding-top detector
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


def detect_breakdowns(bars: pd.DataFrame, base_len: int = 90, r2_min: float = 0.55,
                      min_height: float = 0.08, edge_frac: float = 0.20,
                      cooldown: int = 30) -> pd.DataFrame:
    """Find confirmed rounding-top **breakdown** bars in one name's tape.

    For each bar ``t`` we look back at the window of closes ``[t-base_len+1 .. t]``
    (log scale), fit a parabola, and require: negative curvature (``a < 0``) with
    ``R^2 >= r2_min`` (genuine dome), an interior vertex (``edge_frac`` from each
    edge), a peak at least ``min_height`` (log) above the support, and that the
    **close at t is the first** to break below the support (= the window's
    starting level) — the confirmed breakdown. A ``cooldown`` suppresses repeat
    fires from the same dome.

    No look-ahead: the window ends at ``t`` and the trigger uses only data up to
    ``t``. Returns a frame indexed by the breakdown timestamps with columns
    ``a, r2, height, support, vertex_frac`` (entry happens at t+1, handled in the
    backtest).
    """
    close = bars["close"].to_numpy(dtype=float)
    logc = np.log(close)
    idx = bars.index
    n = len(close)
    rows = []
    last_fire = -10 ** 9
    lo_edge = int(edge_frac * base_len)
    hi_edge = base_len - lo_edge
    # Cheap vectorised pre-filter: a confirmed breakdown MUST be the first close to
    # break below the left support (= the window's first close), so only bars where
    # logc[t] < logc[t-L+1] and logc[t-1] >= logc[t-L+1] can fire. This eliminates
    # ~all bars before the (costly) parabola fit — same result, ~100x faster.
    # for t = base_len + k (k>=0): left-support index = t - base_len + 1 = k + 1
    t_arr = np.arange(base_len, n)
    cur = logc[t_arr]                                            # logc[t]
    prev = logc[t_arr - 1]                                       # logc[t-1]
    supp_arr = logc[t_arr - base_len + 1]                        # left support of each window
    cand = t_arr[(cur < supp_arr) & (prev >= supp_arr)]
    for t in cand:
        t = int(t)
        if t - last_fire < cooldown:
            continue
        w = logc[t - base_len + 1: t + 1]          # length base_len
        a, b, d, r2 = _parabola_fit(w)
        if a >= 0 or r2 < r2_min:
            continue
        vertex = -b / (2.0 * a)                     # in window coords 0..base_len-1
        if not (lo_edge <= vertex <= hi_edge):
            continue
        support = min(w[0], w[-1])                   # lower of the two window ends
        peak = w.max()
        height = peak - support
        if height < min_height:
            continue
        # confirmed breakdown: the close at t is the FIRST to break the left-support
        # level (the level being broken); the prior close was at/above it.
        left_support = w[0]
        if not (logc[t] < left_support and logc[t - 1] >= left_support):
            continue
        rows.append({"date": idx[t], "a": a, "r2": r2, "height": float(height),
                     "support": float(np.exp(left_support)), "vertex_frac": float(vertex / base_len)})
        last_fire = t
    return pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame(
        columns=["a", "r2", "height", "support", "vertex_frac"])


# --------------------------------------------------------------------------- #
# Forward-return measurement (one execution lag, SHORT trade + borrow)
# --------------------------------------------------------------------------- #
def forward_returns(bars: pd.DataFrame, entry_dates, horizon: int,
                    cost_bps: float = 0.0, borrow_bps: float = 0.0) -> np.ndarray:
    """Forward ``horizon``-day SHORT return per entry, entered at t+1 open, exited
    at close.

    For a signal known at the close of bar ``t`` (``entry_dates``), short at bar
    ``t+1``'s open and cover at the close ``horizon`` bars later: gross short
    return = ``1 - close[t+1+H] / open[t+1]``. ``cost_bps`` is the one-way cost
    charged once on the round trip (x2 inside); ``borrow_bps`` is an ANNUALIZED
    stock-borrow rate pro-rated over the holding horizon (``horizon / 252``) — the
    house-rule cost a short position pays that a long position does not. Windows
    that overrun the tape are dropped. Returns the per-entry net (or gross if both
    costs are 0) return array.
    """
    open_ = bars["open"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    pos = {ts: i for i, ts in enumerate(bars.index)}
    c = cost_bps * 1e-4
    b = borrow_bps * 1e-4 * (horizon / 252.0)
    out = []
    for ts in entry_dates:
        i = pos.get(ts)
        if i is None or i + 1 >= len(close):
            continue
        e = i + 1
        x = e + horizon
        if x >= len(close):
            continue
        gross_short = 1.0 - close[x] / open_[e]
        out.append(gross_short - 2.0 * c - b)
    return np.asarray(out, dtype=float)


def base_rate_returns(bars: pd.DataFrame, horizon: int) -> np.ndarray:
    """Every-bar forward ``horizon``-day SHORT return (short t+1 open, cover close) —
    the base rate the figure must beat. The unconditional 'just short a random day'
    benchmark — negative on average because of the equity drift premium."""
    open_ = bars["open"].to_numpy(dtype=float)
    close = bars["close"].to_numpy(dtype=float)
    out = []
    for e in range(1, len(close) - horizon):
        out.append(1.0 - close[e + horizon] / open_[e])
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
                          cost_bps: float = 0.0, borrow_bps: float = 0.0,
                          **det) -> tuple[np.ndarray, int]:
    """Pool the breakdown forward SHORT returns across every name in the panel.

    Returns (pooled return array, number of names that produced >=1 breakdown).
    """
    pooled = []
    n_names = 0
    for tk, bars in panel.items():
        bk = detect_breakdowns(bars, base_len=base_len, **det)
        if len(bk) == 0:
            continue
        rets = forward_returns(bars, list(bk.index), horizon,
                               cost_bps=cost_bps, borrow_bps=borrow_bps)
        if len(rets):
            pooled.append(rets)
            n_names += 1
    if not pooled:
        return np.array([]), 0
    return np.concatenate(pooled), n_names


def pooled_base_rate(panel: dict, horizon: int) -> np.ndarray:
    """Pool every-bar forward SHORT returns across the panel — the base rate."""
    pooled = [base_rate_returns(b, horizon) for b in panel.values()]
    return np.concatenate(pooled) if pooled else np.array([])


def permutation_placebo(panel: dict, base_len: int, horizon: int,
                        n_signals_per_name: dict, n_draws: int = 2000,
                        seed: int = 705) -> dict:
    """Date-shuffle placebo: draw the same number of RANDOM entry dates and short
    them, mean forward SHORT return (points); repeat ``n_draws`` times.

    ``n_signals_per_name`` maps ticker -> number of real breakdowns to match. ``p``
    = P[random-entry pooled mean >= observed pooled mean] — the honest "is the
    *shape* doing anything beyond picking that many dates to short in this name?"
    null.
    """
    rng = np.random.default_rng(seed)
    # precompute every-bar forward SHORT returns per name (valid entry positions)
    fwd = {}
    for tk, bars in panel.items():
        open_ = bars["open"].to_numpy(dtype=float)
        close = bars["close"].to_numpy(dtype=float)
        nb = len(close)
        valid = np.arange(1, nb - horizon)
        r = 1.0 - close[valid + horizon] / open_[valid]
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
                   cost_bps: float = 5.0, borrow_bps: float = 30.0,
                   r2_min: float = 0.55, min_height: float = 0.08,
                   edge_frac: float = 0.20, cooldown: int = 30, placebo: bool = True,
                   n_draws: int = 2000, seed: int = 705) -> dict:
    """End-to-end: detect breakdowns across the panel, short-side measure forward
    returns, run the base-rate Welch t, one-sample/HAC t, costs+borrow, and the
    date-shuffle placebo.

    Returns a headline dict (the numbers that land in docs/results.md).
    """
    det = dict(r2_min=r2_min, min_height=min_height, edge_frac=edge_frac, cooldown=cooldown)

    # per-name breakdown counts (for the placebo to match)
    counts = {}
    for tk, bars in panel.items():
        bk = detect_breakdowns(bars, base_len=base_len, **det)
        counts[tk] = len(bk)

    gross, n_names = pooled_signal_returns(panel, base_len, horizon,
                                           cost_bps=0.0, borrow_bps=0.0, **det)
    net, _ = pooled_signal_returns(panel, base_len, horizon,
                                   cost_bps=cost_bps, borrow_bps=borrow_bps, **det)
    base = pooled_base_rate(panel, horizon)

    res = {
        "base_len": base_len, "horizon": horizon, "cost_bps": cost_bps,
        "borrow_bps": borrow_bps,
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


# --------------------------------------------------------------------------- #
# Synthetic-panel detector helper (mirrors run_experiment for the control cells)
# --------------------------------------------------------------------------- #
def synthetic_detect(panel: dict, base_len: int = 90, horizon: int = 20,
                     n_draws: int = 2000, seed: int = 705) -> dict:
    """Run the headline experiment (gross-only) on a synthetic panel."""
    return run_experiment(panel, base_len=base_len, horizon=horizon, cost_bps=0.0,
                          borrow_bps=0.0, n_draws=n_draws, seed=seed)
