"""Detector + timing rule + honest arbiters — Study 411 (Ascending Triangle).

The ascending triangle is one of the textbook bullish *continuation* figures (Edwards & Magee,
*Technical Analysis of Stock Trends*, 1948). The folk recipe:

1. A **flat resistance** ceiling: the price runs into the same horizontal level several times
   and is turned back (a band of swing highs at roughly the same price).
2. A **rising lower trendline**: each pullback bottoms *higher* than the last, so a line through
   the swing lows slopes up toward the resistance — the triangle "tightens" into an apex.
3. A **breakout**: the price finally closes **above** the resistance. The folklore says the
   figure resolves *upward* far more often than down, so you **buy the breakout** and it runs.

Chart figures are partly in the eye of the beholder, so we test the closest **mechanical**
definition we can write down and we say so. Our detector, on daily closes:

* Find **swing pivots** (local highs/lows over a symmetric window).
* A *flat top* = ``min_taps`` swing highs inside a window that all sit within ``flat_tol`` of
  their mean (a horizontal resistance the price tapped repeatedly).
* A *rising floor* = the swing lows between the first and last tap trend **up** — the line
  through them has a positive slope and the last low is meaningfully above the first.
* A **confirmed breakout** = the first close after the last tap that exceeds the resistance by
  ``breakout_buf``. That is the entry signal.

Timing: the breakout is known at its close; we **enter the next close** (one documented lag) and
hold ``H`` trading days. The honest question: do forward returns after a *confirmed* ascending-
triangle breakout beat the name's own base rate? Arbiters: a one-sample / HAC *t* of the
post-breakout excess over the base rate, a **same-tape label-shuffle placebo** (random "breakout"
dates on the same tape), costs, and the synthetic positive control. A natural myth-check rides
along: the *downward* break of the same figure (does it really break up far more than down?).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 40)        # trading-day forward horizons


# --------------------------------------------------------------------------- #
# Swing pivots
# --------------------------------------------------------------------------- #
def swing_pivots(close: np.ndarray, w: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Indices of local-high and local-low swing pivots on ``close``.

    A bar ``i`` is a swing high if ``close[i]`` is the max over the symmetric window
    ``[i-w, i+w]`` (and that max sits at the centre), a swing low if it is the min. Returns
    ``(high_idx, low_idx)`` as sorted integer arrays. The window means a pivot is only
    confirmable ``w`` bars later — the detector respects that lag downstream.
    """
    n = len(close)
    highs, lows = [], []
    for i in range(w, n - w):
        seg = close[i - w:i + w + 1]
        if close[i] == seg.max() and np.argmax(seg) == w:
            highs.append(i)
        if close[i] == seg.min() and np.argmin(seg) == w:
            lows.append(i)
    return np.array(highs, dtype=int), np.array(lows, dtype=int)


# --------------------------------------------------------------------------- #
# Ascending-triangle detector
# --------------------------------------------------------------------------- #
def detect_triangles(close: np.ndarray, w: int = 5, min_taps: int = 3,
                     tri_min: int = 25, tri_max: int = 160, flat_tol: float = 0.04,
                     min_rise: float = 0.05, breakout_buf: float = 0.0,
                     side: str = "up", search_buf: int = 30) -> list[dict]:
    """Detect confirmed ascending-triangle breakouts on a close series.

    For each starting swing high, gather the run of subsequent swing highs that stay within
    ``flat_tol`` of the running mean (the *flat resistance taps*). Require at least ``min_taps``
    taps spanning ``tri_min..tri_max`` bars. Then check the swing **lows** between the first and
    last tap form a **rising** floor: the last qualifying low must be at least ``min_rise`` above
    the first, and the floor must approach (but stay below) the ceiling. Finally, scan forward
    from the last tap for the first close that clears the resistance by ``breakout_buf``
    (``side="up"``) — or, for the myth-check, the first close that breaks *below* the rising
    floor by ``breakout_buf`` (``side="down"``).

    Returns a list of dicts, one per confirmed breakout, with integer positions: ``first_tap``,
    ``last_tap``, ``breakout_idx`` (the signal bar), ``resistance`` level, ``rise`` (low-to-low
    rise fraction), ``n_taps``. Non-overlapping: once a breakout is taken the scan resumes after
    it. Everything uses data up to each bar; the breakout is the first qualifying close.
    """
    if side not in ("up", "down"):
        raise ValueError(f"side must be 'up' or 'down', got {side!r}")
    n = len(close)
    highs, lows = swing_pivots(close, w)
    tris: list[dict] = []
    used_until = -1

    for hi_start in range(len(highs)):
        first = highs[hi_start]
        if first <= used_until:
            continue
        # gather the flat-resistance taps: consecutive swing highs near the running mean
        taps = [first]
        level = close[first]
        for j in range(hi_start + 1, len(highs)):
            hj = highs[j]
            if hj - first > tri_max:
                break
            if abs(close[hj] - level) / level <= flat_tol:
                taps.append(hj)
                level = float(np.mean(close[taps]))   # running resistance estimate
        if len(taps) < min_taps:
            continue
        first_tap, last_tap = taps[0], taps[-1]
        span = last_tap - first_tap
        if span < tri_min or span > tri_max:
            continue
        resistance = float(np.mean(close[taps]))
        # rising floor: swing lows strictly between first and last tap
        inner = lows[(lows > first_tap) & (lows < last_tap)]
        if len(inner) < 2:
            continue
        first_low, last_low = close[inner[0]], close[inner[-1]]
        rise = (last_low - first_low) / first_low
        if rise < min_rise:
            continue
        # the floor must stay under the ceiling (a triangle, not a box)
        if last_low >= resistance * (1.0 - 0.005):
            continue
        # slope of a line through the inner lows must be positive
        slope = np.polyfit(inner.astype(float), close[inner], 1)[0]
        if slope <= 0:
            continue
        # breakout scan from just after the last tap
        bo = None
        scan_end = min(last_tap + search_buf, n)
        for k in range(last_tap + 1, scan_end):
            if side == "up" and close[k] > resistance * (1.0 + breakout_buf):
                bo = k
                break
            if side == "down":
                # rising floor extrapolated to bar k
                floor_k = first_low + slope * (k - inner[0])
                if close[k] < floor_k * (1.0 - breakout_buf):
                    bo = k
                    break
        if bo is None:
            continue
        tris.append({
            "first_tap": int(first_tap), "last_tap": int(last_tap),
            "breakout_idx": int(bo), "resistance": resistance,
            "rise": float(rise), "n_taps": int(len(taps)), "side": side,
        })
        used_until = bo
    return tris


# --------------------------------------------------------------------------- #
# Forward returns after the breakout
# --------------------------------------------------------------------------- #
def forward_returns(close: np.ndarray, signal_idx: list[int], horizon: int,
                    lag: int = 1) -> np.ndarray:
    """Forward ``horizon``-day return entered ``lag`` bars after each signal close.

    Signal known at ``signal_idx`` close; enter at ``signal_idx + lag`` close (one documented
    lag); exit ``horizon`` bars later. Drops signals whose window overruns the tape.
    """
    out = []
    n = len(close)
    for si in signal_idx:
        entry = si + lag
        exit_ = entry + horizon
        if entry < 0 or exit_ >= n:
            continue
        out.append(close[exit_] / close[entry] - 1.0)
    return np.asarray(out, dtype=float)


def base_rate(close: np.ndarray, horizon: int) -> float:
    """The name's own unconditional mean forward ``horizon``-day return (the base rate)."""
    if len(close) <= horizon + 1:
        return float("nan")
    r = close[horizon:] / close[:-horizon] - 1.0
    return float(np.mean(r))


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def one_sample_t(sample: np.ndarray, mu0: float = 0.0) -> float:
    """One-sample t of ``sample`` mean against ``mu0``. NaN if fewer than 2 points."""
    if len(sample) < 2:
        return float("nan")
    se = sample.std(ddof=1) / np.sqrt(len(sample))
    if se == 0:
        return float("nan")
    return float((sample.mean() - mu0) / se)


def hac_t(sample: np.ndarray, mu0: float = 0.0) -> float:
    """Newey-West (HAC) t of the mean against ``mu0`` — autocorrelation-robust.

    Events are roughly ordered in time; we use a small Bartlett lag so clustered breakouts don't
    inflate the t. No quantlab dependency.
    """
    r = np.asarray(sample, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 6:
        return one_sample_t(r, mu0)
    mu = r.mean()
    e = r - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        wgt = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * wgt * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    if se == 0:
        return float("nan")
    return float((mu - mu0) / se)


# --------------------------------------------------------------------------- #
# Panel-level orchestration
# --------------------------------------------------------------------------- #
def collect_breakouts(panel: dict, names: list[str] | None = None,
                      **detect_kw) -> dict[str, list[dict]]:
    """Run :func:`detect_triangles` on every name in the panel; return ticker -> breakout list."""
    closes = panel["close"]
    if names is None:
        names = list(closes.columns)
    out = {}
    for tk in names:
        c = closes[tk].dropna().to_numpy(dtype=float)
        out[tk] = detect_triangles(c, **detect_kw)
    return out


def run_experiment(panel: dict, horizon: int = 20, names: list[str] | None = None,
                   cost_bps: float = 5.0, n_draws: int = 5000, lag: int = 1,
                   excess: bool = True, seed: int = 411, side: str = "up",
                   **detect_kw) -> dict:
    """Pool every confirmed breakout across the panel and arbitrate the forward edge.

    For each name: detect ascending triangles, take the forward-``horizon`` return after each
    breakout (1-day lag), and subtract the name's own base rate (``excess=True``) so the test is
    "does the figure beat buy-and-hold *for that name*", not "is the market up". Pools the
    per-event excess across names; reports n, mean, win-rate, one-sample t, HAC t, a same-tape
    placebo p, and the net-of-cost mean (one round trip = 2 * cost_bps one-way).

    The same-tape placebo draws random entry dates (same count per name, same base-rate
    subtraction) and asks how often a random set beats the observed mean — the honest control
    for the tape's own up-drift.
    """
    closes = panel["close"]
    if names is None:
        names = list(closes.columns)
    pooled, raw_pooled = [], []
    n_break = 0
    placebo_means = []
    rng = np.random.default_rng(seed)
    for tk in names:
        c = closes[tk].dropna().to_numpy(dtype=float)
        tris = detect_triangles(c, side=side, **detect_kw)
        n_break += len(tris)
        sig = [d["breakout_idx"] for d in tris]
        fr = forward_returns(c, sig, horizon, lag=lag)
        if len(fr) == 0:
            continue
        br = base_rate(c, horizon)
        raw_pooled.extend(fr.tolist())
        pooled.extend((fr - (br if excess else 0.0)).tolist())
        n = len(c)
        hi = n - horizon - lag - 1
        if hi > 1 and len(fr) > 0:
            for _ in range(n_draws):
                si = rng.integers(1, hi, size=len(fr))
                entry = si + lag
                exit_ = entry + horizon
                r = c[exit_] / c[entry] - 1.0
                placebo_means.append(r.mean() - (br if excess else 0.0))
    pooled = np.asarray(pooled, dtype=float)
    raw_pooled = np.asarray(raw_pooled, dtype=float)
    if len(pooled) == 0:
        return {"horizon": horizon, "n_breakouts": int(n_break), "n_events": 0,
                "mean": float("nan"), "raw_mean": float("nan"), "win": float("nan"),
                "t": float("nan"), "hac_t": float("nan"), "p_placebo": float("nan"),
                "net": float("nan")}
    obs = float(pooled.mean())
    cost = 2.0 * cost_bps / 1e4
    p = (float(np.mean(np.asarray(placebo_means) >= obs))
         if placebo_means else float("nan"))
    return {
        "horizon": horizon, "n_breakouts": int(n_break), "n_events": int(len(pooled)),
        "mean": obs, "raw_mean": float(raw_pooled.mean()),
        "win": float((pooled > 0).mean()),
        "t": one_sample_t(pooled), "hac_t": hac_t(pooled),
        "p_placebo": p, "net": float(obs - cost),
    }
