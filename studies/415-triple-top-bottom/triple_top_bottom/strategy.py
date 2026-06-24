"""Detector + timing rule + honest arbiters — Study 415 (Triple Top & Bottom).

The triple top and triple bottom are the textbook three-tap **reversal** figures (Edwards &
Magee, *Technical Analysis of Stock Trends*, 1948; Bulkowski, *Encyclopedia of Chart Patterns*).
The folk recipe:

* **Triple bottom (bullish).** The price falls to the same **support** level and bounces **three**
  times — three valleys at roughly one price, "three failures to break lower." Each bounce rallies
  up to a **neckline** (the highest high between the lows). When the price finally closes **above**
  the neckline, the figure is *confirmed*: the downtrend is exhausted, you **buy the breakout**.
* **Triple top (bearish).** The mirror image: three peaks at one **resistance** level, "three
  failures to break higher," and a confirmed close **below** the neckline (the lowest low between
  the peaks) triggers a **short**.

Chart figures are partly in the eye of the beholder — *"three failures at one level"* is exactly
the kind of phrase three chartists will draw three ways — so we test the closest **mechanical**
definition we can write down and we say so. Our detector, on daily closes:

* Find **swing pivots** (local highs/lows over a symmetric window).
* A *triple bottom* = three swing **lows** inside a window that all sit within ``level_tol`` of
  their mean (a horizontal support tapped three times), separated by genuine rallies (the
  intervening swing highs must rise at least ``min_bounce`` above the support).
* The **neckline** = the highest swing high between the first and last tap.
* A **confirmed breakout** = the first close after the last tap that clears the neckline by
  ``breakout_buf`` (for a triple top: the first close that breaks the neckline *down*). That is
  the entry signal.

Timing: the breakout is known at its close; we **enter the next close** (one documented lag) and
hold ``H`` trading days. The honest question: do forward returns after a *confirmed* triple-bottom
breakout beat the name's own base rate? Arbiters: a one-sample / HAC *t* of the post-breakout
excess over the base rate, a **same-tape label-shuffle placebo** (random "breakout" dates on the
same tape — the honest control for the tape's own up-drift), costs, a detector-strictness
robustness sweep, and the synthetic positive control. A natural myth-check rides along: the
**triple top** (the bearish twin) — does the figure really reverse symmetrically?
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
# Triple-top / triple-bottom detector
# --------------------------------------------------------------------------- #
def detect_triples(close: np.ndarray, w: int = 5, level_tol: float = 0.04,
                   min_bounce: float = 0.05, fig_min: int = 25, fig_max: int = 180,
                   breakout_buf: float = 0.0, side: str = "bottom",
                   search_buf: int = 30) -> list[dict]:
    """Detect confirmed triple-bottom (or triple-top) breakouts on a close series.

    ``side="bottom"`` (bullish): scan triplets of swing **lows**; ``side="top"`` (bearish): scan
    triplets of swing **highs**. For each candidate first tap, greedily gather the next two taps
    that sit within ``level_tol`` of the running mean of the level (three taps at one price). The
    three taps must span ``fig_min..fig_max`` bars. Between consecutive taps there must be a
    genuine *bounce* — for a bottom, an intervening swing **high** at least ``min_bounce`` above
    the support; for a top, an intervening swing **low** at least ``min_bounce`` below the
    resistance — so a flat shelf is not mistaken for the figure.

    The **neckline** is the most extreme intervening pivot (highest high for a bottom, lowest low
    for a top). The breakout is the first close after the third tap that clears the neckline by
    ``breakout_buf`` in the reversal direction (up for a bottom, down for a top), searched up to
    ``search_buf`` bars out. Non-overlapping: once a breakout is taken the scan resumes after it.

    Returns a list of dicts, one per confirmed breakout, with integer positions: ``taps`` (the
    three pivot indices), ``last_tap``, ``breakout_idx`` (the signal bar), ``level`` (the support
    or resistance), ``neckline``, ``side``.
    """
    if side not in ("bottom", "top"):
        raise ValueError(f"side must be 'bottom' or 'top', got {side!r}")
    n = len(close)
    highs, lows = swing_pivots(close, w)
    pivots = lows if side == "bottom" else highs       # the taps live on these pivots
    inner_pivots = highs if side == "bottom" else lows  # the bounces live on these

    out: list[dict] = []
    used_until = -1

    for a in range(len(pivots)):
        first = pivots[a]
        if first <= used_until:
            continue
        # greedily gather two more taps near the running level
        taps = [first]
        level = close[first]
        for b in range(a + 1, len(pivots)):
            pj = pivots[b]
            if pj - first > fig_max:
                break
            if abs(close[pj] - level) / level <= level_tol:
                taps.append(pj)
                level = float(np.mean(close[taps]))
                if len(taps) == 3:
                    break
        if len(taps) < 3:
            continue
        first_tap, last_tap = taps[0], taps[-1]
        span = last_tap - first_tap
        if span < fig_min or span > fig_max:
            continue
        level = float(np.mean(close[taps]))

        # genuine bounces between consecutive taps
        bounces = []
        ok = True
        for k in range(2):
            seg = inner_pivots[(inner_pivots > taps[k]) & (inner_pivots < taps[k + 1])]
            if len(seg) == 0:
                ok = False
                break
            if side == "bottom":
                peak = seg[np.argmax(close[seg])]
                if (close[peak] - level) / level < min_bounce:
                    ok = False
                    break
            else:
                trough = seg[np.argmin(close[seg])]
                if (level - close[trough]) / level < min_bounce:
                    ok = False
                    break
            bounces.append(int(seg[np.argmax(close[seg]) if side == "bottom"
                                  else np.argmin(close[seg])]))
        if not ok or len(bounces) < 2:
            continue

        # the neckline = most extreme intervening pivot
        if side == "bottom":
            neckline = float(np.max(close[bounces]))
        else:
            neckline = float(np.min(close[bounces]))

        # breakout scan from just after the third tap
        bo = None
        scan_end = min(last_tap + search_buf, n)
        for j in range(last_tap + 1, scan_end):
            if side == "bottom" and close[j] > neckline * (1.0 + breakout_buf):
                bo = j
                break
            if side == "top" and close[j] < neckline * (1.0 - breakout_buf):
                bo = j
                break
        if bo is None:
            continue

        out.append({
            "taps": [int(t) for t in taps], "last_tap": int(last_tap),
            "breakout_idx": int(bo), "level": level, "neckline": neckline,
            "side": side,
        })
        used_until = bo
    return out


# --------------------------------------------------------------------------- #
# Forward returns after the breakout
# --------------------------------------------------------------------------- #
def forward_returns(close: np.ndarray, signal_idx: list[int], horizon: int,
                    lag: int = 1, direction: int = 1) -> np.ndarray:
    """Forward ``horizon``-day return entered ``lag`` bars after each signal close.

    Signal known at ``signal_idx`` close; enter at ``signal_idx + lag`` close (one documented
    lag); exit ``horizon`` bars later. ``direction`` is +1 for a long (triple bottom) and −1 for
    a short (triple top) so the returned numbers are the **trade's** P&L in the believer's
    intended direction. Drops signals whose window overruns the tape.
    """
    out = []
    n = len(close)
    for si in signal_idx:
        entry = si + lag
        exit_ = entry + horizon
        if entry < 0 or exit_ >= n:
            continue
        out.append(direction * (close[exit_] / close[entry] - 1.0))
    return np.asarray(out, dtype=float)


def base_rate(close: np.ndarray, horizon: int, direction: int = 1) -> float:
    """The name's own unconditional mean forward ``horizon``-day return (the base rate).

    ``direction`` flips the sign for a short book so the excess is apples-to-apples (a short's
    base rate is the negative of buy-and-hold).
    """
    if len(close) <= horizon + 1:
        return float("nan")
    r = close[horizon:] / close[:-horizon] - 1.0
    return float(direction * np.mean(r))


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
                      side: str = "bottom", **detect_kw) -> dict[str, list[dict]]:
    """Run :func:`detect_triples` on every name in the panel; return ticker -> breakout list."""
    closes = panel["close"]
    if names is None:
        names = list(closes.columns)
    out = {}
    for tk in names:
        c = closes[tk].dropna().to_numpy(dtype=float)
        out[tk] = detect_triples(c, side=side, **detect_kw)
    return out


def run_experiment(panel: dict, horizon: int = 20, names: list[str] | None = None,
                   cost_bps: float = 5.0, n_draws: int = 5000, lag: int = 1,
                   excess: bool = True, seed: int = 415, side: str = "bottom",
                   **detect_kw) -> dict:
    """Pool every confirmed breakout across the panel and arbitrate the forward edge.

    For each name: detect triple bottoms (``side="bottom"``, a long) or triple tops
    (``side="top"``, a short), take the forward-``horizon`` return after each breakout in the
    trade's intended direction (1-day lag), and subtract the name's own (direction-matched) base
    rate (``excess=True``) so the test is "does the figure beat buy-and-hold *for that name*", not
    "is the market up". Pools the per-event excess across names; reports n, mean, win-rate,
    one-sample t, HAC t, a same-tape placebo p, and the net-of-cost mean (one round trip =
    2 * cost_bps one-way).

    The same-tape placebo draws random entry dates (same count per name, same base-rate
    subtraction, same direction) and asks how often a random set beats the observed mean — the
    honest control for the tape's own drift.
    """
    closes = panel["close"]
    if names is None:
        names = list(closes.columns)
    direction = 1 if side == "bottom" else -1
    pooled, raw_pooled = [], []
    n_break = 0
    placebo_means = []
    rng = np.random.default_rng(seed)
    for tk in names:
        c = closes[tk].dropna().to_numpy(dtype=float)
        trips = detect_triples(c, side=side, **detect_kw)
        n_break += len(trips)
        sig = [d["breakout_idx"] for d in trips]
        fr = forward_returns(c, sig, horizon, lag=lag, direction=direction)
        if len(fr) == 0:
            continue
        br = base_rate(c, horizon, direction=direction)
        raw_pooled.extend(fr.tolist())
        pooled.extend((fr - (br if excess else 0.0)).tolist())
        n = len(c)
        hi = n - horizon - lag - 1
        if hi > 1 and len(fr) > 0:
            for _ in range(n_draws):
                si = rng.integers(1, hi, size=len(fr))
                entry = si + lag
                exit_ = entry + horizon
                r = direction * (c[exit_] / c[entry] - 1.0)
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
