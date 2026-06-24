"""Detector + timing rule + honest arbiters — Study 410 (Cup & Handle).

William O'Neil's **cup-with-handle** (from *How to Make Money in Stocks*, 1988) is the most
famous — and most subjective — of the classic chart figures. The folk recipe:

1. A **cup**: after an advance, the price rounds down and back up over weeks to months,
   forming a smooth U whose two rims sit at roughly the same level (the left peak = the
   resistance).
2. A **handle**: a short, shallow pullback off the right rim (a few percent), on lighter
   volume — the final shakeout.
3. A **breakout**: the price pushes **above the rim / handle high** ("the pivot"). O'Neil
   says you *buy the breakout* and it runs.

Chart figures are partly in the eye of the beholder, so we test the closest **mechanical**
definition we can write down and we say so. Our detector:

* Find **swing pivots** (local highs/lows over a symmetric window) on daily closes.
* A *cup* = a left-rim high, a trough at least ``min_depth`` below it, then a right-rim high
  back near the left rim (within ``rim_tol``), spanning ``cup_min..cup_max`` bars and
  U-shaped (the trough roughly central, not a sharp V).
* A *handle* = a shallow pullback (≤ ``handle_max_depth``) off the right rim, lasting a few
  bars, that does not undercut the cup's bottom half.
* A **confirmed breakout** = the first close after the handle that exceeds the rim resistance
  by ``breakout_buf``. That is the entry signal.

Timing: the breakout is known at its close; we **enter the next close** (one documented lag)
and hold ``H`` trading days. The honest question: do forward returns after a *confirmed*
cup-with-handle breakout beat the name's own base rate? Arbiters: a one-sample / HAC *t* of
the post-breakout excess over the base rate, a **label-shuffle placebo** (random "breakout"
dates on the same tape), costs, and the synthetic positive control.
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
    ``[i-w, i+w]`` (strictly the max, ties broken to the first), and a swing low if it is the
    min. Returns ``(high_idx, low_idx)`` as sorted integer arrays. The window means a pivot is
    only confirmable ``w`` bars later — the detector respects that lag downstream.
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
# Cup-with-handle detector
# --------------------------------------------------------------------------- #
def detect_cups(close: np.ndarray, w: int = 5, cup_min: int = 30, cup_max: int = 200,
                min_depth: float = 0.12, max_depth: float = 0.50, rim_tol: float = 0.06,
                handle_max_bars: int = 25, handle_max_depth: float = 0.15,
                breakout_buf: float = 0.0, center_tol: float = 0.40) -> list[dict]:
    """Detect confirmed cup-with-handle breakouts on a close series.

    Returns a list of dicts, one per confirmed breakout, with integer positions:
    ``left_idx`` (left rim), ``trough_idx``, ``right_idx`` (right rim), ``handle_low_idx``,
    ``breakout_idx`` (the bar whose close first clears the rim — the signal bar), plus the
    rim level and cup depth. Non-overlapping: once a breakout is taken the scan resumes after
    it. Everything is computed from data up to each bar; the breakout is the first qualifying
    close (no look-ahead in the signal).
    """
    n = len(close)
    highs, lows = swing_pivots(close, w)
    high_set = set(highs.tolist())
    cups: list[dict] = []
    used_until = -1

    for li in highs:
        if li <= used_until:
            continue
        left_rim = close[li]
        # candidate troughs after the left rim
        cand_lows = lows[(lows > li) & (lows < li + cup_max)]
        for ti in cand_lows:
            trough = close[ti]
            depth = (left_rim - trough) / left_rim
            if depth < min_depth or depth > max_depth:
                continue
            # right rim: first swing high after the trough back near the left rim
            cand_highs = highs[(highs > ti) & (highs < li + cup_max)]
            found = False
            for ri in cand_highs:
                right_rim = close[ri]
                cup_bars = ri - li
                if cup_bars < cup_min or cup_bars > cup_max:
                    continue
                if abs(right_rim - left_rim) / left_rim > rim_tol:
                    continue
                # U-shape: trough roughly central (not a sharp V on one side)
                center = (ti - li) / max(cup_bars, 1)
                if abs(center - 0.5) > center_tol:
                    continue
                rim = max(left_rim, right_rim)
                # handle: a shallow pullback off the right rim
                hstart = ri + 1
                hend = min(ri + handle_max_bars, n - 1)
                if hend - hstart < 2:
                    continue
                seg = close[hstart:hend + 1]
                handle_low = seg.min()
                handle_low_pos = hstart + int(np.argmin(seg))
                hdepth = (right_rim - handle_low) / right_rim
                if hdepth > handle_max_depth or hdepth <= 0:
                    continue
                # handle must not undercut the lower half of the cup
                if handle_low < trough + 0.5 * (left_rim - trough):
                    continue
                # breakout: first close after the handle low that clears the rim
                bo = None
                for k in range(handle_low_pos + 1, min(hend + handle_max_bars, n)):
                    if close[k] > rim * (1.0 + breakout_buf):
                        bo = k
                        break
                if bo is None:
                    continue
                cups.append({
                    "left_idx": int(li), "trough_idx": int(ti), "right_idx": int(ri),
                    "handle_low_idx": int(handle_low_pos), "breakout_idx": int(bo),
                    "rim": float(rim), "depth": float(depth),
                    "handle_depth": float(hdepth),
                })
                used_until = bo
                found = True
                break
            if found:
                break
    return cups


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

    Events are roughly ordered in time; we use a small Bartlett lag so clustered breakouts
    don't inflate the t. No quantlab dependency.
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


def placebo_pvalue(close: np.ndarray, n_signals: int, horizon: int, observed_mean: float,
                   n_draws: int = 5000, lag: int = 1, seed: int = 410) -> float:
    """Label-shuffle placebo: random "breakout" dates on the *same* tape.

    Draw ``n_signals`` random entry bars and measure the same forward-``horizon`` mean, many
    times. ``p`` = P[random mean >= observed mean] — the honest "could a random set of dates on
    this tape look this good?" null that controls for the tape's own drift/clustering.
    """
    n = len(close)
    hi = n - horizon - lag - 1
    if hi <= 1 or n_signals < 1:
        return float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_draws)
    for i in range(n_draws):
        si = rng.integers(1, hi, size=n_signals)
        entry = si + lag
        exit_ = entry + horizon
        r = close[exit_] / close[entry] - 1.0
        means[i] = r.mean()
    return float((means >= observed_mean).mean())


# --------------------------------------------------------------------------- #
# Panel-level orchestration
# --------------------------------------------------------------------------- #
def collect_breakouts(panel: dict, names: list[str] | None = None,
                      **detect_kw) -> dict[str, list[dict]]:
    """Run :func:`detect_cups` on every name in the panel; return ticker -> breakout list."""
    closes = panel["close"]
    if names is None:
        names = list(closes.columns)
    out = {}
    for tk in names:
        c = closes[tk].dropna().to_numpy(dtype=float)
        out[tk] = detect_cups(c, **detect_kw)
    return out


def run_experiment(panel: dict, horizon: int = 20, names: list[str] | None = None,
                   cost_bps: float = 5.0, n_draws: int = 5000, lag: int = 1,
                   excess: bool = True, seed: int = 410, **detect_kw) -> dict:
    """Pool every confirmed breakout across the panel and arbitrate the forward edge.

    For each name: detect cups, take the forward-``horizon`` return after each breakout (1-day
    lag), and subtract the name's own base rate (``excess=True``) so the test is "does the
    figure beat buy-and-hold *for that name*", not "is the market up". Pools the per-event
    excess across names; reports n, mean, win-rate, one-sample t, HAC t, a same-tape placebo p,
    and the net-of-cost mean (one round trip = 2 * cost_bps one-way).
    """
    closes = panel["close"]
    if names is None:
        names = list(closes.columns)
    pooled = []
    raw_pooled = []
    n_break = 0
    placebo_means = []
    rng = np.random.default_rng(seed)
    for tk in names:
        c = closes[tk].dropna().to_numpy(dtype=float)
        cups = detect_cups(c, **detect_kw)
        n_break += len(cups)
        sig = [d["breakout_idx"] for d in cups]
        fr = forward_returns(c, sig, horizon, lag=lag)
        if len(fr) == 0:
            continue
        br = base_rate(c, horizon)
        raw_pooled.extend(fr.tolist())
        pooled.extend((fr - (br if excess else 0.0)).tolist())
        # per-name placebo contribution: random dates, same count, same base-rate subtraction
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
