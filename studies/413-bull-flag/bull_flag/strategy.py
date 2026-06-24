"""Detector + timing rule + honest arbiters — Study 413 (Bull Flag).

The bull flag is one of the textbook bullish *continuation* figures (Edwards & Magee,
*Technical Analysis of Stock Trends*, 1948; Bulkowski, *Encyclopedia of Chart Patterns*). The
folk recipe:

1. A **flagpole**: a sharp, near-vertical rally over a handful of sessions — the move that
   "raises the flag."
2. A **flag**: a brief, shallow consolidation that drifts gently *down* or sideways in a tight
   channel — a pause that "lets the move catch its breath," giving back only a fraction of the
   pole.
3. A **breakout**: the price closes back **above** the flag's high. The folklore says the figure
   then *resumes the prior uptrend* — so you **buy the breakout** and ride the second leg, often
   with a target of "another flagpole's worth."

Chart figures are partly in the eye of the beholder, so we test the closest **mechanical**
definition we can write down and we say so. Our detector, on daily closes:

* A **flagpole** = a run of ``pole_w`` bars whose net rise clears ``min_pole`` (a steep rally).
* A **flag** = the next ``flag_min..flag_max`` bars consolidate: the pullback gives back no more
  than ``max_retrace`` of the pole, the flag's range is tight (``max_flag_range``), and its net
  drift is not strongly *up* (a flat-to-down pause, not a fresh rally).
* A **confirmed breakout** = the first close after the flag that exceeds the flag's high by
  ``breakout_buf``. That is the entry signal.

Timing: the breakout is known at its close; we **enter the next close** (one documented lag) and
hold ``H`` trading days. The honest question: do forward returns after a *confirmed* bull-flag
breakout beat the name's own base rate? Arbiters: a one-sample / HAC *t* of the post-breakout
excess over the base rate, a **same-tape label-shuffle placebo** (random "breakout" dates on the
same tape), costs, and the synthetic positive control. A natural myth-check rides along: the
*downward* break of the same flag (does the flag really resolve up far more than down?).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 40)        # trading-day forward horizons


# --------------------------------------------------------------------------- #
# Bull-flag detector
# --------------------------------------------------------------------------- #
def detect_flags(close: np.ndarray, pole_w: int = 12, min_pole: float = 0.12,
                 flag_min: int = 5, flag_max: int = 20, max_retrace: float = 0.5,
                 max_flag_range: float = 0.10, max_flag_drift: float = 0.03,
                 breakout_buf: float = 0.0, side: str = "up",
                 search_buf: int = 15) -> list[dict]:
    """Detect confirmed bull-flag breakouts on a close series.

    Slide a ``pole_w``-bar window; where the net rise over it clears ``min_pole`` (a flagpole),
    test the bars that follow for a **flag**: a consolidation lasting ``flag_min..flag_max`` bars
    in which (a) the deepest pullback from the pole top gives back no more than ``max_retrace`` of
    the pole height, (b) the flag's high-to-low range is at most ``max_flag_range`` of the pole
    top (a *tight* pause), and (c) the flag does not itself drift strongly *up*
    (``net drift <= max_flag_drift`` — a flat-to-down pause, not a second rally). Then scan
    forward for the first close that clears the flag's high by ``breakout_buf`` (``side="up"``)
    — or, for the myth-check, the first close that breaks *below* the flag's low by
    ``breakout_buf`` (``side="down"``).

    Returns a list of dicts, one per confirmed breakout, with integer positions: ``pole_start``,
    ``pole_top`` (the bar), ``flag_end``, ``breakout_idx`` (the signal bar), ``flag_high``,
    ``flag_low``, ``pole_ret``. Non-overlapping: once a breakout is taken the scan resumes after
    it. Everything uses data up to each bar; the breakout is the first qualifying close.
    """
    if side not in ("up", "down"):
        raise ValueError(f"side must be 'up' or 'down', got {side!r}")
    n = len(close)
    flags: list[dict] = []
    i = pole_w
    while i < n - flag_min - 2:
        pole_start = i - pole_w
        pole_bot = close[pole_start]
        pole_top_px = close[i]
        pole_ret = pole_top_px / pole_bot - 1.0
        if pole_ret < min_pole:
            i += 1
            continue
        pole_height = pole_top_px - pole_bot
        # try flag lengths from flag_min..flag_max
        found = None
        for fw in range(flag_min, flag_max + 1):
            flag_end = i + fw
            if flag_end >= n - 2:
                break
            seg = close[i + 1:flag_end + 1]            # flag bars (after the pole top)
            if len(seg) < flag_min:
                break
            flag_high = float(seg.max())
            flag_low = float(seg.min())
            # (a) retrace: deepest pullback from the pole top, as a fraction of pole height
            retrace = (pole_top_px - flag_low) / pole_height if pole_height > 0 else 1.0
            if retrace > max_retrace:
                continue
            # (b) tightness: the flag's own range, as a fraction of the pole top
            flag_range = (flag_high - flag_low) / pole_top_px
            if flag_range > max_flag_range:
                continue
            # (c) the flag should not be a fresh rally (net drift not strongly up)
            flag_drift = seg[-1] / seg[0] - 1.0
            if flag_drift > max_flag_drift:
                continue
            found = {"flag_end": flag_end, "flag_high": flag_high, "flag_low": flag_low}
            break
        if found is None:
            i += 1
            continue
        flag_end = found["flag_end"]
        flag_high = found["flag_high"]
        flag_low = found["flag_low"]
        # breakout scan from just after the flag
        bo = None
        scan_end = min(flag_end + search_buf, n)
        for k in range(flag_end + 1, scan_end):
            if side == "up" and close[k] > flag_high * (1.0 + breakout_buf):
                bo = k
                break
            if side == "down" and close[k] < flag_low * (1.0 - breakout_buf):
                bo = k
                break
        if bo is None:
            i += 1
            continue
        flags.append({
            "pole_start": int(pole_start), "pole_top": int(i),
            "flag_end": int(flag_end), "breakout_idx": int(bo),
            "flag_high": flag_high, "flag_low": flag_low,
            "pole_ret": float(pole_ret), "side": side,
        })
        i = bo + 1          # non-overlapping: resume after the breakout
    return flags


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
    """Run :func:`detect_flags` on every name in the panel; return ticker -> breakout list."""
    closes = panel["close"]
    if names is None:
        names = list(closes.columns)
    out = {}
    for tk in names:
        c = closes[tk].dropna().to_numpy(dtype=float)
        out[tk] = detect_flags(c, **detect_kw)
    return out


def run_experiment(panel: dict, horizon: int = 20, names: list[str] | None = None,
                   cost_bps: float = 5.0, n_draws: int = 5000, lag: int = 1,
                   excess: bool = True, seed: int = 413, side: str = "up",
                   **detect_kw) -> dict:
    """Pool every confirmed breakout across the panel and arbitrate the forward edge.

    For each name: detect bull flags, take the forward-``horizon`` return after each breakout
    (1-day lag), and subtract the name's own base rate (``excess=True``) so the test is "does the
    figure beat buy-and-hold *for that name*", not "is the market up". Pools the per-event excess
    across names; reports n, mean, win-rate, one-sample t, HAC t, a same-tape placebo p, and the
    net-of-cost mean (one round trip = 2 * cost_bps one-way).

    The same-tape placebo draws random entry dates (same count per name, same base-rate
    subtraction) and asks how often a random set beats the observed mean — the honest control for
    the tape's own up-drift.
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
        fls = detect_flags(c, side=side, **detect_kw)
        n_break += len(fls)
        sig = [d["breakout_idx"] for d in fls]
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
