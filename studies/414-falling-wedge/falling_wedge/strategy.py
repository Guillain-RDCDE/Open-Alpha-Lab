"""Detector + timing rule + honest arbiters — Study 414 (Falling Wedge).

The falling wedge is one of the textbook **bullish** chart figures (Edwards & Magee, *Technical
Analysis of Stock Trends*, 1948; Bulkowski, *Encyclopedia of Chart Patterns*). The folk recipe:

1. Price drifts **down** inside two **downward-sloping trendlines** — an upper line through the
   swing highs and a lower line through the swing lows.
2. The two lines **converge**: the highs fall *faster* than the lows, so the trading range
   narrows toward an apex. (That is what makes it a *wedge*, not a channel.)
3. A **breakout**: price finally closes **above** the upper (resistance) line. The folklore says
   a falling wedge resolves *upward* far more often than down — so you **buy the upside break**
   and it runs. It is sold as the bullish twin of the (bearish) rising wedge.

Chart figures are partly in the eye of the beholder, so we test the closest **mechanical**
definition we can write down and we say so. Our detector, on daily closes:

* Find **swing pivots** (local highs/lows over a symmetric window).
* Fit a line through a run of swing **highs** and a line through the intervening swing **lows**.
* Require **both slopes negative** (the whole thing slopes down), the **upper line steeper than
  the lower** (highs fall faster — convergence), and the band to actually **narrow** from start
  to apex by at least ``min_narrow``.
* A **confirmed breakout** = the first close after the last swing high that closes **above** the
  extrapolated upper trendline by ``breakout_buf``. That is the entry signal.

Timing: the breakout is known at its close; we **enter the next close** (one documented lag) and
hold ``H`` trading days. The honest question: do forward returns after a *confirmed* falling-wedge
breakout beat the name's own base rate? Arbiters: a one-sample / HAC *t* of the post-breakout
excess over the base rate, a **same-tape label-shuffle placebo** (random "breakout" dates on the
same tape), costs, and the synthetic positive control. A natural myth-check rides along: the
*downward* break of the same figure (does it really break up far more than down?).
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
# Falling-wedge detector
# --------------------------------------------------------------------------- #
def detect_wedges(close: np.ndarray, w: int = 5, min_highs: int = 3, min_lows: int = 3,
                  wedge_min: int = 25, wedge_max: int = 160, min_narrow: float = 0.25,
                  breakout_buf: float = 0.0, side: str = "up",
                  search_buf: int = 30, flat_tol: float = 0.02) -> list[dict]:
    """Detect confirmed falling-wedge breakouts on a close series.

    For each starting swing high, gather the run of subsequent swing highs out to ``wedge_max``
    bars; fit a least-squares line through them (the **upper** trendline) and a line through the
    swing **lows** that fall in the same span (the **lower** trendline). A falling wedge requires:

      * at least ``min_highs`` highs and ``min_lows`` lows in the span,
      * **both** trendline slopes **negative** (the figure slopes down),
      * the **upper** line **steeper** (more negative) than the lower (highs fall faster),
      * the vertical band (upper − lower) **narrows** from the start to the apex by at least
        ``min_narrow`` (the convergence that makes it a wedge, not a channel),
      * the lines still **converge in the future** (the band stays positive over the span).

    Then scan forward from the last swing high for the first close that clears the *extrapolated
    upper* line by ``breakout_buf`` (``side="up"``) — or, for the myth-check, the first close that
    breaks *below* the extrapolated lower line by ``breakout_buf`` (``side="down"``).

    Returns a list of dicts, one per confirmed breakout, with integer positions: ``first_high``,
    ``last_high``, ``breakout_idx`` (the signal bar), ``up_slope``/``lo_slope`` (per-bar trendline
    slopes), ``narrow`` (band-narrowing fraction), ``n_highs``/``n_lows``. Non-overlapping: once a
    breakout is taken the scan resumes after it. Everything uses data up to each bar; the breakout
    is the first qualifying close.
    """
    if side not in ("up", "down"):
        raise ValueError(f"side must be 'up' or 'down', got {side!r}")
    n = len(close)
    highs, lows = swing_pivots(close, w)
    wedges: list[dict] = []
    used_until = -1

    for hi_start in range(len(highs)):
        first = highs[hi_start]
        if first <= used_until:
            continue
        # gather subsequent swing highs within wedge_max bars, requiring each new high to be
        # (roughly) LOWER than the running max — a falling wedge has *descending* highs, so the
        # run stops as soon as a higher high appears (e.g. a post-breakout run-up), which keeps
        # the trendline fit clean and prevents the gather from spilling past the figure.
        run_highs = [first]
        run_max = close[first]
        for hcand in highs[hi_start + 1:]:
            if hcand - first > wedge_max:
                break
            if close[hcand] > run_max * (1.0 + flat_tol):
                break
            run_highs.append(int(hcand))
            run_max = max(run_max, close[hcand])
        if len(run_highs) < min_highs:
            continue
        first_high, last_high = run_highs[0], run_highs[-1]
        span = last_high - first_high
        if span < wedge_min or span > wedge_max:
            continue
        # swing lows that fall inside the span
        run_lows = lows[(lows >= first_high) & (lows <= last_high)]
        if len(run_lows) < min_lows:
            continue

        hx = np.asarray(run_highs, dtype=float)
        hy = close[run_highs]
        lx = run_lows.astype(float)
        ly = close[run_lows]
        up_slope, up_int = np.polyfit(hx, hy, 1)
        lo_slope, lo_int = np.polyfit(lx, ly, 1)

        # both lines slope DOWN, upper steeper than lower (highs fall faster -> convergence)
        if up_slope >= 0 or lo_slope >= 0:
            continue
        if up_slope >= lo_slope:        # up_slope must be MORE negative
            continue

        # band must narrow from the start to the apex by at least min_narrow
        def _upper(k):  # noqa: E306
            return up_slope * k + up_int

        def _lower(k):  # noqa: E306
            return lo_slope * k + lo_int

        band_start = _upper(first_high) - _lower(first_high)
        band_apex = _upper(last_high) - _lower(last_high)
        if band_start <= 0 or band_apex <= 0:
            continue
        narrow = (band_start - band_apex) / band_start
        if narrow < min_narrow:
            continue

        # breakout scan from just after the last swing high
        bo = None
        scan_end = min(last_high + search_buf, n)
        for k in range(last_high + 1, scan_end):
            up_k = _upper(k)
            lo_k = _lower(k)
            if side == "up" and close[k] > up_k * (1.0 + breakout_buf):
                bo = k
                break
            if side == "down" and close[k] < lo_k * (1.0 - breakout_buf):
                bo = k
                break
        if bo is None:
            continue
        wedges.append({
            "first_high": int(first_high), "last_high": int(last_high),
            "breakout_idx": int(bo), "up_slope": float(up_slope),
            "lo_slope": float(lo_slope), "narrow": float(narrow),
            "n_highs": int(len(run_highs)), "n_lows": int(len(run_lows)),
            "side": side,
        })
        used_until = bo
    return wedges


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
    """Run :func:`detect_wedges` on every name in the panel; return ticker -> breakout list."""
    closes = panel["close"]
    if names is None:
        names = list(closes.columns)
    out = {}
    for tk in names:
        c = closes[tk].dropna().to_numpy(dtype=float)
        out[tk] = detect_wedges(c, **detect_kw)
    return out


def run_experiment(panel: dict, horizon: int = 20, names: list[str] | None = None,
                   cost_bps: float = 5.0, n_draws: int = 5000, lag: int = 1,
                   excess: bool = True, seed: int = 414, side: str = "up",
                   **detect_kw) -> dict:
    """Pool every confirmed breakout across the panel and arbitrate the forward edge.

    For each name: detect falling wedges, take the forward-``horizon`` return after each breakout
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
        wedges = detect_wedges(c, side=side, **detect_kw)
        n_break += len(wedges)
        sig = [d["breakout_idx"] for d in wedges]
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
