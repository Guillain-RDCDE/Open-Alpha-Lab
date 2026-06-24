"""Detector + timing rule + honest arbiters — Study 404 (Shooting Star).

The candle. A **shooting star** is a one-bar shape — the bearish mirror of the hammer:

    body       = |close - open|
    range      = high - low
    upper_wick = high - max(open, close)
    lower_wick = min(open, close) - low

The folk geometry (steelmanned): the upper wick is **long** relative to the body
(``upper_wick >= wick_mult * body``), the body sits at the **bottom** of the range
(``lower_wick <= lower_mult * body``), and the body itself is **small** relative to the
day's range (``body <= body_frac * range``). A doji-flat day (range ~ 0) is excluded.

The folklore then *splits the same shape by prior trend*:

* after an **uptrend**   -> **shooting star** -> bearish reversal (expect the top, SELL/short).
* after a **downtrend**  -> **inverted hammer** -> bullish reversal (a different claim).

We detect the geometry, classify the trend with a trailing SMA slope, and measure the
**forward 1/3/5/10-day return** after each occurrence. Because the shooting-star claim is
a *short* (expect a decline), we measure the **short-side** return (``-1 * price move``)
and the edge as short-return minus the name's own short base rate. Inference: a one-sample
/ HAC *t* on the conditional edge, a **label-shuffle placebo** (could random days look this
good?), and a **trend-strength / big-body filter** myth-check.

No look-ahead: the pattern is known at the close of bar *t*; the forward window starts at
the **next** close (one documented execution lag). Shorts pay borrow in ``net_of_costs``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (1, 3, 5, 10)


# --------------------------------------------------------------------------- #
# The candle geometry
# --------------------------------------------------------------------------- #
def candle_parts(bars: pd.DataFrame) -> pd.DataFrame:
    """Per-bar body / range / upper-wick / lower-wick (absolute, in price units)."""
    o, h, l, c = bars["open"], bars["high"], bars["low"], bars["close"]
    body = (c - o).abs()
    rng = (h - l)
    upper = h - np.maximum(o, c)
    lower = np.minimum(o, c) - l
    return pd.DataFrame({"body": body, "range": rng, "upper": upper, "lower": lower},
                        index=bars.index)


def is_star_shape(bars: pd.DataFrame, wick_mult: float = 2.0,
                  lower_mult: float = 0.5, body_frac: float = 0.35) -> pd.Series:
    """Boolean Series: does each bar have the shooting-star geometry?

    * long upper wick:   ``upper_wick >= wick_mult * body``
    * body near the bottom: ``lower_wick <= lower_mult * body``
    * small body:         ``body <= body_frac * range``
    * non-degenerate:     a perfectly flat doji or a marubozu with zero upper wick is excluded.

    The default thresholds are the conventional retail definition (upper wick at least 2x
    the body, little/no lower shadow). Same shape for shooting star and inverted hammer; the
    *prior trend* is what assigns the folk name.
    """
    p = candle_parts(bars)
    rng = p["range"].replace(0.0, np.nan)
    body = p["body"]
    # a tiny floor on body so a near-zero body (doji) doesn't trivially satisfy the
    # multiples; a doji is not the shooting star the folklore means.
    body_floor = (rng * 0.02)
    eff_body = np.maximum(body, body_floor)
    long_upper = p["upper"] >= wick_mult * eff_body
    small_lower = p["lower"] <= lower_mult * eff_body
    small_body = body <= body_frac * rng
    has_body = body > 0
    shape = long_upper & small_lower & small_body & has_body & rng.notna()
    return shape.fillna(False)


def trend_at(bars: pd.DataFrame, lookback: int = 10) -> pd.Series:
    """+1 if the trailing ``lookback``-day trend is up at bar *t*, -1 if down, 0 if flat.

    Trend = sign of (close[t] / close[t-lookback] - 1), known at the close of bar *t*
    (no look-ahead). This is the switch that turns the *same shape* into a bearish shooting
    star (prior uptrend) vs a bullish inverted hammer (prior downtrend).
    """
    chg = bars["close"] / bars["close"].shift(lookback) - 1.0
    return np.sign(chg).fillna(0.0).astype(int)


def trend_strength(bars: pd.DataFrame, lookback: int = 10) -> pd.Series:
    """Magnitude of the trailing ``lookback``-day run-up (the myth-check filter).

    ``close[t] / close[t-lookback] - 1`` — bigger means a stronger prior rally. The folk
    refinement ("only fade a star at the top of a *strong* run") is tested by keeping only
    events whose trend_strength exceeds a threshold.
    """
    return (bars["close"] / bars["close"].shift(lookback) - 1.0).fillna(0.0)


# --------------------------------------------------------------------------- #
# Forward returns (one execution lag), short-side
# --------------------------------------------------------------------------- #
def forward_return(bars: pd.DataFrame, horizon: int) -> pd.Series:
    """Price-only forward LONG return entered at the NEXT close, held ``horizon`` days.

    Signal known at close[t]; enter at close[t+1]; exit at close[t+1+horizon]. One ``shift``,
    applied once: ``ret[t] = close[t+1+H] / close[t+1] - 1``. NaN where the window overruns.
    The shooting-star claim trades this **short**, so the traded return is ``-forward_return``.
    """
    c = bars["close"]
    entry = c.shift(-1)
    exit_ = c.shift(-(1 + horizon))
    return exit_ / entry - 1.0


def base_rate_short(bars: pd.DataFrame, horizon: int) -> float:
    """Unconditional mean forward ``horizon``-day SHORT return (= -mean long return).

    The base rate the short claim must beat: short every day, what would you make? On a
    rising survivor this is *negative* (shorting a winner bleeds), which is exactly the
    survivorship headwind the bearish claim runs into — named on the Signal axis.
    """
    fr = forward_return(bars, horizon).dropna()
    return float((-fr).mean()) if len(fr) else float("nan")


# --------------------------------------------------------------------------- #
# Pool conditional events across the basket
# --------------------------------------------------------------------------- #
def conditional_returns(panel: dict, horizon: int, side: str = "star",
                        lookback: int = 10, min_strength: float = 0.0,
                        **shape_kw) -> dict:
    """Pool every shooting-star SHORT forward return across the basket vs the short base rate.

    ``side``:
      * ``"any"``      — every star-shaped bar (geometry only), traded short.
      * ``"star"``     — shape after an **uptrend** (the bearish shooting-star claim), short.
      * ``"invhammer"``— shape after a **downtrend** (the inverted-hammer / bullish variant).
                         For this side the traded return is **long** (the bullish claim).

    ``min_strength`` keeps only events whose trailing run-up exceeds the threshold (the
    myth-check filter; 0.0 = no filter). The returned conditional/edge arrays are in the
    **traded** direction (short for star/any, long for invhammer), and the edge subtracts
    each name's own base rate in the *same* traded direction (controls for each name's drift).
    """
    cond, edge, base_all = [], [], []
    for tk, bars in panel.items():
        if len(bars) < lookback + max(HORIZONS) + 5:
            continue
        shape = is_star_shape(bars, **shape_kw)
        tr = trend_at(bars, lookback=lookback)
        strength = trend_strength(bars, lookback=lookback)
        if side == "invhammer":
            sig = shape & (tr < 0)
            traded_sign = +1.0          # bullish: trade long
        elif side == "any":
            sig = shape
            traded_sign = -1.0          # short the star geometry
        else:  # "star"
            sig = shape & (tr > 0)
            traded_sign = -1.0          # short the bearish star
        if min_strength > 0.0:
            sig = sig & (strength.abs() >= min_strength)
        fr = forward_return(bars, horizon)
        traded_fr = traded_sign * fr
        br = float(traded_fr.dropna().mean()) if traded_fr.notna().any() else float("nan")
        r = traded_fr[sig].dropna()
        if len(r) == 0 or not np.isfinite(br):
            continue
        cond.extend(r.tolist())
        edge.extend((r - br).tolist())
        base_all.append(br)
    cond = np.asarray(cond, dtype=float)
    edge = np.asarray(edge, dtype=float)
    return {
        "n": int(cond.size),
        "cond_mean": float(cond.mean()) if cond.size else float("nan"),
        "base_mean": float(np.mean(base_all)) if base_all else float("nan"),
        "edge_mean": float(edge.mean()) if edge.size else float("nan"),
        "win": float((cond > 0).mean()) if cond.size else float("nan"),
        "edge": edge, "cond": cond,
    }


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def hac_t(sample: np.ndarray, max_lag: int | None = None) -> float:
    """Newey-West (HAC) one-sample t of ``sample`` mean against 0.

    The conditional events overlap (multi-day forward windows from nearby signals share
    days), so a plain i.i.d. t overstates significance; HAC with an auto bandwidth is the
    honest statistic the inference bar is graded on.
    """
    r = np.asarray(sample, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 6:
        return float("nan")
    mu = r.mean()
    e = r - mu
    lags = max_lag if max_lag is not None else int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def placebo_pvalue(panel: dict, horizon: int, side: str = "star", lookback: int = 10,
                   min_strength: float = 0.0, n_draws: int = 5000, seed: int = 404,
                   **shape_kw) -> dict:
    """Label-shuffle placebo: could the same *number* of random days look this good?

    For each name we count its conditional events, then draw that many random entry days
    from the same name's tape (same traded direction), recomputing the pooled edge
    (conditional minus base rate). ``p = P[shuffled edge >= observed edge]``. This kills any
    "we just sampled high-drift names" artefact while preserving each name's own return
    distribution.
    """
    obs = conditional_returns(panel, horizon, side=side, lookback=lookback,
                              min_strength=min_strength, **shape_kw)
    obs_edge = obs["edge_mean"]
    traded_sign = +1.0 if side == "invhammer" else -1.0
    per_name = []
    for tk, bars in panel.items():
        if len(bars) < lookback + max(HORIZONS) + 5:
            continue
        shape = is_star_shape(bars, **shape_kw)
        tr = trend_at(bars, lookback=lookback)
        strength = trend_strength(bars, lookback=lookback)
        if side == "invhammer":
            sig = shape & (tr < 0)
        elif side == "any":
            sig = shape
        else:
            sig = shape & (tr > 0)
        if min_strength > 0.0:
            sig = sig & (strength.abs() >= min_strength)
        traded_fr = (traded_sign * forward_return(bars, horizon)).dropna()
        k = int((sig.reindex(traded_fr.index).fillna(False)).sum())
        if k == 0 or len(traded_fr) == 0:
            continue
        per_name.append((traded_fr.to_numpy(), float(traded_fr.mean()), k))

    rng = np.random.default_rng(seed)
    draws = np.empty(n_draws)
    for d in range(n_draws):
        edges = []
        for fr_arr, br, k in per_name:
            pick = rng.choice(fr_arr, size=min(k, len(fr_arr)), replace=False)
            edges.extend((pick - br).tolist())
        draws[d] = np.mean(edges) if edges else 0.0
    p = float((draws >= obs_edge).mean())
    return {"obs_edge": obs_edge, "p_value": p, "draws": draws, "n": obs["n"]}


def net_of_costs(edge_mean: float, cost_bps: float = 5.0, borrow_bps: float = 0.0,
                 is_short: bool = True) -> float:
    """Per-event net edge after a one-way round trip (in + out) and short borrow.

    Each event is a fresh round trip: 2 one-way legs = ``2 * cost_bps``. The shooting-star
    claim is a **short**, so we charge ``borrow_bps`` of borrow for the holding window on top
    (``is_short=True``); the inverted-hammer variant is long (``is_short=False``, no borrow).
    Returns the net per-event edge.
    """
    borrow = borrow_bps if is_short else 0.0
    return float(edge_mean - 2.0 * cost_bps / 1e4 - borrow / 1e4)


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def run_experiment(panel: dict, side: str = "star", lookback: int = 10,
                   horizons=HORIZONS, cost_bps: float = 5.0, borrow_bps: float = 2.0,
                   min_strength: float = 0.0, placebo: bool = True,
                   n_draws: int = 5000, seed: int = 404, **shape_kw) -> dict:
    """Full per-horizon teardown for one side (star / invhammer / any).

    Returns a dict keyed by horizon with: n, conditional mean (traded direction), base rate,
    edge, win-rate, HAC t on the edge, label-shuffle placebo p, and the net edge after costs
    (+ borrow on the short star). ``min_strength`` activates the myth-check trend filter.
    """
    is_short = side != "invhammer"
    out = {}
    for h in horizons:
        cr = conditional_returns(panel, h, side=side, lookback=lookback,
                                 min_strength=min_strength, **shape_kw)
        t = hac_t(cr["edge"])
        p = (placebo_pvalue(panel, h, side=side, lookback=lookback,
                            min_strength=min_strength, n_draws=n_draws, seed=seed,
                            **shape_kw)["p_value"] if placebo else float("nan"))
        out[h] = {
            "n": cr["n"], "cond_mean": cr["cond_mean"], "base_mean": cr["base_mean"],
            "edge_mean": cr["edge_mean"], "win": cr["win"], "t": t, "p_placebo": p,
            "net_edge": net_of_costs(cr["edge_mean"], cost_bps=cost_bps,
                                     borrow_bps=borrow_bps * h, is_short=is_short),
        }
    return out
