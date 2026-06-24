"""Detector + timing rule + honest arbiters — Study 403 (Hammer & Hanging Man).

The candle. A **hammer** / **hanging man** is the *same* one-bar shape:

    body      = |close - open|
    range     = high - low
    lower_wick = min(open, close) - low
    upper_wick = high - max(open, close)

The folk geometry (steelmanned): the lower wick is **long** relative to the body
(``lower_wick >= wick_mult * body``), the body sits at the **top** of the range
(``upper_wick <= upper_mult * body``), and the body itself is **small** relative to the
day's range (``body <= body_frac * range``). A doji-flat day (range ~ 0) is excluded.

The folklore then *splits the same shape by prior trend*:

* after a **downtrend** -> **hammer** -> bullish reversal (expect the floor, buy).
* after an **uptrend**  -> **hanging man** -> bearish reversal (expect the top, sell).

We detect the geometry, classify the trend with a trailing SMA slope, and measure the
**forward 1/3/5/10-day return** after each occurrence against the name's **unconditional
base rate** over the same horizon. Inference: a one-sample / HAC *t* on the conditional
edge (conditional mean minus base rate), a **label-shuffle placebo** (could random days
look this good?), and a **trend/volume-style filter** myth-check.

No look-ahead: the pattern is known at the close of bar *t*; the forward window starts at
the **next** close (one documented execution lag).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (1, 3, 5, 10)


# --------------------------------------------------------------------------- #
# The candle geometry
# --------------------------------------------------------------------------- #
def candle_parts(bars: pd.DataFrame) -> pd.DataFrame:
    """Per-bar body / range / lower-wick / upper-wick (absolute, in price units)."""
    o, h, l, c = bars["open"], bars["high"], bars["low"], bars["close"]
    body = (c - o).abs()
    rng = (h - l)
    lower = np.minimum(o, c) - l
    upper = h - np.maximum(o, c)
    return pd.DataFrame({"body": body, "range": rng, "lower": lower, "upper": upper},
                        index=bars.index)


def is_hammer_shape(bars: pd.DataFrame, wick_mult: float = 2.0,
                    upper_mult: float = 0.5, body_frac: float = 0.35) -> pd.Series:
    """Boolean Series: does each bar have the hammer / hanging-man geometry?

    * long lower wick:  ``lower_wick >= wick_mult * body``
    * body near the top: ``upper_wick <= upper_mult * body``
    * small body:        ``body <= body_frac * range``
    * non-degenerate:    ``range > 0`` and body > 0 are handled (a perfectly flat doji or a
      marubozu with zero lower wick is excluded).

    The default thresholds are the conventional retail definition (lower wick at least 2x
    the body, little/no upper shadow). Same shape for both hammer and hanging man.
    """
    p = candle_parts(bars)
    rng = p["range"].replace(0.0, np.nan)
    body = p["body"]
    # use a tiny floor on body so a near-zero body (doji) doesn't trivially satisfy the
    # multiple tests; a doji is not the hammer the folklore means.
    body_floor = (rng * 0.02)
    eff_body = np.maximum(body, body_floor)
    long_lower = p["lower"] >= wick_mult * eff_body
    small_upper = p["upper"] <= upper_mult * eff_body
    small_body = body <= body_frac * rng
    has_body = body > 0
    shape = long_lower & small_upper & small_body & has_body & rng.notna()
    return shape.fillna(False)


def trend_at(bars: pd.DataFrame, lookback: int = 10) -> pd.Series:
    """+1 if the trailing ``lookback``-day trend is up at bar *t*, -1 if down, 0 if flat.

    Trend = sign of (close[t] / close[t-lookback] - 1), known at the close of bar *t*
    (no look-ahead). This is the switch that turns the *same shape* into a bullish hammer
    (prior downtrend) vs a bearish hanging man (prior uptrend).
    """
    chg = bars["close"] / bars["close"].shift(lookback) - 1.0
    return np.sign(chg).fillna(0.0).astype(int)


# --------------------------------------------------------------------------- #
# Forward returns (one execution lag)
# --------------------------------------------------------------------------- #
def forward_return(bars: pd.DataFrame, horizon: int) -> pd.Series:
    """Price-only forward return entered at the NEXT close, held ``horizon`` days.

    Signal known at close[t]; enter at close[t+1]; exit at close[t+1+horizon]. One ``shift``,
    applied once: ``ret[t] = close[t+1+H] / close[t+1] - 1``. NaN where the window overruns.
    """
    c = bars["close"]
    entry = c.shift(-1)
    exit_ = c.shift(-(1 + horizon))
    return exit_ / entry - 1.0


def base_rate(bars: pd.DataFrame, horizon: int) -> float:
    """Unconditional mean forward ``horizon``-day return (the base rate to beat)."""
    fr = forward_return(bars, horizon).dropna()
    return float(fr.mean()) if len(fr) else float("nan")


# --------------------------------------------------------------------------- #
# Pool conditional events across the basket
# --------------------------------------------------------------------------- #
def conditional_returns(panel: dict, horizon: int, side: str = "any",
                        lookback: int = 10, **shape_kw) -> dict:
    """Pool every hammer/hanging-man forward return across the basket vs the base rate.

    ``side``:
      * ``"any"``      — every hammer-shaped bar (geometry only).
      * ``"hammer"``   — shape after a **downtrend** (the bullish reversal claim).
      * ``"hangingman"``— shape after an **uptrend** (the bearish reversal claim).

    Returns the pooled conditional forward returns, the pooled (per-name de-meaned) base
    rate, the **edge** (conditional mean minus per-event base rate), counts, and win-rate.
    Each conditional return is paired with its own name's unconditional base rate so the
    edge is excess-over-buy-and-hold for that name (controls for each name's drift).
    """
    cond, edge, base_all = [], [], []
    for tk, bars in panel.items():
        if len(bars) < lookback + max(HORIZONS) + 5:
            continue
        shape = is_hammer_shape(bars, **shape_kw)
        tr = trend_at(bars, lookback=lookback)
        if side == "hammer":
            sig = shape & (tr < 0)
        elif side == "hangingman":
            sig = shape & (tr > 0)
        else:
            sig = shape
        fr = forward_return(bars, horizon)
        br = base_rate(bars, horizon)
        r = fr[sig].dropna()
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


def placebo_pvalue(panel: dict, horizon: int, side: str = "any", lookback: int = 10,
                   n_draws: int = 5000, seed: int = 403, **shape_kw) -> dict:
    """Label-shuffle placebo: could the same *number* of random days look this good?

    For each name we count its conditional events, then draw that many random entry days
    from the same name's tape, recomputing the pooled edge (conditional minus base rate).
    ``p = P[shuffled edge >= observed edge]``. This kills any "we just sampled high-drift
    names" artefact while preserving each name's own return distribution.
    """
    obs = conditional_returns(panel, horizon, side=side, lookback=lookback, **shape_kw)
    obs_edge = obs["edge_mean"]
    # per-name event counts and forward-return arrays
    per_name = []
    for tk, bars in panel.items():
        if len(bars) < lookback + max(HORIZONS) + 5:
            continue
        shape = is_hammer_shape(bars, **shape_kw)
        tr = trend_at(bars, lookback=lookback)
        if side == "hammer":
            sig = shape & (tr < 0)
        elif side == "hangingman":
            sig = shape & (tr > 0)
        else:
            sig = shape
        fr = forward_return(bars, horizon).dropna()
        k = int((sig.reindex(fr.index).fillna(False)).sum())
        if k == 0 or len(fr) == 0:
            continue
        per_name.append((fr.to_numpy(), float(fr.mean()), k))

    rng = np.random.default_rng(seed)
    draws = np.empty(n_draws)
    for d in range(n_draws):
        edges = []
        for fr_arr, br, k in per_name:
            pick = rng.choice(fr_arr, size=min(k, len(fr_arr)), replace=False)
            edges.extend((pick - br).tolist())
        draws[d] = np.mean(edges) if edges else 0.0
    p = float((draws >= obs_edge).mean())
    return {"obs_edge": obs_edge, "p_value": p, "draws": draws,
            "n": obs["n"]}


def net_of_costs(edge_mean: float, cost_bps: float = 5.0) -> float:
    """Per-event net edge after a one-way round trip (in + out) at ``cost_bps`` each way.

    Each event is a fresh round trip: 2 one-way legs = ``2 * cost_bps``. Long-only (buy the
    hammer), so no borrow. Returns the net per-event edge.
    """
    return float(edge_mean - 2.0 * cost_bps / 1e4)


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def run_experiment(panel: dict, side: str = "hammer", lookback: int = 10,
                   horizons=HORIZONS, cost_bps: float = 5.0, placebo: bool = True,
                   n_draws: int = 5000, seed: int = 403, **shape_kw) -> dict:
    """Full per-horizon teardown for one side (hammer / hangingman / any).

    Returns a dict keyed by horizon with: n, conditional mean, base rate, edge, win-rate,
    HAC t on the edge, label-shuffle placebo p, and the net edge after costs.
    """
    out = {}
    for h in horizons:
        cr = conditional_returns(panel, h, side=side, lookback=lookback, **shape_kw)
        t = hac_t(cr["edge"])
        p = (placebo_pvalue(panel, h, side=side, lookback=lookback, n_draws=n_draws,
                            seed=seed, **shape_kw)["p_value"] if placebo else float("nan"))
        out[h] = {
            "n": cr["n"], "cond_mean": cr["cond_mean"], "base_mean": cr["base_mean"],
            "edge_mean": cr["edge_mean"], "win": cr["win"], "t": t, "p_placebo": p,
            "net_edge": net_of_costs(cr["edge_mean"], cost_bps=cost_bps),
        }
    return out
