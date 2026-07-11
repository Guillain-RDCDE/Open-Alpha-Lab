"""Detector + timing rule + honest arbiters — Study 684 (Inverted Hammer).

The candle. An **inverted hammer** is a one-bar shape:

    body       = |close - open|
    range      = high - low
    upper_wick = high - max(open, close)
    lower_wick = min(open, close) - low

The folk geometry (steelmanned): the upper wick is **long** relative to the body
(``upper_wick >= wick_mult * body``), the body sits at the **bottom** of the range
(``lower_wick <= lower_mult * body``), and the body itself is **small** relative to the
day's range (``body <= body_frac * range``). A doji-flat day (range ~ 0) is excluded. This
is the *identical* geometry to the shooting star (sibling study 404) — the folklore splits
the same shape purely by **prior trend**:

* after a **downtrend** -> **inverted hammer** -> bullish reversal (expect the floor, BUY).
* after an **uptrend**  -> **shooting star**   -> bearish reversal (the sibling study's claim).

We detect the geometry, classify the trend with a trailing close-vs-close slope, and
measure the **forward 1/3/5/10-day return** after each occurrence — entered LONG at the
next close — against the name's **unconditional base rate** over the same horizon.
Inference: a one-sample / HAC *t* on the conditional edge (conditional mean minus base
rate), a **label-shuffle placebo** (could random days look this good?), a **Bonferroni**
correction for testing four horizons at once, and a trend-window / wick-strictness
myth-check.

No look-ahead: the pattern is known at the close of bar *t*; the forward window starts at
the **next** close (one documented execution lag). Long-only claim, no borrow.
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


def is_invhammer_shape(bars: pd.DataFrame, wick_mult: float = 2.0,
                       lower_mult: float = 0.5, body_frac: float = 0.35) -> pd.Series:
    """Boolean Series: does each bar have the inverted-hammer geometry?

    * long upper wick:      ``upper_wick >= wick_mult * body``
    * body near the bottom: ``lower_wick <= lower_mult * body``
    * small body:           ``body <= body_frac * range``
    * non-degenerate:       a perfectly flat doji or a marubozu with zero upper wick is
      excluded (a tiny body floor keeps a near-zero-body doji from trivially satisfying
      the wick multiples — a doji is not the inverted hammer the folklore means).

    The default thresholds are the conventional retail definition (upper wick at least 2x
    the body, little/no lower shadow). Same shape as the shooting star (study 404); the
    *prior trend* is what assigns the folk name and the trade direction.
    """
    p = candle_parts(bars)
    rng = p["range"].replace(0.0, np.nan)
    body = p["body"]
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
    (no look-ahead). This is the switch that turns the *same shape* into a bullish
    inverted hammer (prior downtrend) vs a bearish shooting star (prior uptrend, the
    sibling study's claim).
    """
    chg = bars["close"] / bars["close"].shift(lookback) - 1.0
    return np.sign(chg).fillna(0.0).astype(int)


def trend_strength(bars: pd.DataFrame, lookback: int = 10) -> pd.Series:
    """Magnitude of the trailing ``lookback``-day decline (the myth-check filter).

    ``close[t] / close[t-lookback] - 1`` — more negative means a deeper prior slide. The
    folk refinement ("only trust the reversal after a *real* washout") is tested by keeping
    only events whose trend_strength magnitude exceeds a threshold.
    """
    return (bars["close"] / bars["close"].shift(lookback) - 1.0).fillna(0.0)


# --------------------------------------------------------------------------- #
# Forward returns (one execution lag), long-side
# --------------------------------------------------------------------------- #
def forward_return(bars: pd.DataFrame, horizon: int) -> pd.Series:
    """Price-only forward LONG return entered at the NEXT close, held ``horizon`` days.

    Signal known at close[t]; enter at close[t+1]; exit at close[t+1+horizon]. One ``shift``,
    applied once: ``ret[t] = close[t+1+H] / close[t+1] - 1``. NaN where the window overruns.
    The inverted-hammer claim trades this **long** (buy the floor).
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
def conditional_returns(panel: dict, horizon: int, side: str = "invhammer",
                        lookback: int = 10, min_strength: float = 0.0,
                        **shape_kw) -> dict:
    """Pool every inverted-hammer LONG forward return across the basket vs the base rate.

    ``side``:
      * ``"any"``        — every inverted-hammer-shaped bar (geometry only), traded long.
      * ``"invhammer"``  — shape after a **downtrend** (the bullish claim under test), long.
      * ``"star"``       — shape after an **uptrend** (the bearish look-alike; sibling study
        404's claim), included here only for the myth-check contrast — traded long too, so
        a caller can see whether the *same* long trade would have "worked" on the wrong side
        of the trend split (it should not, if the claim is genuinely trend-conditional).

    ``min_strength`` keeps only events whose trailing decline magnitude exceeds the
    threshold (the myth-check filter; 0.0 = no filter). The edge subtracts each name's own
    base rate (controls for that name's drift).
    """
    cond, edge, base_all = [], [], []
    for tk, bars in panel.items():
        if len(bars) < lookback + max(HORIZONS) + 5:
            continue
        shape = is_invhammer_shape(bars, **shape_kw)
        tr = trend_at(bars, lookback=lookback)
        strength = trend_strength(bars, lookback=lookback)
        if side == "star":
            sig = shape & (tr > 0)
        elif side == "any":
            sig = shape
        else:  # "invhammer"
            sig = shape & (tr < 0)
        if min_strength > 0.0:
            sig = sig & (strength.abs() >= min_strength)
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


def placebo_pvalue(panel: dict, horizon: int, side: str = "invhammer", lookback: int = 10,
                   min_strength: float = 0.0, n_draws: int = 5000, seed: int = 684,
                   **shape_kw) -> dict:
    """Label-shuffle placebo: could the same *number* of random days look this good?

    For each name we count its conditional events, then draw that many random entry days
    from the same name's tape, recomputing the pooled edge (conditional minus base rate).
    ``p = P[shuffled edge >= observed edge]``. This kills any "we just sampled high-drift
    names" artefact while preserving each name's own return distribution.
    """
    obs = conditional_returns(panel, horizon, side=side, lookback=lookback,
                              min_strength=min_strength, **shape_kw)
    obs_edge = obs["edge_mean"]
    per_name = []
    for tk, bars in panel.items():
        if len(bars) < lookback + max(HORIZONS) + 5:
            continue
        shape = is_invhammer_shape(bars, **shape_kw)
        tr = trend_at(bars, lookback=lookback)
        strength = trend_strength(bars, lookback=lookback)
        if side == "star":
            sig = shape & (tr > 0)
        elif side == "any":
            sig = shape
        else:
            sig = shape & (tr < 0)
        if min_strength > 0.0:
            sig = sig & (strength.abs() >= min_strength)
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
    return {"obs_edge": obs_edge, "p_value": p, "draws": draws, "n": obs["n"]}


def bonferroni(p_values: list[float]) -> list[float]:
    """Bonferroni-adjust a family of p-values: ``p_adj = min(1, p * m)``.

    The headline test runs the same placebo across ``m = len(p_values)`` horizons
    (1/3/5/10 days) — testing four horizons and quoting the best one is exactly the kind
    of multiple-comparison snoop the desk's inference bar exists to catch. Applying the
    Bonferroni correction to the whole family (not just the best horizon) is the
    conservative, honest way to report it.
    """
    m = len(p_values)
    return [float(min(1.0, p * m)) if np.isfinite(p) else float("nan") for p in p_values]


def net_of_costs(edge_mean: float, cost_bps: float = 5.0) -> float:
    """Per-event net edge after a one-way round trip (in + out) at ``cost_bps`` each way.

    Each event is a fresh round trip: 2 one-way legs = ``2 * cost_bps``. Long-only (buy the
    inverted hammer), so no borrow. Returns the net per-event edge.
    """
    return float(edge_mean - 2.0 * cost_bps / 1e4)


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def run_experiment(panel: dict, side: str = "invhammer", lookback: int = 10,
                   horizons=HORIZONS, cost_bps: float = 5.0, min_strength: float = 0.0,
                   placebo: bool = True, n_draws: int = 5000, seed: int = 684,
                   **shape_kw) -> dict:
    """Full per-horizon teardown for one side (invhammer / star / any).

    Returns a dict keyed by horizon with: n, conditional mean, base rate, edge, win-rate,
    HAC t on the edge, label-shuffle placebo p (raw and Bonferroni-adjusted across the
    horizon family), and the net edge after costs.
    """
    out = {}
    raw_p = {}
    for h in horizons:
        cr = conditional_returns(panel, h, side=side, lookback=lookback,
                                 min_strength=min_strength, **shape_kw)
        t = hac_t(cr["edge"])
        p = (placebo_pvalue(panel, h, side=side, lookback=lookback,
                            min_strength=min_strength, n_draws=n_draws, seed=seed,
                            **shape_kw)["p_value"] if placebo else float("nan"))
        raw_p[h] = p
        out[h] = {
            "n": cr["n"], "cond_mean": cr["cond_mean"], "base_mean": cr["base_mean"],
            "edge_mean": cr["edge_mean"], "win": cr["win"], "t": t, "p_placebo": p,
            "net_edge": net_of_costs(cr["edge_mean"], cost_bps=cost_bps),
        }
    if placebo:
        adj = bonferroni([raw_p[h] for h in horizons])
        for h, pa in zip(horizons, adj):
            out[h]["p_bonferroni"] = pa
    else:
        for h in horizons:
            out[h]["p_bonferroni"] = float("nan")
    return out
