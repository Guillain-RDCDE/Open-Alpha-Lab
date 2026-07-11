"""Detector + timing rule + honest arbiters — Study 691 (Homing Pigeon).

The candle pair. Let ``(O0, C0)`` be the prior bar's open/close and ``(O1, C1)`` the
current bar's. A **homing pigeon** requires:

    body0 = O0 - C0   (> 0: the prior bar is a DOWN day; its size is the "big" body)
    body1 = O1 - C1   (> 0: the current bar is ALSO a down day, same colour)

    0 < body1 < body0                                (current body strictly smaller)
    min(O1,C1) >= min(O0,C0)  and  max(O1,C1) <= max(O0,C0)   (current body fully inside)

This is the *same-colour* cousin of the [harami](../../406-harami-pattern/) (which
requires *opposite* colours): two down days, the second shrinking inside the first,
appearing after a **downtrend**. The folklore reads the shrinking down-day as sellers
running out of conviction — a bullish reversal, buy. It is a rarer, more specific version
of the "long wick after a downtrend" floor story tested (and busted) for the
[inverted hammer](../../684-inverted-hammer/).

We detect the geometry, classify the trend with a trailing close-vs-close slope, and
measure the **forward 1/3/5/10-day return** after each occurrence — entered LONG at the
next close — against the name's **unconditional base rate** over the same horizon.
Inference: a HAC one-sample *t* on the conditional edge (conditional mean minus base
rate), a **label-shuffle placebo** (could random days look this good?), a **Bonferroni**
correction for testing four horizons at once, and a trend-window / washout-depth
myth-check.

No look-ahead: the pattern is known at the close of bar *t*; the forward window starts at
the **next** close (one documented execution lag). Long-only claim, no borrow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (1, 3, 5, 10)


# --------------------------------------------------------------------------- #
# The two-bar geometry
# --------------------------------------------------------------------------- #
def is_homing_pigeon(o0: float, c0: float, o1: float, c1: float) -> bool:
    """True iff the prior->current bar pair forms a textbook homing pigeon.

    Both bars must be DOWN days (close < open), the current body strictly smaller than
    the prior body, and the current body fully contained inside the prior body.
    """
    body0 = o0 - c0
    body1 = o1 - c1
    if body0 <= 0 or body1 <= 0 or body1 >= body0:
        return False
    lo0, hi0 = c0, o0
    lo1, hi1 = c1, o1
    return (lo1 >= lo0) and (hi1 <= hi0)


def is_homingpigeon_shape(bars: pd.DataFrame) -> pd.Series:
    """Boolean Series: does the pair (bar t-1, bar t) form a homing pigeon at bar t?

    Vectorised equivalent of ``is_homing_pigeon`` applied to ``bars`` shifted by one.
    """
    o0 = bars["open"].shift(1)
    c0 = bars["close"].shift(1)
    o1 = bars["open"]
    c1 = bars["close"]
    body0 = o0 - c0
    body1 = o1 - c1
    smaller = (body0 > 0) & (body1 > 0) & (body1 < body0)
    inside = (c1 >= c0) & (o1 <= o0)
    shape = smaller & inside
    return shape.fillna(False)


def trend_at(bars: pd.DataFrame, lookback: int = 10) -> pd.Series:
    """+1 if the trailing ``lookback``-day trend is up at bar *t*, -1 if down, 0 if flat.

    Trend = sign of (close[t] / close[t-lookback] - 1), known at the close of bar *t*
    (no look-ahead). This is the switch that turns the shape into the bullish claim under
    test (prior downtrend) vs the "wrong side" myth-check contrast (prior uptrend).
    """
    chg = bars["close"] / bars["close"].shift(lookback) - 1.0
    return np.sign(chg).fillna(0.0).astype(int)


def trend_strength(bars: pd.DataFrame, lookback: int = 10) -> pd.Series:
    """Magnitude of the trailing ``lookback``-day decline (the myth-check filter).

    ``close[t] / close[t-lookback] - 1`` — more negative means a deeper prior slide. The
    folk refinement ("only trust the reversal after a *real* washout") is tested by
    keeping only events whose trend_strength magnitude exceeds a threshold.
    """
    return (bars["close"] / bars["close"].shift(lookback) - 1.0).fillna(0.0)


# --------------------------------------------------------------------------- #
# Forward returns (one execution lag), long-side
# --------------------------------------------------------------------------- #
def forward_return(bars: pd.DataFrame, horizon: int) -> pd.Series:
    """Price-only forward LONG return entered at the NEXT close, held ``horizon`` days.

    Signal known at close[t]; enter at close[t+1]; exit at close[t+1+horizon]. One
    ``shift``, applied once: ``ret[t] = close[t+1+H] / close[t+1] - 1``. NaN where the
    window overruns. The homing-pigeon claim trades this **long** (buy the floor).
    """
    c = bars["close"]
    entry = c.shift(-1)
    exit_ = c.shift(-(1 + horizon))
    return exit_ / entry - 1.0


def base_rate(bars: pd.DataFrame, horizon: int) -> float:
    """Unconditional mean forward ``horizon``-day return (the base rate to beat)."""
    fr = forward_return(bars, horizon).dropna()
    return float(fr.mean()) if len(fr) else float("nan")


def downtrend_pool(panel: dict, horizon: int, lookback: int = 10) -> np.ndarray:
    """Pool the forward LONG return of every bar sitting in a downtrend — SHAPE not
    required.

    This is the honest "would buying any dip in a downtrend already get you this?"
    comparison — the alpha-vs-beta cut. A pattern that only "works" because it fires
    exclusively inside downtrends, where mean reversion is generic, is not adding
    information beyond the trend filter itself; a real *shape* effect must beat this
    pool, not just the unconditional (any-day) base rate.
    """
    vals = []
    for bars in panel.values():
        if len(bars) < lookback + max(HORIZONS) + 5:
            continue
        tr = trend_at(bars, lookback=lookback)
        fr = forward_return(bars, horizon)
        sel = fr[tr < 0].dropna()
        if len(sel):
            vals.append(sel.to_numpy())
    return np.concatenate(vals) if vals else np.array([])


# --------------------------------------------------------------------------- #
# Pool conditional events across the basket
# --------------------------------------------------------------------------- #
def conditional_returns(panel: dict, horizon: int, side: str = "pigeon",
                        lookback: int = 10, min_strength: float = 0.0) -> dict:
    """Pool every homing-pigeon LONG forward return across the basket vs the base rate.

    ``side``:
      * ``"any"``     — every homing-pigeon-shaped pair (geometry only), traded long.
      * ``"pigeon"``  — shape after a **downtrend** (the bullish claim under test), long.
      * ``"wrongside"`` — shape after an **uptrend** (the myth-check contrast) — traded
        long too, so a caller can see whether the *same* long trade would have "worked"
        on the wrong side of the trend split (it should not, if the claim is genuinely
        trend-conditional).

    ``min_strength`` keeps only events whose trailing decline magnitude exceeds the
    threshold (the myth-check filter; 0.0 = no filter). The edge subtracts each name's own
    base rate (controls for that name's drift).
    """
    cond, edge, base_all = [], [], []
    for tk, bars in panel.items():
        if len(bars) < lookback + max(HORIZONS) + 5:
            continue
        shape = is_homingpigeon_shape(bars)
        tr = trend_at(bars, lookback=lookback)
        strength = trend_strength(bars, lookback=lookback)
        if side == "wrongside":
            sig = shape & (tr > 0)
        elif side == "any":
            sig = shape
        else:  # "pigeon"
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


def event_clustering(panel: dict, side: str = "pigeon", lookback: int = 10) -> dict:
    """How concentrated are the pigeon events in calendar time (cross-name clustering)?

    Pooling many names' events together only inflates a naive t-stat if the "many"
    events are really a handful of shared market-wide dates (e.g. everyone prints a
    homing pigeon the same week of a crash) — HAC corrects for a name's own
    autocorrelation, not for that kind of cross-sectional co-occurrence. This counts
    events per ISO calendar week across the whole panel and reports how much of the
    total sits in the busiest 10 weeks — a large share would flag a clustering
    artefact; a small share means the effect is broad-based across the sample.
    """
    dates = []
    for tk, bars in panel.items():
        if len(bars) < lookback + max(HORIZONS) + 5:
            continue
        shape = is_homingpigeon_shape(bars)
        tr = trend_at(bars, lookback=lookback)
        sig = shape & (tr > 0) if side == "wrongside" else (shape if side == "any" else shape & (tr < 0))
        dates.extend(bars.index[sig].tolist())
    if not dates:
        return {"n": 0, "n_weeks": 0, "top10_week_share": float("nan")}
    s = pd.Series(1, index=pd.DatetimeIndex(dates))
    by_week = s.groupby(s.index.to_period("W")).sum().sort_values(ascending=False)
    top10_share = float(by_week.head(10).sum() / len(dates))
    return {"n": len(dates), "n_weeks": int(by_week.size), "top10_week_share": top10_share}


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


def welch_t(sample: np.ndarray, base: np.ndarray) -> float:
    """Welch t of mean(sample) - mean(base) (unequal variance) — the alpha-vs-beta test."""
    a = np.asarray(sample, float); a = a[np.isfinite(a)]
    b = np.asarray(base, float); b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def placebo_pvalue(panel: dict, horizon: int, side: str = "pigeon", lookback: int = 10,
                   min_strength: float = 0.0, n_draws: int = 5000, seed: int = 691) -> dict:
    """Label-shuffle placebo: could the same *number* of random days look this good?

    For each name we count its conditional events, then draw that many random entry days
    from the same name's tape, recomputing the pooled edge (conditional minus base rate).
    ``p = P[shuffled edge >= observed edge]``. This kills any "we just sampled high-drift
    names" artefact while preserving each name's own return distribution.
    """
    obs = conditional_returns(panel, horizon, side=side, lookback=lookback,
                              min_strength=min_strength)
    obs_edge = obs["edge_mean"]
    per_name = []
    for tk, bars in panel.items():
        if len(bars) < lookback + max(HORIZONS) + 5:
            continue
        shape = is_homingpigeon_shape(bars)
        tr = trend_at(bars, lookback=lookback)
        strength = trend_strength(bars, lookback=lookback)
        if side == "wrongside":
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
    homing pigeon), so no borrow. Returns the net per-event edge.
    """
    return float(edge_mean - 2.0 * cost_bps / 1e4)


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def run_experiment(panel: dict, side: str = "pigeon", lookback: int = 10,
                   horizons=HORIZONS, cost_bps: float = 5.0, min_strength: float = 0.0,
                   placebo: bool = True, n_draws: int = 5000, seed: int = 691) -> dict:
    """Full per-horizon teardown for one side (pigeon / wrongside / any).

    Returns a dict keyed by horizon with: n, conditional mean, base rate, edge, win-rate,
    HAC t on the edge, label-shuffle placebo p (raw and Bonferroni-adjusted across the
    horizon family), and the net edge after costs.
    """
    out = {}
    raw_p = {}
    for h in horizons:
        cr = conditional_returns(panel, h, side=side, lookback=lookback,
                                 min_strength=min_strength)
        t = hac_t(cr["edge"])
        p = (placebo_pvalue(panel, h, side=side, lookback=lookback,
                            min_strength=min_strength, n_draws=n_draws, seed=seed)["p_value"]
             if placebo else float("nan"))
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
