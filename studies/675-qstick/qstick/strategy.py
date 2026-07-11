"""Qstick as a falsifiable mechanical rule — Study 675.

Tushar Chande's **Qstick** (*The New Technical Trader*, Chande & Kroll, 1994) reads each bar's
body — close minus open — as a vote for buyers (green body) or sellers (red body), then smooths
the daily votes with a moving average:

    Qstick_t = SMA_N(close - open)_t

A run of green-bodied closes pushes Qstick above zero ("buyers in control"); a run of red bodies
pushes it below. The folklore, echoed across charting sites, is the same shape as Balance of
Power's (study 473) and the Force Index's (study 423): **the zero-cross times a trend** — cross
up, buy; cross down, sell/short.

We encode the tightest mechanical version a proponent would accept and test it three ways:

1. **Smoothed Qstick** — a trailing ``smooth``-day average of the per-bar body, normalised by
   the prior close (``(close-open)/close_prev``) so the indicator is comparable in scale across
   instruments at different price levels (a documented normalisation choice; it does not change
   which bars cross zero, since the prior close is always positive). Causal — uses only bars up
   to and including *t* — so no look-ahead.
2. **Zero-cross entry** — a long fires when smoothed Qstick crosses **up** through zero
   (negative on *t-1*, non-negative on *t*): buying pressure has just taken over. Entry is at
   the **next** close (one documented lag); we then measure the forward H-day return.
3. **Controls.** (a) a **random-entry** baseline (same instrument, same epoch, same hold) that
   captures the tape's drift — the honest test on an upward-drifting tape; (b) a **sign-scramble
   placebo** that permutes the per-bar bodies before smoothing, destroying the temporal
   structure the cross depends on while keeping the marginal; and (c) a **trend-proxy check** —
   is Qstick just a relabelled, laggier version of plain trailing price momentum? By
   construction ``SMA_N(close) - SMA_N(open) ~= SMA_N(close) - SMA_N(close, shifted 1)``
   whenever the open sits close to the prior close (as it typically does for liquid ETFs), which
   telescopes to roughly ``(close_t - close_{t-N}) / N`` — the average daily price change over
   the last N days. We measure the empirical correlation between Qstick and that quantity
   directly, plus the overlap between Qstick's zero-crosses and a plain trailing-momentum
   zero-cross's, rather than asserting the identity.

No look-ahead: smoothed Qstick is a trailing average, the up-cross is read on the close of *t*,
the position is entered at the close of *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)
SMOOTH = 8      # Chande & Kroll's original default Qstick window


# --------------------------------------------------------------------------- #
# The indicator
# --------------------------------------------------------------------------- #
def raw_body(bars: pd.DataFrame) -> pd.Series:
    """Per-bar normalised body: (close - open) / prior close.

    Normalising by the prior close (rather than reporting raw price-unit close-open, as Chande's
    original does) makes the indicator comparable across instruments at different price levels
    without changing the sign of any bar or which bars trigger a zero-cross. The first bar (no
    prior close) is set to 0 — no information, matches the "no data yet" convention.
    """
    prev_close = bars["close"].shift(1)
    body = (bars["close"] - bars["open"]) / prev_close
    return body.fillna(0.0)


def qstick(bars: pd.DataFrame, smooth: int = SMOOTH) -> pd.Series:
    """Smoothed Qstick: trailing ``smooth``-day simple MA of the per-bar normalised body.

    Causal (uses only bars up to *t*). NaN until ``smooth`` bars exist.
    """
    return raw_body(bars).rolling(smooth, min_periods=smooth).mean()


def trend_momentum(bars: pd.DataFrame, n: int = SMOOTH) -> pd.Series:
    """The 'slow trend proxy' Qstick is accused of being: the trailing N-day average daily
    price change, ``(close_t - close_{t-N}) / N / close_{t-N}`` — the same units as
    :func:`qstick` (an average per-day fractional move), computed from price levels alone, with
    no reference to the open at all.
    """
    c = bars["close"]
    return ((c - c.shift(n)) / n) / c.shift(n)


# --------------------------------------------------------------------------- #
# Entries
# --------------------------------------------------------------------------- #
def qstick_cross_entries(bars: pd.DataFrame, smooth: int = SMOOTH) -> pd.DatetimeIndex:
    """Bars where smoothed Qstick crosses **up** through zero — the 'buyers take over' rule.

    Negative on *t-1*, non-negative on *t*. Read on the close of *t*; entry is executed at the
    next close by :func:`forward_returns`.
    """
    qs = qstick(bars, smooth=smooth)
    prev = qs.shift(1)
    mask = (prev < 0.0) & (qs >= 0.0) & qs.notna() & prev.notna()
    return bars.index[mask.to_numpy()]


def momentum_cross_entries(bars: pd.DataFrame, n: int = SMOOTH) -> pd.DatetimeIndex:
    """Zero-cross of the plain price-momentum proxy (:func:`trend_momentum`) — the naive
    'buy when the N-day trend turns up' rule, with no reference to the open at all. Used only
    to test whether Qstick's crosses are just a relabelling of this simpler rule.
    """
    mom = trend_momentum(bars, n=n)
    prev = mom.shift(1)
    mask = (prev < 0.0) & (mom >= 0.0) & mom.notna() & prev.notna()
    return bars.index[mask.to_numpy()]


def random_entries(bars: pd.DataFrame, n: int, smooth: int = SMOOTH,
                   seed: int = 0) -> pd.DatetimeIndex:
    """``n`` random entry dates (after the smoothing warm-up), the drift-matched baseline."""
    rng = np.random.default_rng(seed)
    valid = bars.index[smooth:]
    if len(valid) == 0:
        return pd.DatetimeIndex([])
    chosen = rng.choice(valid, size=min(n, len(valid)), replace=False)
    return pd.DatetimeIndex(sorted(chosen))


# --------------------------------------------------------------------------- #
# Forward-return engine
# --------------------------------------------------------------------------- #
def forward_returns(close: pd.Series, entries, horizon: int, cost_bps: float = 0.0) -> np.ndarray:
    """Forward ``horizon``-day return for each entry, entered at the *next* close (one lag).

    ``cost_bps`` is a one-way cost (charged twice: in + out) subtracted from each trade's
    return. Trades whose window overruns the tape are dropped.
    """
    pos = {d: i for i, d in enumerate(close.index)}
    p = close.to_numpy(dtype=float)
    n = p.size
    out = []
    for d in entries:
        i = pos.get(d)
        if i is None or i + 1 + horizon >= n:
            continue
        e = i + 1                      # enter at next close
        r = p[e + horizon] / p[e] - 1.0
        out.append(r - 2.0 * cost_bps * 1e-4)
    return np.asarray(out, dtype=float)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def hac_t(x: np.ndarray) -> float:
    """Newey-West (HAC) one-sample t-stat of the mean against zero."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 6:
        return float("nan")
    mu = x.mean()
    e = x - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for kk in range(1, lags + 1):
        w = 1.0 - kk / (lags + 1.0)
        lrv += 2.0 * w * float(e[kk:] @ e[:-kk]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def summarize(returns: np.ndarray) -> dict:
    """Headline per-trade stats: count, win-rate, mean (bps), per-trade Sharpe, HAC t."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    return {
        "n": int(n),
        "win": float((r > 0).mean()) if n else float("nan"),
        "mean_bps": float(r.mean() * 1e4) if n else float("nan"),
        "sharpe": float(r.mean() / r.std(ddof=1)) if n > 1 and r.std() > 0 else float("nan"),
        "t": hac_t(r),
    }


def scramble_placebo(bars: pd.DataFrame, horizon: int, smooth: int = SMOOTH,
                     n_draws: int = 500, seed: int = 675) -> dict:
    """Placebo: permute the per-bar body values before smoothing, destroying the time order.

    Keeps the body **marginal** (same set of close-open values) but scrambles their sequence, so
    the smoothed series and its zero-crosses become temporally meaningless. Returns the share of
    placebo runs whose mean cross-entry forward return **beats** the real one — the honest "does
    the body series' ordering carry information?" p-value, plus the observed mean.
    """
    obs = float(np.mean(forward_returns(
        bars["close"], qstick_cross_entries(bars, smooth=smooth), horizon)))
    rb = raw_body(bars).to_numpy(dtype=float)
    idx = bars.index
    close = bars["close"]
    rng = np.random.default_rng(seed)
    beats = 0
    valid = 0
    for _ in range(n_draws):
        perm = rng.permutation(rb)
        sb = pd.Series(perm, index=idx).rolling(smooth, min_periods=smooth).mean()
        prev = sb.shift(1)
        mask = (prev < 0.0) & (sb >= 0.0) & sb.notna() & prev.notna()
        ent = idx[mask.to_numpy()]
        rr = forward_returns(close, ent, horizon)
        if rr.size == 0:
            continue
        valid += 1
        if rr.mean() >= obs:
            beats += 1
    p = (beats + 1) / (valid + 1) if valid else float("nan")
    return {"obs": obs, "p_value": float(p), "n_draws": valid}


def trend_proxy_stats(bars: pd.DataFrame, smooth: int = SMOOTH) -> dict:
    """Is Qstick just a slow trend proxy? Two checks:

    (a) Pearson correlation between smoothed Qstick and the plain trailing N-day momentum
    proxy (:func:`trend_momentum`) that uses only price levels, never the open.
    (b) The overlap (Jaccard index) between the two rules' zero-cross entry dates — if Qstick's
    crosses are mostly the same dates a momentum-only rule would fire, the open/close split adds
    nothing beyond a laggier read of the same trend.
    """
    qs = qstick(bars, smooth=smooth)
    mom = trend_momentum(bars, n=smooth)
    df = pd.concat([qs.rename("qs"), mom.rename("mom")], axis=1).dropna()
    corr = float(np.corrcoef(df["qs"], df["mom"])[0, 1]) if len(df) >= 10 else float("nan")

    ent_qs = set(qstick_cross_entries(bars, smooth=smooth))
    ent_mom = set(momentum_cross_entries(bars, n=smooth))
    union = ent_qs | ent_mom
    jaccard = len(ent_qs & ent_mom) / len(union) if union else float("nan")
    return {"corr": corr, "n_obs": len(df), "n_qs_entries": len(ent_qs),
            "n_mom_entries": len(ent_mom), "jaccard": jaccard}


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def run_experiment(bars: pd.DataFrame, smooth: int = SMOOTH, cost_bps: float = 1.0,
                   random_seed: int = 7) -> dict:
    """Run the full gauntlet on one tape: Qstick up-cross vs random-entry baseline, all
    horizons. Returns a dict keyed by horizon with the cross-entry summary (gross + net), the
    drift-matched random-entry baseline, and the cross-minus-random delta.
    """
    close = bars["close"]
    ent = qstick_cross_entries(bars, smooth=smooth)
    res = {"n_entries": int(len(ent)), "by_h": {}}
    for h in HORIZONS:
        g = summarize(forward_returns(close, ent, h, cost_bps=0.0))
        net = summarize(forward_returns(close, ent, h, cost_bps=cost_bps))
        rnd = summarize(forward_returns(
            close, random_entries(bars, max(len(ent), 50), smooth=smooth, seed=random_seed), h))
        res["by_h"][h] = {
            "gross": g, "net": net, "random": rnd,
            "delta_bps": (g["mean_bps"] - rnd["mean_bps"])
            if np.isfinite(g["mean_bps"]) and np.isfinite(rnd["mean_bps"]) else float("nan"),
        }
    return res
