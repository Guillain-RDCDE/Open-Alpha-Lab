"""Strategy + inference for Study 408 — the Three Black Crows candlestick pattern.

The claim (a famous bearish-reversal candle): **three black crows** — three consecutive
long *red* (down) candles, each opening inside the prior body and closing near its low,
progressively lower — signal that a top is in and a sell-off / crash is coming. The folk
recipe: spot three black crows, get short (or get out), because the next days fall hard.

Precise detector (the textbook three-black-crows definition), on bars (t-2, t-1, t):

    Let (O0,C0),(O1,C1),(O2,C2) be the three bars oldest->newest, with H/L for wicks.
    1. all three are RED real bodies:        C0 < O0,  C1 < O1,  C2 < O2
    2. each closes lower than the last:       C2 < C1 < C0
    3. each opens within the prior real body: O0 >= O1 >= O2  and
                                              C0 < O1 <= O0  and  C1 < O2 <= O1
    (Optional myth-check filters: require "long bodies + short lower wicks" — the strict
     textbook crow — and/or a prior uptrend so it's a genuine *reversal*.)

The pattern is confirmed at the **close** of bar t. Signed by direction, three black
crows is a BEARISH signal: we go SHORT, so the signed forward return = -1 * forward move.

Measurement (no look-ahead, one documented lag):

    Confirm at close of t; enter at the **next** bar's open (t+1) — one execution lag —
    and measure the forward 1 / 3 / 5 / 10-day return, signed short. The honest benchmark
    is the **unconditional** base rate (the same forward return on *every* bar).

Inference, the desk's shared spirit:

  * a **one-sample / HAC (Newey-West) t** of the signed post-pattern forward return vs 0
    (the inference-bar number) and a Welch t vs the unconditional base rate;
  * a **label-shuffle placebo** — draw the same count of random bars, sign each by a
    coin, and ask how often a random pick looks this good;
  * a **win-rate** (share of events where shorting paid);
  * **costs** (one-way bps × NAV on the round trip; the short side pays borrow);
  * a **myth-check**: does requiring a strict-crow shape and/or a prior uptrend rescue it?

The decisive number is the signed forward return, net, with its HAC t — and whether it
clears t >= 2 against zero on the real tape. (A *crash* signal must produce a positive
signed-short return; if shorting after the pattern does nothing, the crow is busted.)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (1, 3, 5, 10)            # trading-day forward-return horizons


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #
def is_three_black_crows(o0: float, c0: float, h0: float, l0: float,
                         o1: float, c1: float, h1: float, l1: float,
                         o2: float, c2: float, h2: float, l2: float,
                         strict: bool = False) -> int:
    """Return -1 (three black crows) or 0 for the bar triple (oldest -> newest).

    Bars are passed oldest first: (0) = t-2, (1) = t-1, (2) = t (the confirming bar).
    The basic test requires three stacked red bodies, each closing lower, each opening
    within the prior body. ``strict`` additionally requires long bodies (each real body
    is the dominant part of its range, i.e. small lower wicks) — the purist's crow.
    """
    # 1. three red bodies
    if not (c0 < o0 and c1 < o1 and c2 < o2):
        return 0
    # 2. each closes lower than the last
    if not (c2 < c1 < c0):
        return 0
    # 3. each opens within the prior real body (stacking), opens stepping down
    if not (o0 >= o1 >= o2):
        return 0
    if not (c0 < o1 <= o0 and c1 < o2 <= o1):
        return 0
    if strict:
        # long bodies + short lower shadows: the close sits near the low of each bar
        for o, c, h, l in ((o0, c0, h0, l0), (o1, c1, h1, l1), (o2, c2, h2, l2)):
            rng = h - l
            body = o - c
            if rng <= 0:
                return 0
            lower_wick = c - l
            if body < 0.5 * rng:          # body must dominate the range
                return 0
            if lower_wick > 0.25 * rng:   # close near the low (short lower shadow)
                return 0
    return -1


def tbc_signals(bars: pd.DataFrame, strict: bool = False) -> pd.Series:
    """-1 / 0 per bar: -1 where the bar CONFIRMS a three-black-crows (uses t-2, t-1, t)."""
    o = bars["open"].to_numpy(float)
    c = bars["close"].to_numpy(float)
    h = bars["high"].to_numpy(float)
    l = bars["low"].to_numpy(float)
    n = len(bars)
    sig = np.zeros(n, dtype=int)
    for i in range(2, n):
        sig[i] = is_three_black_crows(
            o[i - 2], c[i - 2], h[i - 2], l[i - 2],
            o[i - 1], c[i - 1], h[i - 1], l[i - 1],
            o[i], c[i], h[i], l[i], strict=strict)
    return pd.Series(sig, index=bars.index, name="tbc")


# --------------------------------------------------------------------------- #
# Forward returns (one execution lag: enter next open, exit close +H)
# --------------------------------------------------------------------------- #
def forward_returns(bars: pd.DataFrame, horizon: int) -> np.ndarray:
    """Per-bar forward return entering the NEXT open, exiting the close ``horizon`` later.

    Bar t's value = close[t+horizon] / open[t+1] - 1. NaN where the window overruns.
    This is the unsigned price move; the event study signs it short (bearish pattern).
    """
    o = bars["open"].to_numpy(float)
    c = bars["close"].to_numpy(float)
    n = len(bars)
    out = np.full(n, np.nan)
    for i in range(n):
        e = i + 1
        x = e + horizon - 1
        if e < n and x < n:
            out[i] = c[x] / o[e] - 1.0
    return out


def _trend_up(bars: pd.DataFrame, lookback: int = 10) -> np.ndarray:
    """True where close[t] > close[t-lookback] (a prior uptrend) — for the myth-check.

    A three-black-crows is meant to be a *reversal* of an uptrend; we look at the close
    just BEFORE the three-bar pattern began (t-3) vs lookback earlier.
    """
    c = bars["close"].to_numpy(float)
    n = len(bars)
    out = np.zeros(n, dtype=bool)
    for i in range(lookback + 3, n):
        out[i] = c[i - 3] > c[i - 3 - lookback]
    return out


# --------------------------------------------------------------------------- #
# Event extraction across the basket
# --------------------------------------------------------------------------- #
def collect_events(panel: dict[str, pd.DataFrame], horizon: int,
                   strict: bool = False, require_trend: bool = False,
                   trend_lookback: int = 10) -> pd.DataFrame:
    """Pool three-black-crows events across the basket with their SIGNED forward return.

    For every name, find every confirming bar (optionally the strict-crow shape and/or a
    prior uptrend for the myth-check), attach the forward ``horizon`` return signed SHORT
    (a crash signal: signed_ret = -1 * forward move), drop events whose window overruns.
    Returns columns: ticker, date, signed_ret.
    """
    rows = []
    for tk, bars in panel.items():
        if bars is None or len(bars) < horizon + 5:
            continue
        sig = tbc_signals(bars, strict=strict).to_numpy()
        fwd = forward_returns(bars, horizon)
        idx = bars.index
        tu = _trend_up(bars, trend_lookback) if require_trend else None
        for i in range(len(bars)):
            if sig[i] == 0 or not np.isfinite(fwd[i]):
                continue
            if require_trend and not tu[i]:
                continue
            rows.append({"ticker": tk, "date": idx[i],
                         "signed_ret": float(-1.0 * fwd[i])})
    return pd.DataFrame(rows)


def unconditional_base(panel: dict[str, pd.DataFrame], horizon: int) -> np.ndarray:
    """The unconditional forward-return base rate: every bar's forward H return (unsigned).

    The honest benchmark — what a no-signal trader earns on the same names/window. Because
    the event return is signed SHORT, the relevant always-on drift is the *long* base
    rate's mirror; we keep the raw (long) unconditional pool and compare carefully.
    """
    vals = []
    for bars in panel.values():
        if bars is None or len(bars) < horizon + 5:
            continue
        fwd = forward_returns(bars, horizon)
        vals.append(fwd[np.isfinite(fwd)])
    return np.concatenate(vals) if vals else np.array([])


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def hac_t(sample: np.ndarray) -> float:
    """Newey-West HAC one-sample t of ``sample`` mean against 0 (event ordering)."""
    r = np.asarray(sample, float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 6:
        return float("nan")
    mu = r.mean()
    e = r - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def onesample_t(sample: np.ndarray) -> float:
    """Plain one-sample t of ``sample`` mean against 0."""
    r = np.asarray(sample, float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return float("nan")
    se = r.std(ddof=1) / np.sqrt(r.size)
    return float(r.mean() / se) if se > 0 else float("nan")


def welch_t(sample: np.ndarray, base: np.ndarray) -> float:
    """Welch t of mean(sample) - mean(base) (unequal variance)."""
    a = np.asarray(sample, float); a = a[np.isfinite(a)]
    b = np.asarray(base, float); b = b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / a.size + b.var(ddof=1) / b.size)
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def placebo_pvalue(panel: dict[str, pd.DataFrame], horizon: int, n_events: int,
                   obs_mean: float, n_draws: int = 5000, seed: int = 408) -> dict:
    """Label-shuffle placebo: random bars (same count, coin-signed) vs the observed mean.

    Draw ``n_events`` random bars from the unconditional pool, sign each by a fair coin (a
    random "direction call"), and ask how often the random mean >= the observed signed
    mean. The honest "could a random pick of the same many trades look this good?" test.
    """
    base = unconditional_base(panel, horizon)
    if base.size == 0 or n_events <= 0 or not np.isfinite(obs_mean):
        return {"p_value": float("nan"), "draws": np.array([]), "obs": obs_mean}
    rng = np.random.default_rng(seed)
    means = np.empty(n_draws)
    for i in range(n_draws):
        pick = rng.choice(base, size=n_events, replace=True)
        signs = rng.choice([-1.0, 1.0], size=n_events)
        means[i] = float((signs * pick).mean())
    p = float((means >= obs_mean).mean())
    return {"obs": obs_mean, "p_value": p, "draws": means}


# --------------------------------------------------------------------------- #
# Costs
# --------------------------------------------------------------------------- #
def net_of_costs(signed_mean: float, horizon: int, cost_bps: float = 5.0,
                 borrow_bps_ann: float = 50.0) -> float:
    """Per-event signed-short return net of a one-way round trip + borrow.

    Each event is a fresh short round trip: ``2 * cost_bps`` (in + out) × NAV one-way,
    plus ``borrow_bps_ann`` annualised over the holding period (every event is a short).
    Returns the net per-event mean.
    """
    c = cost_bps / 1e4
    round_trip = 2.0 * c
    borrow = (borrow_bps_ann / 1e4) * (horizon / 252.0)
    return float(signed_mean - round_trip - borrow)


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def summarize(panel: dict[str, pd.DataFrame], horizon: int, n_draws: int = 5000,
              placebo: bool = True, cost_bps: float = 5.0,
              strict: bool = False, require_trend: bool = False) -> dict:
    """Headline stats for one horizon: n, mean, base, win, HAC t, Welch t, placebo p, net."""
    ev = collect_events(panel, horizon, strict=strict, require_trend=require_trend)
    base = unconditional_base(panel, horizon)
    if len(ev) == 0:
        return {"horizon": horizon, "n_events": 0, "mean": float("nan"),
                "base_mean": float("nan"), "win": float("nan"), "t_hac": float("nan"),
                "t_one": float("nan"), "t_welch": float("nan"),
                "p_placebo": float("nan"), "net": float("nan")}
    sr = ev["signed_ret"].to_numpy(float)
    mean = float(sr.mean())
    base_mean = float(base.mean()) if base.size else float("nan")
    win = float((sr > 0).mean())
    t_hac = hac_t(sr)
    t_one = onesample_t(sr)
    # Welch vs the unconditional pool signed SHORT (the always-down "drift" = -base): the
    # crow believer must beat just-shorting-everything-blindly.
    t_welch = welch_t(sr, -base)
    p = (placebo_pvalue(panel, horizon, len(ev), mean, n_draws=n_draws)["p_value"]
         if placebo else float("nan"))
    return {
        "horizon": horizon, "n_events": int(len(ev)),
        "mean": mean, "base_mean": base_mean, "win": win,
        "t_hac": t_hac, "t_one": t_one, "t_welch": t_welch,
        "p_placebo": p, "net": net_of_costs(mean, horizon, cost_bps=cost_bps),
    }


def run_experiment(panel: dict[str, pd.DataFrame], horizons=HORIZONS,
                   n_draws: int = 5000, cost_bps: float = 5.0,
                   strict: bool = False, require_trend: bool = False) -> pd.DataFrame:
    """Run the full event study across horizons -> a tidy results frame."""
    rows = [summarize(panel, h, n_draws=n_draws, cost_bps=cost_bps,
                      strict=strict, require_trend=require_trend) for h in horizons]
    return pd.DataFrame(rows)
