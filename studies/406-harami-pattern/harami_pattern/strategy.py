"""Strategy + inference for Study 406 — the Harami candlestick pattern.

The claim (a classic two-candle reversal in candlestick lore): a **bullish harami** — a
large down candle followed by a small up candle whose real body sits *inside* the prior
body — marks a bottom, so the next day(s) rise; a **bearish harami** marks a top, so the
next day(s) fall. The harami ("pregnant") is the geometric inverse of the engulfing
pattern: where engulfing has the *second* body swallow the first, harami has the *first*
body contain the second. We test it as a clean event study on a fixed liquid large-cap +
SPY basket.

Precise detector (real-body harami, the textbook definition):

    Let O0,C0 be the prior bar's open/close and O1,C1 the current bar's.
    body0 = |C0-O0|, body1 = |C1-O1|.
    The current real body must sit fully INSIDE the prior real body and be strictly
    smaller, with opposite colours.

    Bullish harami  : C0 < O0   (prior red, the big down day)
                      C1 > O1   (curr green, the small up day)
                      min(O1,C1) >= min(O0,C0)  and  max(O1,C1) <= max(O0,C0)  (inside)
                      body1 < body0             (strictly smaller body)
    Bearish harami  : the mirror (prior green/big up, curr red/small down, inside, smaller).

Measurement (no look-ahead, one documented lag):

    The pattern is confirmed at the **close** of the current bar (day t). We enter at the
    **next** bar's open (day t+1) — one execution lag — and measure the forward
    1 / 3 / 5 / 10-day return, *signed by the pattern direction* (long after bullish,
    short after bearish). The honest benchmark is the **unconditional** base rate: the
    same forward return measured on *every* bar of the same names (what you'd get with no
    signal at all).

Inference, the desk's shared spirit:

  * a **one-sample / HAC (Newey-West) t** of the signed post-pattern forward return vs 0
    (the inference-bar number) and a Welch t vs the unconditional base rate;
  * a **label-shuffle placebo** — draw the same many bars at random, sign each by a coin,
    and ask how often a random pick looks this good;
  * a **win-rate**;
  * **costs** (one-way bps × NAV, charged on the round trip; shorts pay borrow);
  * a **myth-check**: does requiring a prior trend (downtrend for bullish, uptrend for
    bearish) and/or a volume contraction on the inside day rescue it?

The decisive number is the day-after (and few-day) signed forward return, net, with its
HAC t — and whether it clears t >= 2 against zero on the real tape.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (1, 3, 5, 10)            # trading-day forward-return horizons


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #
def is_harami(o0: float, c0: float, o1: float, c1: float) -> int:
    """Return +1 (bullish), -1 (bearish), or 0 for the prior->current bar pair.

    Real-body harami (the textbook definition): the current real body must sit fully
    inside the prior real body and be strictly smaller, with opposite colours.
    """
    body0 = abs(c0 - o0)
    body1 = abs(c1 - o1)
    if body0 <= 0 or body1 >= body0:
        return 0
    hi0, lo0 = max(o0, c0), min(o0, c0)
    hi1, lo1 = max(o1, c1), min(o1, c1)
    inside = (lo1 >= lo0) and (hi1 <= hi0)
    if not inside:
        return 0
    # bullish: prior red (big down), current green (small up) inside
    if (c0 < o0) and (c1 > o1):
        return 1
    # bearish: prior green (big up), current red (small down) inside
    if (c0 > o0) and (c1 < o1):
        return -1
    return 0


def harami_signals(bars: pd.DataFrame) -> pd.Series:
    """+1 / -1 / 0 per bar: the harami direction confirmed at that bar's close.

    Uses bars t-1 and t; NaN-safe. Indexed identically to ``bars``.
    """
    o = bars["open"].to_numpy(float)
    c = bars["close"].to_numpy(float)
    n = len(bars)
    sig = np.zeros(n, dtype=int)
    for i in range(1, n):
        sig[i] = is_harami(o[i - 1], c[i - 1], o[i], c[i])
    return pd.Series(sig, index=bars.index, name="harami")


# --------------------------------------------------------------------------- #
# Forward returns (one execution lag: enter next open, exit close +H)
# --------------------------------------------------------------------------- #
def forward_returns(bars: pd.DataFrame, horizon: int) -> np.ndarray:
    """Per-bar forward return entering the NEXT open, exiting the close ``horizon`` later.

    Bar t's value = close[t+horizon] / open[t+1] - 1. NaN where the window overruns.
    This is the unsigned price move; the event study signs it by pattern direction.
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


def _trend_down(bars: pd.DataFrame, lookback: int = 10) -> np.ndarray:
    """True where close[t] < close[t-lookback] (a prior downtrend) — for the myth-check."""
    c = bars["close"].to_numpy(float)
    n = len(bars)
    out = np.zeros(n, dtype=bool)
    for i in range(lookback, n):
        out[i] = c[i] < c[i - lookback]
    return out


def _vol_contraction(bars: pd.DataFrame, lookback: int = 20, mult: float = 1.0) -> np.ndarray:
    """True where volume[t] < mult * trailing mean volume — for the myth-check.

    The harami's inside day is conventionally read as a pause / loss of conviction, so a
    *contraction* in volume (not a spike) is the natural confirming filter for it.
    """
    v = bars["volume"].to_numpy(float)
    n = len(bars)
    out = np.zeros(n, dtype=bool)
    for i in range(lookback, n):
        m = v[i - lookback:i].mean()
        out[i] = m > 0 and v[i] < mult * m
    return out


# --------------------------------------------------------------------------- #
# Event extraction across the basket
# --------------------------------------------------------------------------- #
def collect_events(panel: dict[str, pd.DataFrame], horizon: int,
                   require_trend: bool = False, require_volume: bool = False,
                   trend_lookback: int = 10, vol_lookback: int = 20,
                   vol_mult: float = 1.0) -> pd.DataFrame:
    """Pool harami events across the basket with their SIGNED forward return.

    For every name, find every harami bar (optionally filtered by a prior trend in the
    folk-predicted direction and/or a volume contraction on the inside day for the
    myth-check), attach the forward ``horizon`` return signed by the pattern direction
    (long after bullish, short after bearish), drop events whose window overruns. Returns
    columns: ticker, date, dir, signed_ret.
    """
    rows = []
    for tk, bars in panel.items():
        if bars is None or len(bars) < horizon + 5:
            continue
        sig = harami_signals(bars).to_numpy()
        fwd = forward_returns(bars, horizon)
        idx = bars.index
        td = _trend_down(bars, trend_lookback) if require_trend else None
        vc = _vol_contraction(bars, vol_lookback, vol_mult) if require_volume else None
        for i in range(len(bars)):
            d = sig[i]
            if d == 0 or not np.isfinite(fwd[i]):
                continue
            if require_trend:
                # bullish needs a prior downtrend; bearish needs a prior uptrend
                if d > 0 and not td[i]:
                    continue
                if d < 0 and td[i]:
                    continue
            if require_volume and not vc[i]:
                continue
            rows.append({"ticker": tk, "date": idx[i], "dir": int(d),
                         "signed_ret": float(d * fwd[i])})
    return pd.DataFrame(rows)


def unconditional_base(panel: dict[str, pd.DataFrame], horizon: int) -> np.ndarray:
    """The unconditional forward-return base rate: every bar's forward H return (unsigned).

    The honest benchmark — what a no-signal trader earns on the same names/window. The
    harami event return is signed long/short (so roughly market-neutral); we report the
    unconditional LONG mean so the reader sees the always-up drift the bullish leg must beat.
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
                   n_draws: int = 5000, seed: int = 406) -> dict:
    """Label-shuffle placebo: random bars (same count, coin-signed) vs the observed mean.

    We draw ``n_events`` random bars from the unconditional pool, sign each by a fair coin
    (a random "direction call"), and ask how often the random mean >= observed. The honest
    "could a random pick of the same many trades look this good?" test.
    """
    base = unconditional_base(panel, horizon)
    if base.size == 0 or n_events <= 0:
        return {"obs": float("nan"), "p_value": float("nan"), "draws": np.array([])}
    ev = collect_events(panel, horizon)
    obs = float(ev["signed_ret"].mean()) if len(ev) else float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_draws)
    for i in range(n_draws):
        pick = rng.choice(base, size=n_events, replace=True)
        signs = rng.choice([-1.0, 1.0], size=n_events)
        means[i] = float((signs * pick).mean())
    p = float((means >= obs).mean())
    return {"obs": obs, "p_value": p, "draws": means}


# --------------------------------------------------------------------------- #
# Costs
# --------------------------------------------------------------------------- #
def net_of_costs(signed_mean: float, horizon: int, cost_bps: float = 5.0,
                 borrow_bps_ann: float = 50.0, short_share: float = 0.5) -> float:
    """Per-event signed return net of a one-way round trip + borrow on the short share.

    Each event is a fresh round trip: ``2 * cost_bps`` (in + out) × NAV one-way. The
    short legs (a ``short_share`` of events are bearish) pay ``borrow_bps_ann`` annualised
    over the holding period. Returns the net per-event mean.
    """
    c = cost_bps / 1e4
    round_trip = 2.0 * c
    borrow = short_share * (borrow_bps_ann / 1e4) * (horizon / 252.0)
    return float(signed_mean - round_trip - borrow)


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def summarize(panel: dict[str, pd.DataFrame], horizon: int, n_draws: int = 5000,
              placebo: bool = True, cost_bps: float = 5.0,
              require_trend: bool = False, require_volume: bool = False) -> dict:
    """Headline stats for one horizon: n, mean, base, win, HAC t, Welch t, placebo p, net."""
    ev = collect_events(panel, horizon, require_trend=require_trend,
                        require_volume=require_volume)
    base = unconditional_base(panel, horizon)
    if len(ev) == 0:
        return {"horizon": horizon, "n_events": 0, "mean": float("nan"),
                "base_long": float("nan"), "win": float("nan"), "t_hac": float("nan"),
                "t_one": float("nan"), "t_welch": float("nan"),
                "p_placebo": float("nan"), "net": float("nan"),
                "n_bull": 0, "n_bear": 0}
    sr = ev["signed_ret"].to_numpy(float)
    mean = float(sr.mean())
    base_long = float(base.mean()) if base.size else float("nan")
    win = float((sr > 0).mean())
    t_hac = hac_t(sr)
    t_one = onesample_t(sr)
    t_welch = welch_t(sr, base)
    p = (placebo_pvalue(panel, horizon, len(ev), n_draws=n_draws)["p_value"]
         if placebo else float("nan"))
    return {
        "horizon": horizon, "n_events": int(len(ev)),
        "n_bull": int((ev["dir"] > 0).sum()), "n_bear": int((ev["dir"] < 0).sum()),
        "mean": mean, "base_long": base_long, "win": win,
        "t_hac": t_hac, "t_one": t_one, "t_welch": t_welch,
        "p_placebo": p, "net": net_of_costs(mean, horizon, cost_bps=cost_bps),
    }


def run_experiment(panel: dict[str, pd.DataFrame], horizons=HORIZONS,
                   n_draws: int = 5000, cost_bps: float = 5.0) -> pd.DataFrame:
    """Run the full event study across horizons -> a tidy results frame."""
    rows = [summarize(panel, h, n_draws=n_draws, cost_bps=cost_bps) for h in horizons]
    return pd.DataFrame(rows)
