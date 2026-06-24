"""Strategy + inference for Study 407 — Dark Cloud Cover & Piercing Line.

The claim (a classic two-candle reversal pair from Nison's candlestick canon):

  * **Piercing Line** (bullish reversal): a long *down* day, then an *up* day that gaps
    DOWN at the open (opens below the prior low) but rallies to close back **above the
    midpoint** of the prior down body — a failed sell-off that marks a bottom; buy, the
    next day(s) rise.
  * **Dark Cloud Cover** (bearish reversal): the mirror — a long *up* day, then a *down*
    day that gaps UP at the open (opens above the prior high) but sells off to close back
    **below the midpoint** of the prior up body — a failed rally that marks a top; sell,
    the next day(s) fall.

We test the pair as one clean event study on a fixed liquid large-cap + SPY basket.

Precise detector (the textbook OHLC rules):

    Let O0,H0,L0,C0 be the prior bar and O1,H1,L1,C1 the current bar.
    body0 = |C0-O0|, mid0 = (O0+C0)/2.
    Piercing Line  : C0 < O0          (prior down day, real body)
                     C1 > O1          (current up day)
                     O1 < L0          (opens below the prior low — a downside gap)
                     C1 > mid0        (closes back above the prior body midpoint)
                     C1 < O0          (but does NOT fully engulf — close stays under prior open)
    Dark Cloud Cover : the mirror (prior up day; opens above prior high; closes below mid0;
                     close stays above prior open, i.e. not a full bearish engulfing).

We require a non-trivial prior body (``min_body_frac`` of price) so a doji prior bar with no
real midpoint cannot trigger the pattern.

Measurement (no look-ahead, one documented lag):

    The pattern is confirmed at the **close** of the current bar (day t). We enter at the
    **next** bar's open (day t+1) — one execution lag — and measure the forward
    1 / 3 / 5 / 10-day return, *signed by the pattern direction* (long after a Piercing,
    short after a Dark Cloud). The honest benchmark is the **unconditional** base rate: the
    same forward return measured on *every* bar of the same names (no-signal drift).

Inference (the desk's shared spirit):

  * a **one-sample / HAC (Newey-West) t** of the signed post-pattern forward return vs 0
    (the inference-bar number) and a Welch t vs the unconditional base pool;
  * a **label-shuffle placebo** — draw the same number of random bars, sign each by a fair
    coin, and ask how often a random pick beats the observed mean;
  * a **win-rate** vs the unconditional base rate;
  * **costs** (one-way bps x NAV, charged on the round trip; shorts pay borrow);
  * a **myth-check**: does requiring a prior trend / a volume spike rescue it?

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
def is_dark_cloud_or_piercing(o0: float, h0: float, l0: float, c0: float,
                              o1: float, h1: float, l1: float, c1: float,
                              min_body_frac: float = 0.001) -> int:
    """Return +1 (Piercing Line, bullish), -1 (Dark Cloud Cover, bearish), or 0.

    Two-bar reversal twins with a *deep penetration past the prior body midpoint* but
    short of a full engulfing (which is its own pattern, Study 402):

    - Piercing Line   : prior long down day; current up day gaps below the prior low and
      closes back above the prior body's midpoint (but below the prior open).
    - Dark Cloud Cover: prior long up day; current down day gaps above the prior high and
      closes back below the prior body's midpoint (but above the prior open).

    ``min_body_frac`` guards against a near-doji prior bar (no meaningful midpoint).
    """
    body0 = abs(c0 - o0)
    ref = max(abs(c0), 1e-12)
    if body0 / ref < min_body_frac:
        return 0
    mid0 = 0.5 * (o0 + c0)
    # Piercing Line (bullish): prior down, current up, gap-down open, close past midpoint,
    # but not a full bullish engulfing (close stays below the prior open).
    if (c0 < o0) and (c1 > o1) and (o1 < l0) and (c1 > mid0) and (c1 < o0):
        return 1
    # Dark Cloud Cover (bearish): prior up, current down, gap-up open, close below midpoint,
    # but not a full bearish engulfing (close stays above the prior open).
    if (c0 > o0) and (c1 < o1) and (o1 > h0) and (c1 < mid0) and (c1 > o0):
        return -1
    return 0


def twin_signals(bars: pd.DataFrame, min_body_frac: float = 0.001) -> pd.Series:
    """+1 / -1 / 0 per bar: the twin direction confirmed at that bar's close.

    +1 = Piercing Line (bullish), -1 = Dark Cloud Cover (bearish). Uses bars t-1 and t;
    indexed identically to ``bars``.
    """
    o = bars["open"].to_numpy(float)
    h = bars["high"].to_numpy(float)
    l = bars["low"].to_numpy(float)
    c = bars["close"].to_numpy(float)
    n = len(bars)
    sig = np.zeros(n, dtype=int)
    for i in range(1, n):
        sig[i] = is_dark_cloud_or_piercing(
            o[i - 1], h[i - 1], l[i - 1], c[i - 1], o[i], h[i], l[i], c[i],
            min_body_frac=min_body_frac)
    return pd.Series(sig, index=bars.index, name="twin")


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


def _trend(bars: pd.DataFrame, lookback: int = 10) -> np.ndarray:
    """Signed prior trend: +1 if close[t] > close[t-lookback], -1 if below, 0 early bars."""
    c = bars["close"].to_numpy(float)
    n = len(bars)
    out = np.zeros(n, dtype=int)
    for i in range(lookback, n):
        out[i] = 1 if c[i] > c[i - lookback] else -1
    return out


def _vol_spike(bars: pd.DataFrame, lookback: int = 20, mult: float = 1.5) -> np.ndarray:
    """True where volume[t] > mult * trailing mean volume — for the myth-check."""
    v = bars["volume"].to_numpy(float)
    n = len(bars)
    out = np.zeros(n, dtype=bool)
    for i in range(lookback, n):
        m = v[i - lookback:i].mean()
        out[i] = m > 0 and v[i] > mult * m
    return out


# --------------------------------------------------------------------------- #
# Event extraction across the basket
# --------------------------------------------------------------------------- #
def collect_events(panel: dict[str, pd.DataFrame], horizon: int,
                   require_trend: bool = False, require_volume: bool = False,
                   trend_lookback: int = 10, vol_lookback: int = 20,
                   vol_mult: float = 1.5, min_body_frac: float = 0.001) -> pd.DataFrame:
    """Pool twin events across the basket with their SIGNED forward return.

    For every name, find every Piercing/Dark-Cloud bar (optionally filtered by a prior
    trend in the textbook-required direction and/or a volume spike), attach the forward
    ``horizon`` return signed by the pattern direction (long after a Piercing, short after
    a Dark Cloud), drop events whose window overruns. Columns: ticker, date, dir, signed_ret.
    """
    rows = []
    for tk, bars in panel.items():
        if bars is None or len(bars) < horizon + 5:
            continue
        sig = twin_signals(bars, min_body_frac=min_body_frac).to_numpy()
        fwd = forward_returns(bars, horizon)
        idx = bars.index
        tr = _trend(bars, trend_lookback) if require_trend else None
        vs = _vol_spike(bars, vol_lookback, vol_mult) if require_volume else None
        for i in range(len(bars)):
            d = sig[i]
            if d == 0 or not np.isfinite(fwd[i]):
                continue
            if require_trend:
                # Piercing (bullish) needs a prior downtrend; Dark Cloud a prior uptrend.
                if d > 0 and tr[i] >= 0:
                    continue
                if d < 0 and tr[i] <= 0:
                    continue
            if require_volume and not vs[i]:
                continue
            rows.append({"ticker": tk, "date": idx[i], "dir": int(d),
                         "signed_ret": float(d * fwd[i])})
    return pd.DataFrame(rows)


def unconditional_base(panel: dict[str, pd.DataFrame], horizon: int) -> np.ndarray:
    """The unconditional forward-return base rate: every bar's forward H return (unsigned).

    The honest benchmark — what a no-signal trader earns on the same names/window.
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
                   n_draws: int = 5000, seed: int = 407) -> dict:
    """Label-shuffle placebo: random bars (same count), coin-signed, vs the observed mean.

    Draw ``n_events`` random bars from the unconditional pool, sign each by a fair coin (a
    random "direction call"), and ask how often the random mean >= observed. The honest
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

    Each event is a fresh round trip: ``2 * cost_bps`` (in + out) x NAV one-way. The short
    legs (a ``short_share`` of events are Dark Cloud) pay ``borrow_bps_ann`` annualised
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


def leg_split(panel: dict[str, pd.DataFrame], horizon: int) -> dict:
    """Split the signed return by leg: Piercing (bullish) vs Dark Cloud (bearish)."""
    ev = collect_events(panel, horizon)
    out = {}
    for name, sign in (("bull", 1), ("bear", -1)):
        leg = ev[ev["dir"] == sign]["signed_ret"].to_numpy(float)
        if leg.size:
            out[name] = {"n": int(leg.size), "mean": float(leg.mean()),
                         "t_hac": hac_t(leg), "win": float((leg > 0).mean())}
        else:
            out[name] = {"n": 0, "mean": float("nan"), "t_hac": float("nan"),
                         "win": float("nan")}
    return out


def run_experiment(panel: dict[str, pd.DataFrame], horizons=HORIZONS,
                   n_draws: int = 5000, cost_bps: float = 5.0) -> pd.DataFrame:
    """Run the full event study across horizons -> a tidy results frame."""
    rows = [summarize(panel, h, n_draws=n_draws, cost_bps=cost_bps) for h in horizons]
    return pd.DataFrame(rows)
