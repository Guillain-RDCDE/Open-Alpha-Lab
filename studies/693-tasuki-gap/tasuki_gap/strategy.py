"""Strategy + inference for Study 693 — the Tasuki Gap candlestick pattern.

The claim (a classic Japanese candlestick **continuation** pattern — Nison, *Japanese
Candlestick Charting Techniques*): the **tasuki gap** says a trend that gaps and then
takes a small, incomplete pullback is a trend that is about to *resume*, not reverse.

    **Upside tasuki gap** (bullish continuation), bars oldest -> newest (0, 1, 2):
      1. bar0: a white (bullish) candle:                       C0 > O0
      2. bar1: a white (bullish) candle that GAPS UP from bar0 (a genuine gap between
         the real bodies — the default, LOOSE gap; the strict myth-check tightens this
         to a full-range gap, L1 > H0):
             min(O1,C1) > max(O0,C0)     and     C1 > O1
      3. bar2: a black (bearish) candle that opens INSIDE bar1's real body and closes
         back INSIDE the gap without fully closing it — the pullback stalls in mid-air:
             C2 < O2,   O1 <= O2 <= C1,   C0 < C2 < O1

    **Downside tasuki gap** (bearish continuation) is the exact mirror: bar0 and bar1
    are both black candles gapping DOWN, and bar2 is a white candle that opens inside
    bar1's body and closes back inside the gap without filling it:
             C0 < O0,  C1 < O1,  max(O1,C1) < min(O0,C0) [loose] / H1 < L0 [strict]
             C2 > O2,  C1 <= O2 <= O1,  O1 < C2 < C0

The gap staying open (not fully filled) is the whole point of the pattern's logic: the
folklore reads it as "the market couldn't even close the gap — the trend has enough
force left to resume." Confirmed at the CLOSE of bar2. Signed by trend direction: an
upside tasuki predicts the market keeps rising (LONG), a downside tasuki predicts it
keeps falling (SHORT).

Measurement (no look-ahead, one documented execution lag):

    Confirm at close of bar2; enter at the NEXT bar's open (bar2+1) — one lag — and
    measure the signed forward 1 / 5 / 10 / 20-day return, signed toward the predicted
    trend direction (+1 * forward move for an upside tasuki, -1 * forward move for a
    downside tasuki). The honest benchmark is the basket's own UNCONDITIONAL forward
    return (matched by direction: an upside event is compared against the plain
    unconditional pool, a downside event against the *negated* pool — "what would a
    trader who went long/short on ANY random day of this basket have earned"), since the
    basket carries its own drift and a naive vs-zero test is contaminated by it.

Inference, the desk's shared spirit:

  * a one-sample **Newey-West (HAC) t** of the signed forward return vs 0 (informational
    — contaminated by the basket's own drift) and a **Welch t** vs the direction-matched
    unconditional base rate (the decisive, drift-neutral comparison);
  * a **hit rate** (share of events where the continuation paid) vs the direction-matched
    base rate's hit rate, both with Wilson intervals;
  * a **mix-matched coin placebo** — draw the same count of random bars from the
    unconditional pool, sign each by a coin weighted to the observed up/down mix, and ask
    how often a random pick looks this good;
  * **Bonferroni across the basket** — a per-ticker (pooled up+down) Welch t, corrected
    for the number of tickers tested simultaneously, so one lucky name in a 30-name
    basket can't pass as "the" signal;
  * a **continuation timer** — the same event study at four holding horizons (1/5/10/20
    trading days) with costs (one-way bps x NAV, round trip on every event; borrow on the
    short/downside leg only).

The decisive number is the direction-signed forward return, net, with its Welch t vs the
direction-matched unconditional base — and whether it clears t >= 2, basket-wide AND
ticker-by-ticker after Bonferroni.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (1, 5, 10, 20)           # trading-day forward-return horizons ("the timer")


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #
def tasuki_signal(o0: float, c0: float, h0: float, l0: float,
                  o1: float, c1: float, h1: float, l1: float,
                  o2: float, c2: float, h2: float, l2: float,
                  true_range_gap: bool = False) -> int:
    """Return +1 (upside tasuki), -1 (downside tasuki) or 0 for the bar triple.

    Bars are passed oldest first: (0) = t-2, (1) = t-1 (the gapping candle, same color
    as bar 0), (2) = t (the confirming pullback candle, opposite color, opening inside
    bar 1's body and closing back inside the gap without filling it).
    ``true_range_gap=True`` enforces the textbook strict gap (bar 1's entire wick range
    clears bar 0's, e.g. ``l1 > h0``); the default is the looser, realistic body-level
    gap (bar 1's real body clears bar 0's real body — liquid US equities rarely gap by
    their whole daily range).
    """
    # -- upside tasuki gap: two white candles gapping up, one black pullback candle
    if c0 > o0 and c1 > o1:
        gap_ok = (l1 > h0) if true_range_gap else (min(o1, c1) > max(o0, c0))
        if gap_ok and c2 < o2 and (o1 <= o2 <= c1) and (c0 < c2 < o1):
            return 1

    # -- downside tasuki gap: two black candles gapping down, one white pullback candle
    if c0 < o0 and c1 < o1:
        gap_ok = (h1 < l0) if true_range_gap else (min(o0, c0) > max(o1, c1))
        if gap_ok and c2 > o2 and (c1 <= o2 <= o1) and (o1 < c2 < c0):
            return -1

    return 0


def tasuki_signals(bars: pd.DataFrame, true_range_gap: bool = False) -> pd.Series:
    """+1 / -1 / 0 per bar: nonzero where the bar CONFIRMS a tasuki gap (uses t-2,t-1,t)."""
    o = bars["open"].to_numpy(float)
    c = bars["close"].to_numpy(float)
    h = bars["high"].to_numpy(float)
    l = bars["low"].to_numpy(float)
    n = len(bars)
    sig = np.zeros(n, dtype=int)
    for i in range(2, n):
        sig[i] = tasuki_signal(
            o[i - 2], c[i - 2], h[i - 2], l[i - 2],
            o[i - 1], c[i - 1], h[i - 1], l[i - 1],
            o[i], c[i], h[i], l[i], true_range_gap=true_range_gap)
    return pd.Series(sig, index=bars.index, name="tasuki")


# --------------------------------------------------------------------------- #
# Forward returns (one execution lag: enter next open, exit close +H)
# --------------------------------------------------------------------------- #
def forward_returns(bars: pd.DataFrame, horizon: int) -> np.ndarray:
    """Per-bar forward return entering the NEXT open, exiting the close ``horizon`` later.

    Bar t's value = close[t+horizon] / open[t+1] - 1. NaN where the window overruns.
    This is the unsigned (long) price move; the event study signs it by trend direction.
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


def _trend_context(bars: pd.DataFrame, lookback: int = 10) -> np.ndarray:
    """Sign of the prior move: +1 up / -1 down / 0 flat, close[t-3] vs close[t-3-lookback].

    For the myth-check: "does the tasuki only work as a genuine continuation of an
    established trend?" We look at the close just BEFORE the three-bar pattern began
    (t-3) vs ``lookback`` sessions earlier.
    """
    c = bars["close"].to_numpy(float)
    n = len(bars)
    out = np.zeros(n, dtype=int)
    for i in range(lookback + 3, n):
        out[i] = 1 if c[i - 3] > c[i - 3 - lookback] else -1
    return out


# --------------------------------------------------------------------------- #
# Event extraction across the basket
# --------------------------------------------------------------------------- #
def collect_events(panel: dict[str, pd.DataFrame], horizon: int,
                   true_range_gap: bool = False, require_trend: bool = False,
                   trend_lookback: int = 10) -> pd.DataFrame:
    """Pool tasuki-gap events across the basket with their trend-signed forward return.

    For every name, find every confirming bar (optionally the strict-gap shape and/or a
    matching prior trend for the myth-check), attach the forward ``horizon`` return
    signed toward the predicted trend direction (signed_ret = direction * forward move),
    drop events whose window overruns. Returns columns: ticker, date, direction,
    signed_ret.
    """
    rows = []
    for tk, bars in panel.items():
        if bars is None or len(bars) < horizon + 15:
            continue
        sig = tasuki_signals(bars, true_range_gap=true_range_gap).to_numpy()
        fwd = forward_returns(bars, horizon)
        idx = bars.index
        tctx = _trend_context(bars, trend_lookback) if require_trend else None
        for i in range(len(bars)):
            d = sig[i]
            if d == 0 or not np.isfinite(fwd[i]):
                continue
            if require_trend and tctx[i] != d:
                continue
            rows.append({"ticker": tk, "date": idx[i], "direction": int(d),
                         "signed_ret": float(d * fwd[i])})
    return pd.DataFrame(rows)


def unconditional_base(panel: dict[str, pd.DataFrame], horizon: int) -> np.ndarray:
    """The unconditional forward-return base rate: every bar's forward H return (unsigned).

    The honest benchmark — what a no-signal trader earns on the same names/window. An
    upside-tasuki event is compared against this pool as-is (a long-everyday baseline);
    a downside-tasuki event is compared against its NEGATION (a short-everyday baseline).
    """
    vals = []
    for bars in panel.values():
        if bars is None or len(bars) < horizon + 15:
            continue
        fwd = forward_returns(bars, horizon)
        vals.append(fwd[np.isfinite(fwd)])
    return np.concatenate(vals) if vals else np.array([])


# --------------------------------------------------------------------------- #
# Inference primitives
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


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a binomial share k/n."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


def _matched_base(base: np.ndarray, n_up: int, n_down: int) -> np.ndarray:
    """Direction-matched null pool: the base pool signed +1 (as many times as up-events)
    concatenated with the base pool signed -1 (as many times as down-events) — "what an
    unconditional long-or-short trader earns, mixed in the same proportion as the real
    event sample." Sample sizes need not match between the event and null pools for a
    Welch test, but keeping the SHAPE of the mix (all of +base if any up-events exist,
    all of -base if any down-events exist) avoids under- or over-weighting either side.
    """
    parts = []
    if n_up > 0:
        parts.append(base)
    if n_down > 0:
        parts.append(-base)
    return np.concatenate(parts) if parts else np.array([])


def placebo_pvalue(panel: dict[str, pd.DataFrame], horizon: int, n_up: int, n_down: int,
                   obs_mean: float, n_draws: int = 5000, seed: int = 693) -> dict:
    """Mix-matched coin placebo: random bars (same count, coin-signed to the observed
    up/down mix) vs the observed mean.

    Draw ``n_up + n_down`` random bars from the unconditional pool, sign each by a coin
    weighted to match the observed fraction of upside vs downside events (not a plain
    50/50 coin — a real mix that is mostly upside gaps, say, should be judged against a
    null that is mostly-long too), and ask how often the random mean >= the observed
    signed mean. The honest "could a random pick of the same many trades, in the same
    long/short mix, look this good?" test.
    """
    base = unconditional_base(panel, horizon)
    n_events = n_up + n_down
    if base.size == 0 or n_events <= 0 or not np.isfinite(obs_mean):
        return {"p_value": float("nan"), "draws": np.array([]), "obs": obs_mean}
    p_up = n_up / n_events
    rng = np.random.default_rng(seed)
    means = np.empty(n_draws)
    for i in range(n_draws):
        pick = rng.choice(base, size=n_events, replace=True)
        signs = np.where(rng.random(n_events) < p_up, 1.0, -1.0)
        means[i] = float((signs * pick).mean())
    p = float((means >= obs_mean).mean())
    return {"obs": obs_mean, "p_value": p, "draws": means, "p_up": p_up}


# --------------------------------------------------------------------------- #
# Costs
# --------------------------------------------------------------------------- #
def event_net_returns(ev: pd.DataFrame, horizon: int, cost_bps: float = 5.0,
                      borrow_bps_ann: float = 50.0) -> np.ndarray:
    """Per-event signed return net of a one-way round trip + short borrow.

    Every event pays a fresh round trip: ``2 * cost_bps`` (in + out) x NAV, one-way each
    leg. Downside (short) events additionally pay ``borrow_bps_ann`` annualised over the
    holding period; upside (long) events pay no borrow.
    """
    c = cost_bps / 1e4
    round_trip = 2.0 * c
    borrow = (borrow_bps_ann / 1e4) * (horizon / 252.0)
    d = ev["direction"].to_numpy(float)
    sr = ev["signed_ret"].to_numpy(float)
    cost = round_trip + np.where(d < 0, borrow, 0.0)
    return sr - cost


# --------------------------------------------------------------------------- #
# Bonferroni across the basket — one pooled (up+down) Welch t per ticker
# --------------------------------------------------------------------------- #
def per_ticker_stats(panel: dict[str, pd.DataFrame], horizon: int,
                     true_range_gap: bool = False, min_events: int = 6) -> pd.DataFrame:
    """One row per ticker with >= ``min_events`` tasuki events: n, mean, HAC t, Welch t.

    ``t_hac`` is the one-sample HAC t of the trend-signed mean against **zero** —
    informational only, because names carry their own idiosyncratic drift, which biases
    a vs-zero test. ``t_welch`` compares the pooled (up+down) event sample against
    **that ticker's own** direction-matched unconditional pool (drift-neutral) and is the
    number the Bonferroni correction is applied to: with ``k`` tickers tested
    independently, the per-ticker two-sided significance bar becomes
    ``|t| >= z(1 - 0.025/k)`` instead of the usual 1.96 — so a single lucky name in a
    wide basket can't pass as *the* signal.
    """
    rows = []
    for tk, bars in panel.items():
        if bars is None or len(bars) < horizon + 15:
            continue
        sig = tasuki_signals(bars, true_range_gap=true_range_gap).to_numpy()
        fwd = forward_returns(bars, horizon)
        mask = (sig != 0) & np.isfinite(fwd)
        if mask.sum() < min_events:
            continue
        d = sig[mask]
        sr = d * fwd[mask]
        own_base = fwd[np.isfinite(fwd)]
        matched = _matched_base(own_base, int((d > 0).sum()), int((d < 0).sum()))
        rows.append({"ticker": tk, "n": int(mask.sum()), "mean": float(sr.mean()),
                     "t_hac": hac_t(sr), "t_welch": welch_t(sr, matched)})
    return pd.DataFrame(rows)


def bonferroni_z(k: int, alpha: float = 0.05) -> float:
    """Two-sided Bonferroni-corrected critical |z| for ``k`` simultaneous tests.

    Uses the inverse-normal approximation (large-sample HAC/one-sample t's are
    approximately normal) so the module stays scipy-free.
    """
    from math import erf, sqrt

    if k <= 1:
        return 1.959963984540054
    target = 1.0 - alpha / (2.0 * k)

    def phi(x: float) -> float:
        return 0.5 * (1.0 + erf(x / sqrt(2.0)))

    lo, hi = 0.0, 8.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if phi(mid) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def summarize(panel: dict[str, pd.DataFrame], horizon: int, n_draws: int = 5000,
              placebo: bool = True, cost_bps: float = 5.0, borrow_bps_ann: float = 50.0,
              true_range_gap: bool = False, require_trend: bool = False) -> dict:
    """Headline stats for one horizon: n, mean, base, hit rate, HAC t, Welch t, placebo p,
    net — pooled (up+down) AND split by direction."""
    ev = collect_events(panel, horizon, true_range_gap=true_range_gap,
                        require_trend=require_trend)
    base = unconditional_base(panel, horizon)
    empty = {"horizon": horizon, "n_events": 0, "n_up": 0, "n_down": 0,
             "mean": float("nan"), "base_mean": float("nan"),
             "hit": float("nan"), "base_hit": float("nan"),
             "hit_lo": float("nan"), "hit_hi": float("nan"),
             "t_hac": float("nan"), "t_one": float("nan"), "t_welch": float("nan"),
             "p_placebo": float("nan"), "net": float("nan"),
             "up_mean": float("nan"), "up_t_welch": float("nan"), "n_up_ev": 0,
             "down_mean": float("nan"), "down_t_welch": float("nan"), "n_down_ev": 0}
    if len(ev) == 0:
        return empty
    sr = ev["signed_ret"].to_numpy(float)
    d = ev["direction"].to_numpy(int)
    n_up, n_down = int((d > 0).sum()), int((d < 0).sum())
    mean = float(sr.mean())
    matched = _matched_base(base, n_up, n_down)
    base_mean = float(matched.mean()) if matched.size else float("nan")
    hit_k = int((sr > 0).sum())
    hit = hit_k / len(sr)
    hit_lo, hit_hi = wilson_interval(hit_k, len(sr))
    base_hit = float((matched > 0).mean()) if matched.size else float("nan")
    t_hac = hac_t(sr)
    t_one = onesample_t(sr)
    t_welch = welch_t(sr, matched)
    p = (placebo_pvalue(panel, horizon, n_up, n_down, mean, n_draws=n_draws)["p_value"]
         if placebo else float("nan"))
    net = float(event_net_returns(ev, horizon, cost_bps=cost_bps,
                                  borrow_bps_ann=borrow_bps_ann).mean())

    up = ev[ev["direction"] > 0]["signed_ret"].to_numpy(float)
    down = ev[ev["direction"] < 0]["signed_ret"].to_numpy(float)
    up_t = welch_t(up, base) if len(up) else float("nan")
    down_t = welch_t(down, -base) if len(down) else float("nan")

    return {
        "horizon": horizon, "n_events": int(len(ev)), "n_up": n_up, "n_down": n_down,
        "mean": mean, "base_mean": base_mean, "hit": hit, "base_hit": base_hit,
        "hit_lo": hit_lo, "hit_hi": hit_hi,
        "t_hac": t_hac, "t_one": t_one, "t_welch": t_welch,
        "p_placebo": p, "net": net,
        "up_mean": float(up.mean()) if len(up) else float("nan"),
        "up_t_welch": up_t, "n_up_ev": len(up),
        "down_mean": float(down.mean()) if len(down) else float("nan"),
        "down_t_welch": down_t, "n_down_ev": len(down),
    }


def run_experiment(panel: dict[str, pd.DataFrame], horizons=HORIZONS,
                   n_draws: int = 5000, cost_bps: float = 5.0,
                   true_range_gap: bool = False, require_trend: bool = False) -> pd.DataFrame:
    """Run the full event study across horizons -> a tidy results frame ("the timer")."""
    rows = [summarize(panel, h, n_draws=n_draws, cost_bps=cost_bps,
                      true_range_gap=true_range_gap, require_trend=require_trend)
            for h in horizons]
    return pd.DataFrame(rows)
