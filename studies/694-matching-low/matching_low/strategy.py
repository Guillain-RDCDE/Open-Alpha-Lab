"""Strategy + inference for Study 694 — the Matching Low candlestick pattern.

The claim (a classic Japanese candlestick **reversal** figure — Nison, *Japanese
Candlestick Charting Techniques*): the **matching low** is two consecutive down (black)
candles whose CLOSES land at essentially the same price — "a double-bottom in miniature."
The lore reads the repeated close as a support level the market twice failed to break:
buyers stepped in at the exact same price twice, so the decline should **reverse
upward**.

    Matching low, bars oldest -> newest (0, 1) — bar 1 is the CONFIRMING bar:
      1. bar0: a down (bearish) candle:                  C0 < O0
      2. bar1: a down (bearish) candle:                  C1 < O1
      3. the two closes MATCH within a tolerance ``tol`` (relative to bar0's close,
         the default, LOOSE match a realistic scanner would accept — real closes almost
         never tie to the tick; the strict myth-check tightens ``tol`` to a near-exact
         match):
             |C1 - C0| / |C0| <= tol

Confirmed at the close of bar1 — no look-ahead. Signed **long only**: the pattern makes
no short-side claim, so the honest comparison is the LONG event sample against the
basket's own **unconditional** forward-return pool (what a no-signal, always-long trader
earns on the same names/window) — not a directional/mix-matched comparison (there is no
short leg to match), and not a comparison to zero (which would just measure the basket's
ordinary up-drift, present with or without any pattern).

Measurement (no look-ahead, one documented execution lag):

    Confirm at close of bar1; enter at the NEXT bar's open (bar1+1) — one lag — and
    measure the forward 1 / 5 / 10 / 20-day return ("the timer").

Inference, the desk's shared spirit:

  * a one-sample **Newey-West (HAC) t** of the forward return vs 0 (informational —
    contaminated by the basket's own drift) and a **Welch t** vs the unconditional base
    rate (the decisive, drift-neutral comparison — "forward returns vs base rate");
  * a **hit rate** (share of events with a positive forward return) vs the base rate's
    hit rate, both with Wilson intervals;
  * a **random-draw placebo** — draw the same count of random bars from the
    unconditional pool and ask how often a random pick's mean beats the observed one;
  * **Bonferroni across the basket** — a per-ticker Welch t, corrected for the number of
    tickers tested simultaneously, so one lucky name in a 30-name basket can't pass as
    "the" signal;
  * a **long timer** — the same event study at four holding horizons (1/5/10/20 trading
    days) with one-way round-trip costs (no borrow — the pattern is long-only).

The decisive number is the forward return, net, with its Welch t vs the unconditional
base — and whether it clears t >= 2, basket-wide AND ticker-by-ticker after Bonferroni.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (1, 5, 10, 20)           # trading-day forward-return horizons ("the timer")
DEFAULT_TOL = 0.0015                # loose match: closes within 0.15% of each other
STRICT_TOL = 0.0003                 # strict/purist match: closes within 0.03%


# --------------------------------------------------------------------------- #
# Detector
# --------------------------------------------------------------------------- #
def matching_low_signal(o0: float, c0: float, o1: float, c1: float,
                        tol: float = DEFAULT_TOL) -> bool:
    """True iff bars (0, 1) form a matching low: two down candles, closes within
    ``tol`` (relative to bar0's close) of each other. Bar1 is the confirming bar."""
    if not (c0 < o0 and c1 < o1):
        return False
    denom = abs(c0) if abs(c0) > 1e-9 else 1e-9
    return abs(c1 - c0) / denom <= tol


def matching_low_signals(bars: pd.DataFrame, tol: float = DEFAULT_TOL) -> pd.Series:
    """Boolean per bar: True where the bar CONFIRMS a matching low (uses t-1, t)."""
    o = bars["open"].to_numpy(float)
    c = bars["close"].to_numpy(float)
    n = len(bars)
    down = c < o
    sig = np.zeros(n, dtype=bool)
    if n > 1:
        close_match = np.abs(c[1:] - c[:-1]) <= tol * np.maximum(np.abs(c[:-1]), 1e-9)
        sig[1:] = down[:-1] & down[1:] & close_match
    return pd.Series(sig, index=bars.index, name="matching_low")


# --------------------------------------------------------------------------- #
# Forward returns (one execution lag: enter next open, exit close +H)
# --------------------------------------------------------------------------- #
def forward_returns(bars: pd.DataFrame, horizon: int) -> np.ndarray:
    """Per-bar forward return entering the NEXT open, exiting the close ``horizon`` later.

    Bar t's value = close[t+horizon] / open[t+1] - 1. NaN where the window overruns.
    Unsigned (the pattern is a long-only claim: a matching low predicts a bounce UP).
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


def _prior_downtrend(bars: pd.DataFrame, lookback: int = 10) -> np.ndarray:
    """Boolean per bar: was the market already declining BEFORE the two-candle pattern?

    For the myth-check: "does a matching low only mark support after a genuine prior
    decline?" We compare the close just before the pattern began (t-2, the session
    before the first down candle) against its value ``lookback`` sessions earlier —
    causal, uses only data known before the pattern's first bar.
    """
    c = bars["close"].to_numpy(float)
    n = len(bars)
    out = np.zeros(n, dtype=bool)
    if n > lookback + 2:
        idx = np.arange(lookback + 2, n)
        out[idx] = c[idx - 2] < c[idx - 2 - lookback]
    return out


# --------------------------------------------------------------------------- #
# Event extraction across the basket
# --------------------------------------------------------------------------- #
def collect_events(panel: dict[str, pd.DataFrame], horizon: int, tol: float = DEFAULT_TOL,
                   require_downtrend: bool = False, trend_lookback: int = 10
                   ) -> pd.DataFrame:
    """Pool matching-low events across the basket with their forward ``horizon`` return.

    For every name, find every confirming bar (optionally restricted to a genuine prior
    downtrend for the myth-check), attach the forward return, drop events whose window
    overruns. Returns columns: ticker, date, ret.
    """
    rows = []
    for tk, bars in panel.items():
        if bars is None or len(bars) < horizon + trend_lookback + 15:
            continue
        sig = matching_low_signals(bars, tol=tol).to_numpy()
        fwd = forward_returns(bars, horizon)
        idx = bars.index
        ctx = _prior_downtrend(bars, trend_lookback) if require_downtrend else None
        for i in range(len(bars)):
            if not sig[i] or not np.isfinite(fwd[i]):
                continue
            if require_downtrend and not ctx[i]:
                continue
            rows.append({"ticker": tk, "date": idx[i], "ret": float(fwd[i])})
    return pd.DataFrame(rows)


def unconditional_base(panel: dict[str, pd.DataFrame], horizon: int) -> np.ndarray:
    """The unconditional forward-return base rate: every bar's forward H return.

    The honest benchmark — what a no-signal, always-long trader earns on the same
    names/window. Matching-low events are compared against this pool directly (there is
    no short leg to negate — the pattern is a long-only claim).
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


def placebo_pvalue(panel: dict[str, pd.DataFrame], horizon: int, n_events: int,
                   obs_mean: float, n_draws: int = 5000, seed: int = 694) -> dict:
    """Random-draw placebo: ``n_events`` random bars from the unconditional pool vs the
    observed mean.

    Draw ``n_events`` random bars (with replacement) from the same basket's unconditional
    forward-return pool, ``n_draws`` times, and ask how often the random mean >= the
    observed matching-low mean. The honest "could a random pick of the same many trades
    look this good?" test — the pattern is long-only, so unlike the desk's mixed-
    direction studies there is no long/short mix to match.
    """
    base = unconditional_base(panel, horizon)
    if base.size == 0 or n_events <= 0 or not np.isfinite(obs_mean):
        return {"p_value": float("nan"), "draws": np.array([]), "obs": obs_mean}
    rng = np.random.default_rng(seed)
    means = np.empty(n_draws)
    for i in range(n_draws):
        pick = rng.choice(base, size=n_events, replace=True)
        means[i] = float(pick.mean())
    p = float((means >= obs_mean).mean())
    return {"obs": obs_mean, "p_value": p, "draws": means}


# --------------------------------------------------------------------------- #
# Costs
# --------------------------------------------------------------------------- #
def event_net_returns(ev: pd.DataFrame, cost_bps: float = 5.0) -> np.ndarray:
    """Per-event return net of a one-way round trip x NAV (long only, no borrow).

    Every event pays a fresh round trip: ``2 * cost_bps`` (in + out) x NAV, one-way each
    leg.
    """
    c = cost_bps / 1e4
    round_trip = 2.0 * c
    return ev["ret"].to_numpy(float) - round_trip


# --------------------------------------------------------------------------- #
# Bonferroni across the basket — one Welch t per ticker
# --------------------------------------------------------------------------- #
def per_ticker_stats(panel: dict[str, pd.DataFrame], horizon: int, tol: float = DEFAULT_TOL,
                     min_events: int = 6) -> pd.DataFrame:
    """One row per ticker with >= ``min_events`` matching-low events: n, mean, HAC t,
    Welch t.

    ``t_hac`` is the one-sample HAC t of the forward-return mean against **zero** —
    informational only, because names carry their own idiosyncratic drift, which biases a
    vs-zero test. ``t_welch`` compares the event sample against **that ticker's own**
    unconditional pool (drift-neutral) and is the number the Bonferroni correction is
    applied to: with ``k`` tickers tested independently, the per-ticker two-sided
    significance bar becomes ``|t| >= z(1 - 0.025/k)`` instead of the usual 1.96 — so a
    single lucky name in a wide basket can't pass as *the* signal.
    """
    rows = []
    for tk, bars in panel.items():
        if bars is None or len(bars) < horizon + 15:
            continue
        sig = matching_low_signals(bars, tol=tol).to_numpy()
        fwd = forward_returns(bars, horizon)
        mask = sig & np.isfinite(fwd)
        if mask.sum() < min_events:
            continue
        sr = fwd[mask]
        own_base = fwd[np.isfinite(fwd)]
        rows.append({"ticker": tk, "n": int(mask.sum()), "mean": float(sr.mean()),
                     "t_hac": hac_t(sr), "t_welch": welch_t(sr, own_base)})
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
              placebo: bool = True, cost_bps: float = 5.0, tol: float = DEFAULT_TOL,
              require_downtrend: bool = False, trend_lookback: int = 10) -> dict:
    """Headline stats for one horizon: n, mean, base, hit rate, HAC t, Welch t, placebo p,
    net."""
    ev = collect_events(panel, horizon, tol=tol, require_downtrend=require_downtrend,
                        trend_lookback=trend_lookback)
    base = unconditional_base(panel, horizon)
    empty = {"horizon": horizon, "n_events": 0, "mean": float("nan"),
             "base_mean": float("nan"), "hit": float("nan"), "base_hit": float("nan"),
             "hit_lo": float("nan"), "hit_hi": float("nan"), "t_hac": float("nan"),
             "t_one": float("nan"), "t_welch": float("nan"), "p_placebo": float("nan"),
             "net": float("nan")}
    if len(ev) == 0:
        return empty
    sr = ev["ret"].to_numpy(float)
    mean = float(sr.mean())
    base_mean = float(base.mean()) if base.size else float("nan")
    hit_k = int((sr > 0).sum())
    hit = hit_k / len(sr)
    hit_lo, hit_hi = wilson_interval(hit_k, len(sr))
    base_hit = float((base > 0).mean()) if base.size else float("nan")
    t_hac = hac_t(sr)
    t_one = onesample_t(sr)
    t_welch = welch_t(sr, base)
    p = (placebo_pvalue(panel, horizon, len(ev), mean, n_draws=n_draws)["p_value"]
         if placebo else float("nan"))
    net = float(event_net_returns(ev, cost_bps=cost_bps).mean())

    return {
        "horizon": horizon, "n_events": int(len(ev)), "mean": mean, "base_mean": base_mean,
        "hit": hit, "base_hit": base_hit, "hit_lo": hit_lo, "hit_hi": hit_hi,
        "t_hac": t_hac, "t_one": t_one, "t_welch": t_welch, "p_placebo": p, "net": net,
    }


def run_experiment(panel: dict[str, pd.DataFrame], horizons=HORIZONS, n_draws: int = 5000,
                   cost_bps: float = 5.0, tol: float = DEFAULT_TOL,
                   require_downtrend: bool = False) -> pd.DataFrame:
    """Run the full event study across horizons -> a tidy results frame ("the timer")."""
    rows = [summarize(panel, h, n_draws=n_draws, cost_bps=cost_bps, tol=tol,
                      require_downtrend=require_downtrend) for h in horizons]
    return pd.DataFrame(rows)
