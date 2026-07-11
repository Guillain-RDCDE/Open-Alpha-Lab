"""The breakaway detector, the event study, and its honest controls — Study 692.

The **breakaway** (Nison, *Japanese Candlestick Charting Techniques*) is a five-bar
reversal figure: candle 1 is a long-bodied candle continuing the prevailing trend;
candle 2 **gaps away** from it in the same direction (a genuine window that stays open);
candles 3-4 **run** further in that direction; candle 5 is a long-bodied candle in the
*opposite* direction that closes back **through the gap** ("closes within the gap area"
in Nison's own words — it erases the last leg of the run). The folk recipe: spot the
gap-then-run-then-reversal shape, trade the reversal, because the interrupted trend is
over. Unlike a single-direction figure, the breakaway is claimed **both ways**:

* **Bullish breakaway** — a downtrend, a long bearish candle 1, a gap **down**, two
  candles running further down, then a long bullish candle 5 closing back up through the
  gap: **buy** the reversal.
* **Bearish breakaway** — the exact mirror in an uptrend: **short** the reversal.

Chart figures are partly in the eye of the beholder, so — following the desk's dedup
siblings (417-island-reversal, 687-ladder-bottom) — we test the closest **mechanical**
definition we can write down and say so:

* **Loose cut.** (1) a genuine downtrend/uptrend context into the block; (2) candle 1
  bearish/bullish; (3) candle 2 gaps cleanly (``gap_pct``) beyond candle 1's low/high,
  with the gap staying fully open on candle 2's own bar; (4) candles 2→3→4 make
  successively lower/higher closes (the "run"); (5) candle 5 is bullish/bearish; (6)
  candle 5's close crosses back through candle 2's high/low (the gap window) — the
  reversal is "through the gap", not just a same-direction bounce.
* **Strict, literature-closer cut.** The loose shape plus (a) a **larger** minimum gap
  (``STRICT_GAP_PCT``), (b) candles 1 and 5 are genuinely **long-bodied** (body ≥
  ``LONG_BODY_FRAC`` of that bar's own range — Nison's own emphasis on candles 1 and 5
  being "long"), and (c) candle 5 **fully closes the gap** — its close crosses back
  past candle 1's own low/high, not merely candle 2's.

* **The base rate.** Every reversal trade is compared against the *unconditional* base
  rate: the same directional bet on **every** bar that also sits in a matching
  downtrend/uptrend context, whether or not the specific five-candle shape fired. This
  isolates "does the breakaway shape add information beyond simply being in a trend"
  from plain mean reversion / trend continuation drift.

* **The honest sample-size rule.** Below :data:`MIN_N_FOR_TEST` pooled events we do
  **not** compute a *t*-statistic at all — a *t* on a handful of observations is
  decoration, not evidence. Where a test *does* run, four horizons means a
  **Bonferroni**-corrected critical value (``k=4``), not a naive |t| >= 2.

* **Arbiters.** A Welch *t* of the breakaway-reversal mean vs the base-rate mean (the
  decisive statistic — never a one-sample *t* against zero, which would just measure the
  basket's unconditional drift); a one-sample HAC (Newey-West) *t* where n permits; a
  label-shuffle placebo; costs (one-way bps round trip, no borrow beyond the short leg's
  own book). Bullish and bearish events are pooled into one **combined** headline test
  (both trade directions are already sign-adjusted to "trade P&L", so they combine on
  like-for-like units); the two sides are also reported **separately** as the desk's
  own myth-check — a real reversal figure should not work on only one side.

No look-ahead: the breakaway and its context are known at the close of bar *t* (the
fifth candle); the trade is entered at *t+1*'s open and held a fixed horizon.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (1, 5, 10, 20)
MIN_N_FOR_TEST = 8              # below this, report raw numbers only -- no t-stat theatre
BONFERRONI_K = len(HORIZONS)    # four horizons -> four simultaneous looks

GAP_PCT = 0.005                 # loose cut: minimum clean gap (0.5%)
STRICT_GAP_PCT = 0.012          # strict cut: minimum clean gap (1.2%)
LONG_BODY_FRAC = 0.55           # strict cut: candles 1 and 5 must be this "long"


# ---------------------------------------------------------------------------
# Candle geometry helpers
# ---------------------------------------------------------------------------
def _body(bars: pd.DataFrame) -> pd.Series:
    """Signed body: close - open."""
    return bars["close"] - bars["open"]


def _range(bars: pd.DataFrame) -> pd.Series:
    """Full intrabar range: high - low."""
    return bars["high"] - bars["low"]


# ---------------------------------------------------------------------------
# Breakaway detectors — loose cut
# ---------------------------------------------------------------------------
def bullish_breakaway_flags(bars: pd.DataFrame, gap_pct: float = GAP_PCT,
                            lookback: int = 10) -> pd.Series:
    """Boolean Series: True at bar *t* iff bars *t-4..t* form a LOOSE bullish breakaway.

    Confirmed at the close of bar *t* (the fifth candle), reading back at *t-4..t-1*:

    1. **Downtrend context**: close[t-4] < close[t-4-``lookback``].
    2. **Candle 1** (t-4) bearish: closes below its open.
    3. **Candle 2** (t-3) gaps **down** cleanly: its high stays below
       candle 1's low * (1 - ``gap_pct``) — the window never overlaps.
    4. **The run** (t-3 -> t-2 -> t-1): strictly descending closes.
    5. **Candle 5** (t) bullish: closes above its open.
    6. **Reversal through the gap**: close[t] > high[t-3] — candle 5 closes back above
       the gap-day's own high, crossing back through the window.
    """
    c, o, h, l = bars["close"], bars["open"], bars["high"], bars["low"]

    c4, o4, l4 = c.shift(4), o.shift(4), l.shift(4)
    h3, c3 = h.shift(3), c.shift(3)
    c2 = c.shift(2)
    c1 = c.shift(1)

    ctx = c4 < c.shift(4 + lookback)
    bearish1 = c4 < o4
    gap_down = h3 < l4 * (1.0 - gap_pct)
    run_down = (c3 > c2) & (c2 > c1)
    bullish5 = c > o
    reversal = c > h3

    result = ctx & bearish1 & gap_down & run_down & bullish5 & reversal
    return result.fillna(False).rename("bullish_breakaway")


def bearish_breakaway_flags(bars: pd.DataFrame, gap_pct: float = GAP_PCT,
                            lookback: int = 10) -> pd.Series:
    """Boolean Series: the exact mirror of :func:`bullish_breakaway_flags` (uptrend,
    gap up, run up, long bearish reversal candle closing back down through the gap)."""
    c, o, h, l = bars["close"], bars["open"], bars["high"], bars["low"]

    c4, o4, h4 = c.shift(4), o.shift(4), h.shift(4)
    l3, c3 = l.shift(3), c.shift(3)
    c2 = c.shift(2)
    c1 = c.shift(1)

    ctx = c4 > c.shift(4 + lookback)
    bullish1 = c4 > o4
    gap_up = l3 > h4 * (1.0 + gap_pct)
    run_up = (c3 < c2) & (c2 < c1)
    bearish5 = c < o
    reversal = c < l3

    result = ctx & bullish1 & gap_up & run_up & bearish5 & reversal
    return result.fillna(False).rename("bearish_breakaway")


# ---------------------------------------------------------------------------
# Breakaway detectors — strict, literature-closer cut
# ---------------------------------------------------------------------------
def strict_bullish_breakaway_flags(bars: pd.DataFrame, gap_pct: float = STRICT_GAP_PCT,
                                   lookback: int = 10,
                                   long_body_frac: float = LONG_BODY_FRAC) -> pd.Series:
    """The loose bullish cut plus: a bigger gap, genuinely long candles 1 & 5, and a
    full gap fill (candle 5 closes back past candle 1's OWN low, not just candle 2's)."""
    loose = bullish_breakaway_flags(bars, gap_pct=gap_pct, lookback=lookback)

    c, l4 = bars["close"], bars["low"].shift(4)
    rng = _range(bars).clip(lower=1e-12)
    body = _body(bars).abs()

    long1 = body.shift(4) >= long_body_frac * rng.shift(4)
    long5 = body >= long_body_frac * rng
    full_fill = c > l4

    result = loose & long1 & long5 & full_fill
    return result.fillna(False).rename("strict_bullish_breakaway")


def strict_bearish_breakaway_flags(bars: pd.DataFrame, gap_pct: float = STRICT_GAP_PCT,
                                   lookback: int = 10,
                                   long_body_frac: float = LONG_BODY_FRAC) -> pd.Series:
    """The mirror of :func:`strict_bullish_breakaway_flags`."""
    loose = bearish_breakaway_flags(bars, gap_pct=gap_pct, lookback=lookback)

    c, h4 = bars["close"], bars["high"].shift(4)
    rng = _range(bars).clip(lower=1e-12)
    body = _body(bars).abs()

    long1 = body.shift(4) >= long_body_frac * rng.shift(4)
    long5 = body >= long_body_frac * rng
    full_fill = c < h4

    result = loose & long1 & long5 & full_fill
    return result.fillna(False).rename("strict_bearish_breakaway")


_FLAGS = {"bullish": (bullish_breakaway_flags, strict_bullish_breakaway_flags, 1),
          "bearish": (bearish_breakaway_flags, strict_bearish_breakaway_flags, -1)}


# ---------------------------------------------------------------------------
# Event table -- one row per (ticker, breakaway block)
# ---------------------------------------------------------------------------
def breakaway_events(bars: pd.DataFrame, side: str, gap_pct: float = GAP_PCT,
                     lookback: int = 10, horizons=HORIZONS) -> pd.DataFrame:
    """One row per breakaway event with the forward (direction-signed) returns and the
    strict flag.

    For each confirming bar *i* (the fifth candle): enter the next session's **open**
    (one execution lag, no look-ahead) and hold horizon ``h`` to the close ``h`` sessions
    later (``h=1`` = the entry day's own close). ``side="bearish"`` returns are
    sign-flipped so every number in this frame is the trade's own P&L (long for bullish,
    short for bearish) -- directly comparable and poolable across sides.
    """
    if side not in _FLAGS:
        raise ValueError(f"side must be 'bullish' or 'bearish', got {side!r}")
    loose_fn, strict_fn, direction = _FLAGS[side]

    o = bars["open"].to_numpy(dtype=float)
    c = bars["close"].to_numpy(dtype=float)
    n = len(bars)
    loose = loose_fn(bars, gap_pct=gap_pct, lookback=lookback).to_numpy()
    strict = strict_fn(bars, lookback=lookback).to_numpy()
    hmax = max(horizons)

    rows = []
    for i in range(lookback + 4, n - hmax - 1):
        if not loose[i]:
            continue
        e = i + 1
        entry = o[e]
        if not np.isfinite(entry) or entry <= 0:
            continue
        row = {"pos": i, "strict": bool(strict[i])}
        for h in horizons:
            x = e + h - 1
            row[f"ret_{h}"] = float(direction * (c[x] / entry - 1.0))
        rows.append(row)
    return pd.DataFrame(rows)


def base_rate_events(bars: pd.DataFrame, side: str, lookback: int = 10,
                     horizons=HORIZONS, sample: int | None = None,
                     seed: int = 692) -> pd.DataFrame:
    """The unconditional base rate: the same directional bet on every bar with a
    matching trend context, regardless of whether the specific breakaway shape fired.
    "What does the same buy-the-downtrend (or short-the-uptrend) bet earn on a random bar
    already sitting in that trend?" -- isolates the pattern's own information from plain
    trend-context mean reversion / continuation.
    """
    if side not in _FLAGS:
        raise ValueError(f"side must be 'bullish' or 'bearish', got {side!r}")
    direction = 1 if side == "bullish" else -1

    c = bars["close"].to_numpy(dtype=float)
    o = bars["open"].to_numpy(dtype=float)
    n = len(bars)
    hmax = max(horizons)
    rows = []
    for i in range(lookback + 4, n - hmax - 1):
        if c[i - 4 - lookback] <= 0:
            continue
        ctx = (c[i - 4] < c[i - 4 - lookback]) if side == "bullish" else \
              (c[i - 4] > c[i - 4 - lookback])
        if not ctx:
            continue
        e = i + 1
        entry = o[e]
        if not np.isfinite(entry) or entry <= 0:
            continue
        row = {"pos": i}
        for h in horizons:
            x = e + h - 1
            row[f"ret_{h}"] = float(direction * (c[x] / entry - 1.0))
        rows.append(row)
    out = pd.DataFrame(rows)
    if sample is not None and len(out) > sample:
        out = out.sample(n=sample, random_state=seed).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Pooling across the basket
# ---------------------------------------------------------------------------
def pool_events(panel: dict[str, pd.DataFrame], side: str, lookback: int = 10,
                horizons=HORIZONS) -> pd.DataFrame:
    """Concatenate :func:`breakaway_events` across every ticker in ``panel``."""
    frames = []
    for t, bars in panel.items():
        ev = breakaway_events(bars, side, lookback=lookback, horizons=horizons)
        if len(ev):
            ev = ev.copy()
            ev["ticker"] = t
            frames.append(ev)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def pool_base_rate(panel: dict[str, pd.DataFrame], side: str, horizons=HORIZONS,
                   lookback: int = 10, sample_per: int | None = None,
                   seed: int = 692) -> pd.DataFrame:
    frames = []
    for t, bars in panel.items():
        br = base_rate_events(bars, side, horizons=horizons, lookback=lookback,
                              sample=sample_per, seed=seed)
        if len(br):
            br = br.copy()
            br["ticker"] = t
            frames.append(br)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------
def hac_tstat(x: np.ndarray) -> float:
    """One-sample Newey-West HAC *t*-statistic on the mean of ``x`` against zero."""
    r = np.asarray(x, dtype=float)
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


def welch_t(a: np.ndarray, b: np.ndarray) -> float | None:
    """Welch *t* of mean(a) - mean(b) (unequal variances) -- the decisive statistic for
    "does the breakaway reversal mean differ from the trend-matched base-rate mean?"
    ``None`` (not computed) if either side has fewer than :data:`MIN_N_FOR_TEST` finite
    observations -- the same "no timer if too few" discipline as :func:`summarize`.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < MIN_N_FOR_TEST or len(b) < MIN_N_FOR_TEST:
        return None
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else None


def bonferroni_critical(k: int = BONFERRONI_K, alpha: float = 0.05) -> float:
    """Two-sided Bonferroni-corrected |t| critical value for ``k`` simultaneous tests."""
    from scipy import stats

    return float(stats.norm.ppf(1.0 - alpha / (2.0 * k)))


def summarize(returns: np.ndarray, cost_bps: float = 5.0,
             min_n: int = MIN_N_FOR_TEST) -> dict:
    """Headline stats for a vector of (direction-signed) breakaway reversal returns.

    ``cost_bps`` is a one-way cost per leg; a round trip (entry + exit) deducts
    ``2 * cost_bps`` (shorts are already sign-flipped into "trade P&L", so the same cost
    convention applies to both directions -- no separate borrow line, in line with the
    desk's single-lag/one-way-cost convention for a directional flip-book). Below
    ``min_n`` observations the *t*-stat is deliberately **not computed** (``tstat=None``,
    ``tested=False``) -- a *t* on a handful of points is theatre, not evidence.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n == 0:
        return {"n": 0, "win_rate": float("nan"), "mean_bps": float("nan"),
                "net_bps": float("nan"), "tstat": None, "net_t": None, "tested": False}
    net = r - 2.0 * cost_bps * 1e-4
    tested = n >= min_n
    return {
        "n": int(n),
        "win_rate": float((r > 0).mean()),
        "mean_bps": float(r.mean() * 1e4),
        "sharpe": float(r.mean() / r.std(ddof=1)) if n > 1 and r.std() > 0 else float("nan"),
        "tstat": hac_tstat(r) if tested else None,
        "net_bps": float(net.mean() * 1e4),
        "net_t": hac_tstat(net) if tested else None,
        "tested": tested,
    }


# ---------------------------------------------------------------------------
# Orchestrators
# ---------------------------------------------------------------------------
def _placebo_p(rng_master: np.random.Generator, pool_vals: np.ndarray, n_events: int,
              real_mean: float, n_draws: int) -> tuple[float, float]:
    """Draw ``n_events`` fake events from ``pool_vals`` ``n_draws`` times; return
    ``(placebo_mean, p)`` where ``p`` = share of placebo means >= the observed mean."""
    if pool_vals.size < n_events or n_events < 1 or not np.isfinite(real_mean):
        return float("nan"), float("nan")
    seed_h = int(rng_master.integers(0, 1 << 31))
    r2 = np.random.default_rng(seed_h)
    placebo = np.array([r2.choice(pool_vals, size=n_events, replace=False).mean()
                        for _ in range(n_draws)])
    return float(placebo.mean()), float((placebo >= real_mean).mean())


def run_experiment(panel: dict[str, pd.DataFrame], side: str, lookback: int = 10,
                   horizons=HORIZONS, cost_bps: float = 5.0, n_draws: int = 2000,
                   seed: int = 692) -> dict:
    """Run the full loose-vs-strict-vs-base-rate teardown for ONE side (bullish or
    bearish) on a panel. Same statistical idiom as :func:`combined_experiment`, applied
    to a single trade direction -- this is the desk's own myth-check: does the figure
    work on both sides, or only one?
    """
    ev = pool_events(panel, side, lookback=lookback, horizons=horizons)
    n_loose = len(ev)
    n_strict = int(ev["strict"].sum()) if n_loose else 0
    out = {"side": side, "n_loose": int(n_loose), "n_strict": n_strict,
           "horizons": list(horizons), "bonferroni_crit": bonferroni_critical(len(horizons)),
           "per_horizon": {}, "strict_per_horizon": {}}

    strict_ev = ev[ev["strict"]] if n_loose else ev
    br_all = pool_base_rate(panel, side, horizons=horizons, lookback=lookback, seed=seed)
    rng_master = np.random.default_rng(seed)
    for h in horizons:
        rev = ev[f"ret_{h}"].to_numpy(dtype=float) if n_loose else np.array([])
        s = summarize(rev, cost_bps=cost_bps)
        srev = strict_ev[f"ret_{h}"].to_numpy(dtype=float) if len(strict_ev) else np.array([])
        ss = summarize(srev, cost_bps=cost_bps)

        base = br_all[f"ret_{h}"].to_numpy(dtype=float) if len(br_all) else np.array([])
        b = summarize(base, cost_bps=0.0)
        delta = (s["mean_bps"] - b["mean_bps"]
                 if np.isfinite(s["mean_bps"]) and np.isfinite(b["mean_bps"]) else float("nan"))
        welch = welch_t(rev, base)
        strict_delta = (ss["mean_bps"] - b["mean_bps"]
                        if np.isfinite(ss["mean_bps"]) and np.isfinite(b["mean_bps"])
                        else float("nan"))
        strict_welch = welch_t(srev, base)

        pool_vals = base[np.isfinite(base)]
        real_mean = float(np.nanmean(rev)) if rev.size else float("nan")
        placebo_mean, p = _placebo_p(rng_master, pool_vals, n_loose, real_mean, n_draws)
        strict_real_mean = float(np.nanmean(srev)) if srev.size else float("nan")
        n_strict_h = int(np.isfinite(srev).sum()) if srev.size else 0
        _, strict_p = _placebo_p(rng_master, pool_vals, n_strict_h, strict_real_mean, n_draws)

        out["per_horizon"][h] = {"ladder": s, "base": b, "delta_bps": float(delta),
                                 "welch_t": welch, "placebo_p": p,
                                 "placebo_mean_bps": placebo_mean * 1e4
                                 if np.isfinite(placebo_mean) else float("nan")}
        out["strict_per_horizon"][h] = {"ladder": ss, "delta_bps": float(strict_delta),
                                        "welch_t": strict_welch, "placebo_p": strict_p}
    return out


def combined_experiment(panel: dict[str, pd.DataFrame], lookback: int = 10,
                        horizons=HORIZONS, cost_bps: float = 5.0, n_draws: int = 2000,
                        seed: int = 692) -> dict:
    """The HEADLINE test: bullish + bearish breakaway events pooled into one sample.

    Both sides' returns and base rates are already direction-signed to "trade P&L", so
    pooling them tests the desk's actual claim -- "the breakaway figure marks a
    reversal", not "up markets drift up". Same Welch-vs-base-rate / HAC / Bonferroni /
    label-shuffle-placebo machinery as :func:`run_experiment`, run on the union.
    """
    ev_b = pool_events(panel, "bullish", lookback=lookback, horizons=horizons)
    ev_r = pool_events(panel, "bearish", lookback=lookback, horizons=horizons)
    parts = [e for e in (ev_b, ev_r) if len(e)]
    ev = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    n_loose = len(ev)
    n_strict = int(ev["strict"].sum()) if n_loose else 0
    out = {"n_loose": int(n_loose), "n_strict": n_strict, "n_bullish": int(len(ev_b)),
           "n_bearish": int(len(ev_r)), "horizons": list(horizons),
           "bonferroni_crit": bonferroni_critical(len(horizons)),
           "per_horizon": {}, "strict_per_horizon": {}}

    spans = [(b.index[-1] - b.index[0]).days / 365.25 for b in panel.values() if len(b) > 1]
    out["years"] = float(max(spans)) if spans else float("nan")
    out["n_names"] = len(panel)

    strict_ev = ev[ev["strict"]] if n_loose else ev
    base_b_all = pool_base_rate(panel, "bullish", horizons=horizons, lookback=lookback, seed=seed)
    base_r_all = pool_base_rate(panel, "bearish", horizons=horizons, lookback=lookback, seed=seed)
    base_all_parts = [b for b in (base_b_all, base_r_all) if len(b)]
    base_all = pd.concat(base_all_parts, ignore_index=True) if base_all_parts else pd.DataFrame()
    rng_master = np.random.default_rng(seed)
    for h in horizons:
        rev = ev[f"ret_{h}"].to_numpy(dtype=float) if n_loose else np.array([])
        s = summarize(rev, cost_bps=cost_bps)
        srev = strict_ev[f"ret_{h}"].to_numpy(dtype=float) if len(strict_ev) else np.array([])
        ss = summarize(srev, cost_bps=cost_bps)

        base = base_all[f"ret_{h}"].to_numpy(dtype=float) if len(base_all) else np.array([])
        b = summarize(base, cost_bps=0.0)
        delta = (s["mean_bps"] - b["mean_bps"]
                 if np.isfinite(s["mean_bps"]) and np.isfinite(b["mean_bps"]) else float("nan"))
        welch = welch_t(rev, base)
        strict_delta = (ss["mean_bps"] - b["mean_bps"]
                        if np.isfinite(ss["mean_bps"]) and np.isfinite(b["mean_bps"])
                        else float("nan"))
        strict_welch = welch_t(srev, base)

        pool_vals = base[np.isfinite(base)]
        real_mean = float(np.nanmean(rev)) if rev.size else float("nan")
        placebo_mean, p = _placebo_p(rng_master, pool_vals, n_loose, real_mean, n_draws)
        strict_real_mean = float(np.nanmean(srev)) if srev.size else float("nan")
        n_strict_h = int(np.isfinite(srev).sum()) if srev.size else 0
        _, strict_p = _placebo_p(rng_master, pool_vals, n_strict_h, strict_real_mean, n_draws)

        out["per_horizon"][h] = {"ladder": s, "base": b, "delta_bps": float(delta),
                                 "welch_t": welch, "placebo_p": p,
                                 "placebo_mean_bps": placebo_mean * 1e4
                                 if np.isfinite(placebo_mean) else float("nan")}
        out["strict_per_horizon"][h] = {"ladder": ss, "delta_bps": float(strict_delta),
                                        "welch_t": strict_welch, "placebo_p": strict_p}
    return out


# ---------------------------------------------------------------------------
# Synthetic-control detector (the machinery proof)
# ---------------------------------------------------------------------------
def synthetic_detect(data: dict[str, pd.DataFrame], side: str, horizon: int = 20,
                     lookback: int = 10, seed: int = 692) -> dict:
    """Run the headline breakaway-vs-base-rate split on a synthetic panel, one side.

    The DECISIVE statistic is the **Welch t of breakaway-reversal mean vs the base-rate
    reversal mean** (``welch_t``) -- the same delta the real-tape headline is graded on.
    """
    ev = pool_events(data, side, lookback=lookback, horizons=(horizon,))
    br = pool_base_rate(data, side, horizons=(horizon,), lookback=lookback, seed=seed)
    n = len(ev)
    rev = ev[f"ret_{horizon}"].to_numpy(dtype=float) if n else np.array([])
    base = br[f"ret_{horizon}"].to_numpy(dtype=float) if len(br) else np.array([])
    mean_bps = float(np.nanmean(rev) * 1e4) if n else float("nan")
    base_bps = float(np.nanmean(base) * 1e4) if len(base) else float("nan")
    return {"n": n, "mean_bps": mean_bps, "base_bps": base_bps,
            "delta_bps": mean_bps - base_bps if n and len(base) else float("nan"),
            "tstat": hac_tstat(rev) if n >= MIN_N_FOR_TEST else None,
            "welch_t": welch_t(rev, base)}


def synthetic_detect_combined(data: dict[str, pd.DataFrame], horizon: int = 20,
                              lookback: int = 10, seed: int = 692) -> dict:
    """Combined bullish+bearish synthetic detector -- the machinery proof for the
    headline pooled statistic exactly as :func:`combined_experiment` computes it."""
    ev_b = pool_events(data, "bullish", lookback=lookback, horizons=(horizon,))
    ev_r = pool_events(data, "bearish", lookback=lookback, horizons=(horizon,))
    parts = [e for e in (ev_b, ev_r) if len(e)]
    ev = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    base_b = pool_base_rate(data, "bullish", horizons=(horizon,), lookback=lookback, seed=seed)
    base_r = pool_base_rate(data, "bearish", horizons=(horizon,), lookback=lookback, seed=seed)
    base_parts = [b for b in (base_b, base_r) if len(b)]
    base_df = pd.concat(base_parts, ignore_index=True) if base_parts else pd.DataFrame()
    n = len(ev)
    rev = ev[f"ret_{horizon}"].to_numpy(dtype=float) if n else np.array([])
    base = base_df[f"ret_{horizon}"].to_numpy(dtype=float) if len(base_df) else np.array([])
    mean_bps = float(np.nanmean(rev) * 1e4) if n else float("nan")
    base_bps = float(np.nanmean(base) * 1e4) if len(base) else float("nan")
    return {"n": n, "mean_bps": mean_bps, "base_bps": base_bps,
            "delta_bps": mean_bps - base_bps if n and len(base) else float("nan"),
            "tstat": hac_tstat(rev) if n >= MIN_N_FOR_TEST else None,
            "welch_t": welch_t(rev, base)}
