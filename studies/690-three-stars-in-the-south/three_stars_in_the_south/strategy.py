"""The three-stars-in-the-south detector, the event study, and its honest controls —
Study 690.

**Three stars in the south** (Nison, *Japanese Candlestick Charting Techniques*) is a
three-bar bullish reversal claimed to mark the *exhaustion* of a downtrend: three
consecutive **black** (bearish) candles print, each with a visibly **smaller body/range**
than the one before it, and each with a **higher low** than the one before it — sellers
still pushing, but with less and less conviction, until the third candle is a small
near-marubozu that doesn't even test the second candle's low. The folk recipe: spot the
shrinking, rising-low sequence, and the sell pressure has run out — buy.

What we test, announced before we run it:

* **Detector, two cuts.** A **loose** cut — three consecutive bearish candles with
  strictly **shrinking ranges** and strictly **rising lows**, preceded by a genuine
  downtrend context — is the plain reading a practical chartist would code up. A
  **strict, literature-closer** cut additionally requires (a) the first candle to carry
  a real lower shadow (selling met by some buying, a hammer-like first star), (b) the
  second candle to open **inside** the first candle's real body (no gap down to start
  the sequence), and (c) the third candle to be a **near-marubozu** (small shadows both
  sides — a decisively small, committed body) that **never breaks below** the second
  candle's low. The strict cut is the primary claim under test; the loose cut is the
  honest power/robustness check.

* **The reversal event.** Unlike a bidirectional pattern, three stars in the south is
  claimed in one direction only: **long**. We tag every candidate with the sign of the
  ``prior_lookback``-day move into the block (it must be a genuine downtrend — the
  pattern's own premise) and take the reversal trade **long** at the next session's open
  (one documented execution lag), reading forward 1/5/10/20-day returns.

* **The base rate.** The reversal return is compared against the *unconditional* base
  rate — the same long bet on **every** bar that also sits in a matching downtrend
  context, whether or not the specific three-star shape fired. This isolates "does the
  shrinking/rising-low shape add information beyond simply being in a downtrend" from
  plain mean reversion.

* **The honest sample-size rule.** Three stars in the south needs three independent,
  low-probability shape conditions lined up in a row; pooled across a broad basket the
  count can still be small. Below :data:`MIN_N_FOR_TEST` pooled events we do **not**
  compute a *t*-statistic at all — a *t* on a handful of observations is decoration, not
  evidence — and say plainly "too few to test." Where a test *does* run, four horizons
  means a **Bonferroni**-corrected critical value (``k=4``), not a naive |t| >= 2.

* **Arbiters.** A Welch *t* of the three-stars reversal mean vs the base-rate mean (the
  decisive statistic — never a one-sample *t* against zero, which would just measure the
  basket's unconditional drift); a one-sample HAC (Newey-West) *t* where n permits; a
  label-shuffle placebo; costs (one-way bps round trip, long-only, no borrow).

No look-ahead: the three-star shape is known at the close of bar *t* (the third star);
the trade is entered at *t+1*'s open and held a fixed horizon.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (1, 5, 10, 20)
MIN_N_FOR_TEST = 8            # below this, report raw numbers only -- no t-stat theatre
BONFERRONI_K = len(HORIZONS)  # four horizons -> four simultaneous looks


# ---------------------------------------------------------------------------
# Candle geometry helpers
# ---------------------------------------------------------------------------
def _body(bars: pd.DataFrame) -> pd.Series:
    """Signed body: close - open."""
    return bars["close"] - bars["open"]


def _range(bars: pd.DataFrame) -> pd.Series:
    """Full intrabar range: high - low."""
    return bars["high"] - bars["low"]


def _upper_shadow(bars: pd.DataFrame) -> pd.Series:
    """Upper shadow: high - max(open, close)."""
    return bars["high"] - bars[["open", "close"]].max(axis=1)


def _lower_shadow(bars: pd.DataFrame) -> pd.Series:
    """Lower shadow: min(open, close) - low."""
    return bars[["open", "close"]].min(axis=1) - bars["low"]


# ---------------------------------------------------------------------------
# Three-stars-in-the-south detector
# ---------------------------------------------------------------------------
def three_stars_flags(bars: pd.DataFrame, prior_lookback: int = 10) -> pd.Series:
    """Boolean Series: True at bar *t* iff bars *t-2..t* form a LOOSE three-stars block.

    Confirmed at the close of bar *t* (the third "star"), looking back at *t-2..t*:

    1. **Downtrend context**: close[t-2] < close[t-2-``prior_lookback``] — the pattern's
       own premise, a genuine downtrend leading into the first star.
    2. **Three bearish stars**, t-2..t: each closes below its open.
    3. **Shrinking ranges**: range[t-2] > range[t-1] > range[t] — each star's intrabar
       range is strictly smaller than the one before (selling with less conviction).
    4. **Rising lows**: low[t-2] < low[t-1] < low[t] — each star's low sits strictly
       higher than the previous star's low.
    """
    c, o = bars["close"], bars["open"]
    rng = _range(bars)
    lo = bars["low"]

    close_r2 = c.shift(2)
    ctx = close_r2 < c.shift(2 + prior_lookback)

    bearish_2 = c.shift(2) < o.shift(2)
    bearish_1 = c.shift(1) < o.shift(1)
    bearish_0 = c < o

    shrinking = (rng.shift(2) > rng.shift(1)) & (rng.shift(1) > rng)
    rising_lows = (lo.shift(2) < lo.shift(1)) & (lo.shift(1) < lo)

    result = ctx & bearish_2 & bearish_1 & bearish_0 & shrinking & rising_lows
    return result.fillna(False).rename("three_stars")


def strict_three_stars_flags(bars: pd.DataFrame, prior_lookback: int = 10,
                             lower_shadow_frac: float = 0.20,
                             marubozu_shadow_frac: float = 0.20) -> pd.Series:
    """Boolean Series: the STRICT, literature-closer three-stars-in-the-south (the loose
    cut plus):

    5. The **first** star (t-2) shows a real lower shadow (>= ``lower_shadow_frac`` of
       its range) — a hammer-like first candle, selling met by some intrabar buying.
    6. The **second** star (t-1) opens **inside** the first star's real body — no gap
       down to start the sequence.
    7. The **third** star (t) is a **near-marubozu**: both shadows small
       (<= ``marubozu_shadow_frac`` of its range) — a decisively small, committed body.
    8. The third star **never breaks below** the second star's low: low[t] >= low[t-1]
       (already implied by rising lows, restated here as the literature's explicit "no
       new low" condition).
    """
    loose = three_stars_flags(bars, prior_lookback=prior_lookback)

    rng = _range(bars).clip(lower=1e-12)
    lower_sh = _lower_shadow(bars)
    upper_sh = _upper_shadow(bars)
    o, c = bars["open"], bars["close"]

    body_lo_2 = bars[["open", "close"]].min(axis=1).shift(2)
    body_hi_2 = bars[["open", "close"]].max(axis=1).shift(2)

    hammer_first = lower_sh.shift(2) >= lower_shadow_frac * rng.shift(2)
    opens_inside = (o.shift(1) >= body_lo_2) & (o.shift(1) <= body_hi_2)
    marubozu_third = (upper_sh <= marubozu_shadow_frac * rng) & (lower_sh <= marubozu_shadow_frac * rng)
    no_new_low = bars["low"] >= bars["low"].shift(1)

    result = loose & hammer_first & opens_inside & marubozu_third & no_new_low
    return result.fillna(False).rename("strict_three_stars")


# ---------------------------------------------------------------------------
# Event table -- one row per (ticker, three-stars block)
# ---------------------------------------------------------------------------
def star_events(bars: pd.DataFrame, prior_lookback: int = 10,
                horizons=HORIZONS, active_mask: np.ndarray | None = None) -> pd.DataFrame:
    """One row per three-stars event with the forward LONG returns and the strict flag.

    For each confirming bar *i* (the third star): enter the next session's **open** (one
    execution lag, no look-ahead) and hold horizon ``h`` to the close ``h`` sessions
    later (``h=1`` = the entry day's own close). Bars without a full forward window are
    dropped.

    ``active_mask`` (synthetic tapes only, see :func:`base_rate_events`): skips an event
    whose forward window overlaps a *different*, unrelated forced block -- an event never
    overlaps its own block this way (its forward window starts the session after its own
    three forced days end).
    """
    o = bars["open"].to_numpy(dtype=float)
    c = bars["close"].to_numpy(dtype=float)
    n = len(bars)
    loose = three_stars_flags(bars, prior_lookback=prior_lookback).to_numpy()
    strict = strict_three_stars_flags(bars, prior_lookback=prior_lookback).to_numpy()
    hmax = max(horizons)

    rows = []
    for i in range(prior_lookback + 2, n - hmax - 1):
        if not loose[i]:
            continue
        e = i + 1
        if active_mask is not None and active_mask[e:e + hmax].any():
            continue
        entry = o[e]
        if not np.isfinite(entry) or entry <= 0:
            continue
        row = {"pos": i, "strict": bool(strict[i])}
        for h in horizons:
            x = e + h - 1
            row[f"ret_{h}"] = float(c[x] / entry - 1.0)
        rows.append(row)
    return pd.DataFrame(rows)


def base_rate_events(bars: pd.DataFrame, prior_lookback: int = 10,
                     horizons=HORIZONS, sample: int | None = None,
                     seed: int = 690, active_mask: np.ndarray | None = None) -> pd.DataFrame:
    """The unconditional base rate: the same LONG bet on every bar with a matching
    downtrend context (close[t-2] < close[t-2-prior_lookback]), regardless of whether the
    specific three-star shape fired. "What does the same buy-the-downtrend bet earn on a
    random bar in a downtrend?" -- isolates the pattern's own information from plain
    downtrend mean reversion.

    ``active_mask`` (synthetic tapes only): a boolean array, True on any session touched
    by an ENGINEERED planted decline (see :func:`data.synthetic_panel`). A base-rate
    candidate whose own forward-return window still overlaps somebody else's
    not-yet-finished planted decline is skipped -- otherwise it would see an artificially
    depressed forward return that has nothing to do with whether *it* is a three-stars
    pattern, biasing the synthetic null off zero. Real-tape callers never pass this
    (``None``, the default): nothing here is a hardcoded feature of a real candle.
    """
    c = bars["close"].to_numpy(dtype=float)
    o = bars["open"].to_numpy(dtype=float)
    n = len(bars)
    hmax = max(horizons)
    rows = []
    for i in range(prior_lookback + 2, n - hmax - 1):
        if c[i - 2 - prior_lookback] <= 0:
            continue
        if not (c[i - 2] < c[i - 2 - prior_lookback]):
            continue
        e = i + 1
        if active_mask is not None and active_mask[e:e + hmax].any():
            continue
        entry = o[e]
        if not np.isfinite(entry) or entry <= 0:
            continue
        row = {"pos": i}
        for h in horizons:
            x = e + h - 1
            row[f"ret_{h}"] = float(c[x] / entry - 1.0)
        rows.append(row)
    out = pd.DataFrame(rows)
    if sample is not None and len(out) > sample:
        out = out.sample(n=sample, random_state=seed).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Pooling across the basket
# ---------------------------------------------------------------------------
def pool_events(panel: dict[str, pd.DataFrame], prior_lookback: int = 10,
                horizons=HORIZONS,
                active_masks: dict[str, np.ndarray] | None = None) -> pd.DataFrame:
    """Concatenate :func:`star_events` across every ticker in ``panel``."""
    frames = []
    for t, bars in panel.items():
        mask = active_masks.get(t) if active_masks is not None else None
        ev = star_events(bars, prior_lookback=prior_lookback, horizons=horizons,
                         active_mask=mask)
        if len(ev):
            ev = ev.copy()
            ev["ticker"] = t
            frames.append(ev)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def pool_base_rate(panel: dict[str, pd.DataFrame], horizons=HORIZONS,
                   prior_lookback: int = 10, sample_per: int | None = None,
                   seed: int = 690,
                   active_masks: dict[str, np.ndarray] | None = None) -> pd.DataFrame:
    frames = []
    for t, bars in panel.items():
        mask = active_masks.get(t) if active_masks is not None else None
        br = base_rate_events(bars, horizons=horizons, prior_lookback=prior_lookback,
                              sample=sample_per, seed=seed, active_mask=mask)
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
    "does the three-stars reversal mean differ from the downtrend base-rate mean?"
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
    """Headline stats for a vector of (long) reversal returns.

    ``cost_bps`` is a one-way cost per leg; a round trip (entry + exit) deducts
    ``2 * cost_bps``. Long-only, no borrow. Below ``min_n`` observations the *t*-stat is
    deliberately **not computed** (``tstat=None``, ``tested=False``) -- a *t* on a
    handful of points is theatre, not evidence, and the honest thing is to say "too few
    to test" rather than print a decorative number.
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


def label_shuffle_placebo(panel: dict[str, pd.DataFrame], n_events: int, horizon: int = 20,
                          n_draws: int = 2000, prior_lookback: int = 10,
                          seed: int = 690) -> dict:
    """Placebo: re-draw ``n_events`` 'fake three-stars' bars at random from the pooled
    downtrend base-rate pool and recompute the mean. The returned ``p`` is the fraction of
    placebo means at least as large as the real three-stars mean -- a non-parametric
    null that respects "what does a same-size random set of downtrend bars give?"
    """
    br = pool_base_rate(panel, horizons=(horizon,), prior_lookback=prior_lookback, seed=seed)
    if not len(br):
        return {"placebo_means": np.array([]), "pool": np.array([])}
    col = f"ret_{horizon}"
    vals = br[col].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    rng = np.random.default_rng(seed)
    if len(vals) < n_events or n_events < 1:
        return {"placebo_means": np.array([]), "pool": vals}
    means = np.array([rng.choice(vals, size=n_events, replace=False).mean()
                      for _ in range(n_draws)])
    return {"placebo_means": means, "pool": vals}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_experiment(panel: dict[str, pd.DataFrame], prior_lookback: int = 10,
                   horizons=HORIZONS, cost_bps: float = 5.0,
                   n_draws: int = 2000, seed: int = 690) -> dict:
    """Run the full three-stars-in-the-south teardown on a panel and return a results
    dict.

    Computes, per horizon: the pooled loose-cut reversal stats (gross + net, tested only
    if n >= :data:`MIN_N_FOR_TEST`), the strict-cut stats, the direction-matched
    unconditional base rate, the star-minus-base delta, a pooled label-shuffle placebo
    p-value, and the Bonferroni-corrected critical |t| for ``len(horizons)`` simultaneous
    tests.
    """
    ev = pool_events(panel, prior_lookback=prior_lookback, horizons=horizons)
    n_loose = len(ev)
    n_strict = int(ev["strict"].sum()) if n_loose else 0
    out = {"n_loose": int(n_loose), "n_strict": n_strict, "horizons": list(horizons),
           "bonferroni_crit": bonferroni_critical(len(horizons)),
           "per_horizon": {}, "strict_per_horizon": {}}

    spans = [(b.index[-1] - b.index[0]).days / 365.25 for b in panel.values() if len(b) > 1]
    yrs = max(spans) if spans else float("nan")
    out["years"] = float(yrs)
    out["n_names"] = len(panel)

    strict_ev = ev[ev["strict"]] if n_loose else ev

    rng_master = np.random.default_rng(seed)
    for h in horizons:
        rev = ev[f"ret_{h}"].to_numpy(dtype=float) if n_loose else np.array([])
        s = summarize(rev, cost_bps=cost_bps)
        srev = strict_ev[f"ret_{h}"].to_numpy(dtype=float) if len(strict_ev) else np.array([])
        ss = summarize(srev, cost_bps=cost_bps)

        br_h = pool_base_rate(panel, horizons=(h,), prior_lookback=prior_lookback, seed=seed)
        base = br_h[f"ret_{h}"].to_numpy(dtype=float) if len(br_h) else np.array([])
        b = summarize(base, cost_bps=0.0)
        delta = (s["mean_bps"] - b["mean_bps"]
                 if np.isfinite(s["mean_bps"]) and np.isfinite(b["mean_bps"]) else float("nan"))
        welch = welch_t(rev, base)
        # strict cut vs the SAME base-rate pool -- the decisive stat is always vs base,
        # never a one-sample t against zero (that would just measure basket drift)
        strict_delta = (ss["mean_bps"] - b["mean_bps"]
                        if np.isfinite(ss["mean_bps"]) and np.isfinite(b["mean_bps"])
                        else float("nan"))
        strict_welch = welch_t(srev, base)

        pool_vals = base[np.isfinite(base)]
        real_mean = float(np.nanmean(rev)) if rev.size else float("nan")
        if pool_vals.size >= n_loose and n_loose >= 1 and np.isfinite(real_mean):
            seed_h = int(rng_master.integers(0, 1 << 31))
            r2 = np.random.default_rng(seed_h)
            placebo = np.array([r2.choice(pool_vals, size=n_loose, replace=False).mean()
                                for _ in range(n_draws)])
            p = float((placebo >= real_mean).mean())
        else:
            placebo = np.array([])
            p = float("nan")

        out["per_horizon"][h] = {
            "star": s, "base": b, "delta_bps": float(delta), "welch_t": welch,
            "placebo_p": p,
            "placebo_mean_bps": float(placebo.mean() * 1e4) if placebo.size else float("nan"),
        }

        # strict-cut placebo: same idea, drawn at the strict sample size
        n_strict_h = int(np.isfinite(srev).sum()) if srev.size else 0
        strict_real_mean = float(np.nanmean(srev)) if srev.size else float("nan")
        if pool_vals.size >= n_strict_h and n_strict_h >= 1 and np.isfinite(strict_real_mean):
            seed_hs = int(rng_master.integers(0, 1 << 31))
            r3 = np.random.default_rng(seed_hs)
            strict_placebo = np.array([r3.choice(pool_vals, size=n_strict_h, replace=False).mean()
                                       for _ in range(n_draws)])
            strict_p = float((strict_placebo >= strict_real_mean).mean())
        else:
            strict_p = float("nan")

        out["strict_per_horizon"][h] = {
            "star": ss, "delta_bps": float(strict_delta), "welch_t": strict_welch,
            "placebo_p": strict_p,
        }
    return out


# ---------------------------------------------------------------------------
# Synthetic-control detector (the machinery proof)
# ---------------------------------------------------------------------------
def synthetic_detect(data: dict[str, pd.DataFrame], horizon: int = 20,
                     prior_lookback: int = 10, seed: int = 690,
                     active_masks: dict[str, np.ndarray] | None = None) -> dict:
    """Run the headline star-vs-base-rate split on a synthetic panel.

    The DECISIVE statistic is the **Welch t of three-stars reversal mean vs the
    base-rate reversal mean** (``welch_t``) -- the same delta the real-tape headline is
    graded on. A raw one-sample t against zero is also reported for reference but is
    confounded by drift and is never the certifying number. ``active_masks`` (optional,
    from ``data.synthetic_panel``'s ``truth["active_masks"]``) keeps the base rate clean
    of any bar still mid-way through a *different* planted decline -- see
    :func:`base_rate_events`.
    """
    ev = pool_events(data, prior_lookback=prior_lookback, horizons=(horizon,),
                     active_masks=active_masks)
    br = pool_base_rate(data, horizons=(horizon,), prior_lookback=prior_lookback,
                        seed=seed, active_masks=active_masks)
    n = len(ev)
    rev = ev[f"ret_{horizon}"].to_numpy(dtype=float) if n else np.array([])
    base = br[f"ret_{horizon}"].to_numpy(dtype=float) if len(br) else np.array([])
    mean_bps = float(np.nanmean(rev) * 1e4) if n else float("nan")
    base_bps = float(np.nanmean(base) * 1e4) if len(base) else float("nan")
    return {"n": n, "mean_bps": mean_bps, "base_bps": base_bps,
            "delta_bps": mean_bps - base_bps if n and len(base) else float("nan"),
            "tstat": hac_tstat(rev) if n >= MIN_N_FOR_TEST else None,
            "welch_t": welch_t(rev, base)}
