"""The concealing-baby-swallow detector, the event study, and its honest controls —
Study 688.

The **concealing baby swallow** (Nison; Bulkowski) is a four-bar bullish reversal
claimed to end a downtrend through outright capitulation:

* **Day 1, day 2** — two black **marubozu** (long bearish real bodies, no meaningful
  shadows): one-way, no-hesitation selling.
* **Day 3** — another black candle, but this one **gaps down** at the open, rallies
  *intraday* back up into day 2's real body (a long upper shadow — the rally is
  "concealed" by a candle that still closes near its own low), and closes lower
  again.
* **Day 4** — a black candle that **totally engulfs day 3**, trading above its high
  and below its low, and closes at a fresh low for the move.

The folk reading: the failed rally hidden inside day 3, followed by a low that can't
even hold on day 4, is exhaustion — the reversal is imminent. What we test, announced
before we run it:

* **Detector, two cuts.** A **loose** cut codes the plain geometric reading (near-
  marubozu bodies, the day-3 overlap-then-fail, the day-4 full engulf) a practical
  chartist would implement. A **strict, literature-closer** cut additionally requires
  (a) day 1/2 to be *true* marubozu (essentially zero shadow), (b) day 3 to open with
  a genuine gap down from day 2's close, and (c) day 4 to open *inside* day 3's upper
  shadow before swallowing it. The strict cut is the primary claim under test; the
  loose cut is the honest power/robustness check and the count we report as "how rare,
  really" — literature calls this the single rarest candle formation in the canon, and
  the sample size is itself part of the finding.

* **The reversal event.** Bullish only: we take the reversal trade **long** at the
  next session's **open** (one documented execution lag, no look-ahead), reading
  forward 1/5/10/20-day returns.

* **The base rate.** The reversal return is compared against the *unconditional* base
  rate — the same long bet on **every** bar that sits in a matching downtrend context
  with four bearish closes in a row (the coarse shape), whether or not the precise
  marubozu/overlap/engulf geometry fired. This isolates "does the specific concealing
  shape add information beyond just four red days in a downtrend" from plain mean
  reversion.

* **The honest sample-size rule.** A pattern this restrictive, pooled across a very
  large basket and decades of history, can still return a **near-zero** count. Below
  :data:`MIN_N_FOR_TEST` pooled events we do **not** compute a *t*-statistic at all —
  a *t* on a handful of observations (or none) is decoration, not evidence — and we
  say plainly "too few to test." Where a test *does* run, four horizons means a
  **Bonferroni**-corrected critical value (``k=4``), not a naive |t| >= 2.

* **Arbiters.** A Welch *t* of the reversal mean vs the base-rate mean (the decisive
  statistic — never a one-sample *t* against zero, which would just measure the
  basket's unconditional drift), where *n* permits; a one-sample HAC (Newey-West) *t*
  where *n* permits; a label-shuffle placebo; costs (one-way bps round trip,
  long-only, no borrow).

No look-ahead: the pattern and its context are known at the close of bar *t* (day 4);
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
# Concealing-baby-swallow detector
# ---------------------------------------------------------------------------
def cbs_flags(bars: pd.DataFrame, prior_lookback: int = 10,
             marubozu_frac: float = 0.15, warn_shadow_frac: float = 0.30,
             tail_frac: float = 0.25) -> pd.Series:
    """Boolean Series: True at bar *t* iff bars *t-3..t* form a LOOSE concealing baby
    swallow.

    Confirmed at the close of bar *t* (day 4), looking back at *t-3..t-1* (days 1-3):

    1. **Downtrend context**: close[t-3] < close[t-3-``prior_lookback``] — a genuine
       downtrend leading into day 1, the pattern's own premise.
    2. **Four bearish bodies**, t-3..t: each closes below its open.
    3. **Day 1, day 2 near-marubozu**: combined shadow <= ``marubozu_frac`` of that
       bar's range — essentially no wick, one-way selling.
    4. **Day 2 makes a new low close**: close[t-2] < close[t-3].
    5. **Day 3 rallies into day 2's real body**: high[t-1] > close[t-2] — the failed
       rally the pattern "conceals".
    6. **Day 3 still closes at a new low, near its own low**: close[t-1] < close[t-2]
       and lower shadow[t-1] <= ``tail_frac`` of its range.
    7. **Day 3 shows a real upper shadow**: upper shadow[t-1] >= ``warn_shadow_frac``
       of its range — the rally has to be visible to be "concealed".
    8. **Day 4 totally engulfs day 3** (including the shadow): high[t] >= high[t-1]
       and low[t] <= low[t-1].
    9. **Day 4 closes at a fresh low**: close[t] < close[t-1].
    """
    c, o = bars["close"], bars["open"]
    hi, lo = bars["high"], bars["low"]
    rng = _range(bars).clip(lower=1e-12)
    upper_sh = _upper_shadow(bars)
    lower_sh = _lower_shadow(bars)

    c1, o1 = c.shift(3), o.shift(3)
    c2, o2 = c.shift(2), o.shift(2)
    c3, o3 = c.shift(1), o.shift(1)
    hi3, lo3 = hi.shift(1), lo.shift(1)
    rng1, rng3 = rng.shift(3), rng.shift(1)
    ush3, lsh3 = upper_sh.shift(1), lower_sh.shift(1)

    ctx = c1 < c.shift(3 + prior_lookback)

    bearish1 = c1 < o1
    bearish2 = c2 < o2
    bearish3 = c3 < o3
    bearish4 = c < o

    marubozu1 = (upper_sh.shift(3) + lower_sh.shift(3)) <= marubozu_frac * rng1
    marubozu2 = (upper_sh.shift(2) + lower_sh.shift(2)) <= marubozu_frac * rng.shift(2)

    day2_new_low = c2 < c1
    day3_overlap = hi3 > c2
    day3_new_low = c3 < c2
    day3_small_tail = lsh3 <= tail_frac * rng3
    day3_real_wick = ush3 >= warn_shadow_frac * rng3

    day4_engulfs = (hi >= hi3) & (lo <= lo3)
    day4_new_low = c < c3

    result = (ctx & bearish1 & bearish2 & bearish3 & bearish4
              & marubozu1 & marubozu2 & day2_new_low
              & day3_overlap & day3_new_low & day3_small_tail & day3_real_wick
              & day4_engulfs & day4_new_low)
    return result.fillna(False).rename("cbs")


def strict_cbs_flags(bars: pd.DataFrame, prior_lookback: int = 10,
                     marubozu_frac: float = 0.05, warn_shadow_frac: float = 0.50,
                     tail_frac: float = 0.10) -> pd.Series:
    """Boolean Series: the STRICT, literature-closer concealing baby swallow (the loose
    cut, with tighter marubozu/tail/wick thresholds, plus):

    10. Day 2 **gaps or continues down** from day 1's close: open[t-2] <= close[t-3].
    11. Day 3 **gaps down** at the open from day 2's close: open[t-1] <= close[t-2] —
        a *true* gap-down open, the rally that follows is the whole point.
    12. Day 4 **opens inside day 3's upper shadow**: close[t-1] < open[t] <= high[t-1]
        — the "concealed" territory day 3 rallied into but couldn't hold.
    """
    loose = cbs_flags(bars, prior_lookback=prior_lookback, marubozu_frac=marubozu_frac,
                      warn_shadow_frac=warn_shadow_frac, tail_frac=tail_frac)

    o = bars["open"]
    c1 = bars["close"].shift(3)
    o2 = bars["open"].shift(2)
    c2 = bars["close"].shift(2)
    o3 = bars["open"].shift(1)
    c3 = bars["close"].shift(1)
    hi3 = bars["high"].shift(1)

    day2_gap = o2 <= c1
    day3_gap = o3 <= c2
    day4_opens_in_shadow = (o > c3) & (o <= hi3)

    result = loose & day2_gap & day3_gap & day4_opens_in_shadow
    return result.fillna(False).rename("strict_cbs")


# ---------------------------------------------------------------------------
# Event table -- one row per (ticker, cbs block)
# ---------------------------------------------------------------------------
def cbs_events(bars: pd.DataFrame, prior_lookback: int = 10,
              horizons=HORIZONS) -> pd.DataFrame:
    """One row per concealing-baby-swallow event with the forward LONG returns and
    the strict flag.

    For each confirming bar *i* (day 4): enter the next session's **open** (one
    execution lag, no look-ahead) and hold horizon ``h`` to the close ``h`` sessions
    later (``h=1`` = the entry day's own close). Bars without a full forward window
    are dropped.
    """
    o = bars["open"].to_numpy(dtype=float)
    c = bars["close"].to_numpy(dtype=float)
    n = len(bars)
    loose = cbs_flags(bars, prior_lookback=prior_lookback).to_numpy()
    strict = strict_cbs_flags(bars, prior_lookback=prior_lookback).to_numpy()
    hmax = max(horizons)

    rows = []
    for i in range(prior_lookback + 3, n - hmax - 1):
        if not loose[i]:
            continue
        e = i + 1
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
                     seed: int = 688) -> pd.DataFrame:
    """The unconditional base rate: the same LONG bet on every bar with a matching
    downtrend context AND four bearish closes in a row (the coarse shape), regardless
    of whether the precise marubozu/overlap/engulf geometry fired. "What does the same
    buy-after-four-red-days-in-a-downtrend bet earn on a random bar?" — isolates the
    pattern's own information from plain downtrend mean reversion.
    """
    c = bars["close"].to_numpy(dtype=float)
    o = bars["open"].to_numpy(dtype=float)
    n = len(bars)
    hmax = max(horizons)
    rows = []
    for i in range(prior_lookback + 3, n - hmax - 1):
        i1 = i - 3
        if c[i1 - prior_lookback] <= 0:
            continue
        if not (c[i1] < c[i1 - prior_lookback]):
            continue
        if not all(c[i1 + k] < o[i1 + k] for k in range(4)):
            continue
        e = i + 1
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
                horizons=HORIZONS) -> pd.DataFrame:
    """Concatenate :func:`cbs_events` across every ticker in ``panel``."""
    frames = []
    for t, bars in panel.items():
        ev = cbs_events(bars, prior_lookback=prior_lookback, horizons=horizons)
        if len(ev):
            ev = ev.copy()
            ev["ticker"] = t
            frames.append(ev)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def pool_base_rate(panel: dict[str, pd.DataFrame], horizons=HORIZONS,
                   prior_lookback: int = 10, sample_per: int | None = None,
                   seed: int = 688) -> pd.DataFrame:
    frames = []
    for t, bars in panel.items():
        br = base_rate_events(bars, horizons=horizons, prior_lookback=prior_lookback,
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
    "does the concealing-baby-swallow reversal mean differ from the base-rate mean?"
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
    ``2 * cost_bps``. Long-only, no borrow. Below ``min_n`` observations the *t*-stat
    is deliberately **not computed** (``tstat=None``, ``tested=False``) -- a *t* on a
    handful of points (or none) is theatre, not evidence, and the honest thing is to
    say "too few to test" rather than print a decorative number.
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
                          seed: int = 688) -> dict:
    """Placebo: re-draw ``n_events`` 'fake concealing-baby-swallow' bars at random from
    the pooled base-rate pool and recompute the mean. The returned ``p`` is the
    fraction of placebo means at least as large as the real reversal mean -- a
    non-parametric null that respects "what does a same-size random set of matching
    downtrend bars give?" Returns an empty result if ``n_events < 1``.
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
                   n_draws: int = 2000, seed: int = 688) -> dict:
    """Run the full concealing-baby-swallow teardown on a panel and return a results
    dict.

    Computes, per horizon: the pooled loose-cut reversal stats (gross + net, tested
    only if n >= :data:`MIN_N_FOR_TEST`), the strict-cut stats, the direction-matched
    unconditional base rate, the reversal-minus-base delta, a pooled label-shuffle
    placebo p-value (only where both samples are large enough), and the
    Bonferroni-corrected critical |t| for ``len(horizons)`` simultaneous tests.
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
    out["name_years"] = float(sum((b.index[-1] - b.index[0]).days / 365.25
                                  for b in panel.values() if len(b) > 1))

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
            "ladder": s, "base": b, "delta_bps": float(delta), "welch_t": welch,
            "placebo_p": p,
            "placebo_mean_bps": float(placebo.mean() * 1e4) if placebo.size else float("nan"),
        }

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
            "ladder": ss, "delta_bps": float(strict_delta), "welch_t": strict_welch,
            "placebo_p": strict_p,
        }
    return out


# ---------------------------------------------------------------------------
# Synthetic-control detector (the machinery proof)
# ---------------------------------------------------------------------------
def synthetic_detect(data: dict[str, pd.DataFrame], horizon: int = 20,
                     prior_lookback: int = 10, seed: int = 688) -> dict:
    """Run the headline reversal-vs-base-rate split on a synthetic panel.

    The DECISIVE statistic is the **Welch t of the reversal mean vs the base-rate
    mean** (``welch_t``) -- the same delta the real-tape headline is graded on. A raw
    one-sample t against zero is also reported for reference but is confounded by
    drift and is never the certifying number.
    """
    ev = pool_events(data, prior_lookback=prior_lookback, horizons=(horizon,))
    br = pool_base_rate(data, horizons=(horizon,), prior_lookback=prior_lookback, seed=seed)
    n = len(ev)
    rev = ev[f"ret_{horizon}"].to_numpy(dtype=float) if n else np.array([])
    base = br[f"ret_{horizon}"].to_numpy(dtype=float) if len(br) else np.array([])
    mean_bps = float(np.nanmean(rev) * 1e4) if n else float("nan")
    base_bps = float(np.nanmean(base) * 1e4) if len(base) else float("nan")
    return {"n": n, "mean_bps": mean_bps, "base_bps": base_bps,
            "delta_bps": mean_bps - base_bps if n and len(base) else float("nan"),
            "tstat": hac_tstat(rev) if n >= MIN_N_FOR_TEST else None,
            "welch_t": welch_t(rev, base)}
