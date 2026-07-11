"""The tri-star detector, the event study, and its honest controls — Study 685.

A **doji** is the candlestick of indecision: the session opens and closes at (almost)
the same price, so the real body |close - open| is tiny relative to the bar's total
range high - low. The **tri-star** is the rarest of the classic reversal patterns
(Nison, *Japanese Candlestick Charting Techniques*): **three dojis in a row**,
supposedly marking not just indecision but a *major* trend exhaustion — three
consecutive sessions where neither side can move the tape is read as capitulation of
whichever side had been driving the prior trend.

What we test, announced before we run it:

* **Detector.** A bar is a doji if ``|close - open| <= body_frac * (high - low)`` with
  a non-trivial range (``body_frac = 0.10``, the same textbook cut used by sibling
  studies 405/458). A **tri-star** is three *consecutive* doji bars, positions
  ``t-2, t-1, t``. We report **two cuts**: the *strict, literature-faithful* one --
  Nison's tri-star requires the middle doji's range to clear *both* neighbours (a
  true gapped "island star"), the shape that makes the pattern legendarily rare --
  and a *loose* one -- three dojis back to back with no gap requirement, the plain
  reading of "three dojis in a row" a looser proponent might accept, which is common
  enough to actually run inference on. The strict cut is the primary claim under
  test; the loose cut is the honest power/robustness check ("even without demanding
  the rare gap geometry, is there anything here?").

* **The reversal event.** The folk claim is *directional and conditional on the prior
  trend*: after an up-move, a tri-star at the top calls a top; after a down-move, a
  tri-star at the bottom calls a bottom. We tag each tri-star with the sign of the
  10-day move *into* the block (``prior``, measured up to the first doji) and define
  the **reversal trade** as taking a position *against* that prior move at the next
  session's open: down-into-tri-star -> go long, up-into-tri-star -> go short. Forward
  returns are measured at **5 / 10 / 20 / 60** days (the "major reversal" claim implies
  a multi-week effect, not a 1-day pop) entered at *t+1*'s open (one documented
  execution lag).

* **The base rate.** The reversal return is compared against the *unconditional* base
  rate -- the same against-the-move bet measured on **every** eligible bar in the
  panel (direction-matched by the prior 10-day move) -- so we ask "does a tri-star add
  reversal information beyond what any bar's mean-reversion already gives?"

* **The honest sample-size rule.** A tri-star needs three independent low-probability
  events back to back; pooled across a 60-name, ~25-year panel the count is expected
  to be tiny. If the pooled count at a horizon falls below :data:`MIN_N_FOR_TEST` we
  do **not** compute a t-stat at all (a t on 3 observations is decoration, not
  evidence) -- we report the raw numbers and say plainly "too few to test." Where a
  test *is* run, four horizons means four looks at the same question, so a
  **Bonferroni-corrected** critical value (``k=4``) is the bar, not the naive |t| >= 2.

* **Arbiters.** A one-sample HAC (Newey-West) *t* on the reversal returns against
  zero (when n permits); a **label-shuffle placebo** that re-draws which 3-bar blocks
  are 'tri-stars' at the same rate; costs (one-way bps, shorts pay borrow).

No look-ahead: the tri-star and the prior-trend sign are known at the close of bar
*t* (the third doji); the trade is entered at *t+1*'s open and held a fixed horizon.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)
MIN_N_FOR_TEST = 8          # below this, report raw numbers only -- no t-stat theatre
BONFERRONI_K = len(HORIZONS)  # four horizons -> four simultaneous looks


# ---------------------------------------------------------------------------
# Doji / tri-star detector
# ---------------------------------------------------------------------------
def doji_flags(bars: pd.DataFrame, body_frac: float = 0.10) -> pd.Series:
    """Boolean Series: True on bars whose real body is <= ``body_frac`` of the range.

    A doji needs (a) a tiny body relative to its high-low range and (b) a
    non-degenerate range (bars with high == low are dropped). Aligned to ``bars``.
    """
    rng = (bars["high"] - bars["low"]).to_numpy(dtype=float)
    body = np.abs(bars["close"].to_numpy(dtype=float) - bars["open"].to_numpy(dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(rng > 0, body / rng, np.inf)
    flags = (frac <= body_frac) & (rng > 0)
    return pd.Series(flags, index=bars.index, name="doji")


def tri_star_flags(bars: pd.DataFrame, body_frac: float = 0.10) -> pd.Series:
    """Boolean Series: True at bar *t* iff *t-2, t-1, t* are three consecutive dojis."""
    d = doji_flags(bars, body_frac=body_frac)
    tri = d & d.shift(1).fillna(False) & d.shift(2).fillna(False)
    return tri.rename("tri_star")


def gapped_star_flags(bars: pd.DataFrame, body_frac: float = 0.10) -> pd.Series:
    """Stricter robustness cut: the tri-star *and* the middle doji's range clears both
    neighbours (a true 'island star' shape, gapped both sides -- the purist reading of
    the pattern, vs. the looser "three dojis in a row" primary detector).
    """
    tri = tri_star_flags(bars, body_frac=body_frac)
    hi, lo = bars["high"].to_numpy(float), bars["low"].to_numpy(float)
    n = len(bars)
    out = np.zeros(n, dtype=bool)
    tri_np = tri.to_numpy()
    for i in range(2, n):
        if not tri_np[i]:
            continue
        a, b, c = i - 2, i - 1, i
        gapped = (lo[b] > hi[a] and lo[b] > hi[c]) or (hi[b] < lo[a] and hi[b] < lo[c])
        out[i] = gapped
    return pd.Series(out, index=bars.index, name="gapped_star")


# ---------------------------------------------------------------------------
# Event table -- one row per (ticker, tri-star block)
# ---------------------------------------------------------------------------
def tri_star_events(bars: pd.DataFrame, body_frac: float = 0.10, lookback: int = 10,
                    horizons=HORIZONS, trend_win: int = 20,
                    vol_win: int = 20) -> pd.DataFrame:
    """One row per tri-star block with the prior-trend sign and forward reversal returns.

    For each tri-star ending at integer position ``i`` (the third doji):

    * ``prior`` = sign of ``close[i-2] / close[i-2-lookback] - 1`` (the move *into* the
      block, measured up to its first bar -- the trend the "major reversal" claim
      needs to reverse).
    * ``rev_h`` = the **reversal** return over horizon ``h``, entered at *i+1*'s open
      and exited at *i+h*'s close, signed *against* the prior move
      (``rev = -prior * (close[i+h] / open[i+1] - 1)``). A positive ``rev_h`` means the
      tri-star *did* precede a reversal.
    * ``gapped`` -- whether this block also satisfies the stricter island-star cut.

    Bars without a full forward window are dropped. No look-ahead: everything in a row
    is knowable at the close of *i* except the forward path, which starts at *i+1*'s open.
    """
    o = bars["open"].to_numpy(dtype=float)
    c = bars["close"].to_numpy(dtype=float)
    n = len(bars)
    tri = tri_star_flags(bars, body_frac=body_frac).to_numpy()
    gapped = gapped_star_flags(bars, body_frac=body_frac).to_numpy()
    hmax = max(horizons)

    rows = []
    for i in range(lookback + 2, n - hmax - 1):
        if not tri[i]:
            continue
        if c[i - 2 - lookback] <= 0:
            continue
        prior = np.sign(c[i - 2] / c[i - 2 - lookback] - 1.0)
        if prior == 0:
            continue
        entry = o[i + 1]
        if not np.isfinite(entry) or entry <= 0:
            continue
        row = {"pos": i, "prior": int(prior), "gapped": bool(gapped[i])}
        for h in horizons:
            raw = c[i + h] / entry - 1.0
            row[f"rev_{h}"] = float(-prior * raw)
            row[f"raw_{h}"] = float(raw)
        rows.append(row)
    return pd.DataFrame(rows)


def base_rate_events(bars: pd.DataFrame, lookback: int = 10, horizons=HORIZONS,
                     sample: int | None = None, seed: int = 685) -> pd.DataFrame:
    """The unconditional base rate: the same reversal trade on *every* eligible bar.

    Identical construction to :func:`tri_star_events` but without the tri-star
    condition -- every bar with a defined 10-day prior move and forward window
    contributes a row. "What would the same against-the-trend bet earn on a random
    day?"
    """
    o = bars["open"].to_numpy(dtype=float)
    c = bars["close"].to_numpy(dtype=float)
    n = len(bars)
    hmax = max(horizons)
    rows = []
    for i in range(lookback + 2, n - hmax - 1):
        if c[i - 2 - lookback] <= 0:
            continue
        prior = np.sign(c[i - 2] / c[i - 2 - lookback] - 1.0)
        if prior == 0:
            continue
        entry = o[i + 1]
        if not np.isfinite(entry) or entry <= 0:
            continue
        row = {"pos": i, "prior": int(prior)}
        for h in horizons:
            row[f"rev_{h}"] = float(-prior * (c[i + h] / entry - 1.0))
        rows.append(row)
    out = pd.DataFrame(rows)
    if sample is not None and len(out) > sample:
        out = out.sample(n=sample, random_state=seed).reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# Pooling across the basket
# ---------------------------------------------------------------------------
def pool_events(panel: dict[str, pd.DataFrame], body_frac: float = 0.10,
                horizons=HORIZONS, **kw) -> pd.DataFrame:
    """Concatenate :func:`tri_star_events` across every ticker in ``panel``."""
    frames = []
    for t, bars in panel.items():
        ev = tri_star_events(bars, body_frac=body_frac, horizons=horizons, **kw)
        if len(ev):
            ev = ev.copy()
            ev["ticker"] = t
            frames.append(ev)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def pool_base_rate(panel: dict[str, pd.DataFrame], horizons=HORIZONS, lookback: int = 10,
                   sample_per: int | None = None, seed: int = 685) -> pd.DataFrame:
    frames = []
    for t, bars in panel.items():
        br = base_rate_events(bars, horizons=horizons, lookback=lookback,
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
    "does the tri-star reversal mean differ from the base-rate mean?" ``None`` (not
    computed) if either side has fewer than :data:`MIN_N_FOR_TEST` finite observations --
    the same "no timer if too few" discipline as :func:`summarize`.
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


def summarize(returns: np.ndarray, cost_bps: float = 0.0,
              borrow_bps: float = 0.0, short_frac: float = 0.5,
              min_n: int = MIN_N_FOR_TEST) -> dict:
    """Headline stats for a vector of (reversal) returns.

    ``cost_bps`` is a one-way round-trip cost (entry+exit) deducted from every trade;
    ``borrow_bps`` is a per-trade borrow charge applied to the short fraction (the
    reversal book is ~half short by construction). Below ``min_n`` observations the
    t-stat is deliberately **not computed** (``tstat=None``, ``tested=False``) --
    a t on a handful of points is theatre, not evidence, and the honest thing is to
    say "too few to test" rather than print a decorative number.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n == 0:
        return {"n": 0, "win_rate": float("nan"), "mean_bps": float("nan"),
                "net_bps": float("nan"), "tstat": None, "net_t": None, "tested": False}
    net = r - cost_bps * 1e-4 - borrow_bps * 1e-4 * short_frac
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
                          n_draws: int = 2000, lookback: int = 10,
                          seed: int = 685) -> dict:
    """Placebo: re-draw ``n_events`` 'fake tri-star' bars at random from the pooled
    base-rate pool and recompute the mean. The returned ``p`` is the fraction of
    placebo means at least as large as the real tri-star mean -- a non-parametric null
    that respects "what does a same-size random set of bars give?" Requires the real
    tri-star mean to be passed via the orchestrator.
    """
    br = pool_base_rate(panel, horizons=(horizon,), seed=seed)
    if not len(br):
        return {"placebo_means": np.array([]), "pool": np.array([])}
    col = f"rev_{horizon}"
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
def run_experiment(panel: dict[str, pd.DataFrame], body_frac: float = 0.10,
                   horizons=HORIZONS, cost_bps: float = 2.0,
                   borrow_bps: float = 1.0, n_draws: int = 2000,
                   seed: int = 685) -> dict:
    """Run the full tri-star teardown on a panel and return a results dict.

    Computes, per horizon: the pooled tri-star reversal stats (gross + net, tested
    only if n >= :data:`MIN_N_FOR_TEST`), the direction-matched unconditional base
    rate, the tri-star-minus-base delta, and a pooled label-shuffle placebo p-value.
    Also reports the total tri-star count, the gapped-star sub-count, and the
    Bonferroni-corrected critical |t| for ``len(horizons)`` simultaneous tests.
    """
    ev = pool_events(panel, body_frac=body_frac, horizons=horizons)
    n_tri = len(ev)
    n_gapped = int(ev["gapped"].sum()) if n_tri else 0
    out = {"n_tri_star": int(n_tri), "n_gapped": n_gapped, "horizons": list(horizons),
           "body_frac": body_frac, "bonferroni_crit": bonferroni_critical(len(horizons)),
           "per_horizon": {}, "strict_per_horizon": {}}

    spans = [(b.index[-1] - b.index[0]).days / 365.25 for b in panel.values() if len(b) > 1]
    yrs = max(spans) if spans else float("nan")
    out["years"] = float(yrs)
    out["n_names"] = len(panel)

    gap_ev = ev[ev["gapped"]] if n_tri else ev

    rng_master = np.random.default_rng(seed)
    for h in horizons:
        rev = ev[f"rev_{h}"].to_numpy(dtype=float) if n_tri else np.array([])
        s = summarize(rev, cost_bps=cost_bps, borrow_bps=borrow_bps)
        # the strict, literature-faithful island tri-star (both gaps) -- the genuinely
        # rare cut; almost never has enough n to test, and that IS the finding.
        srev = gap_ev[f"rev_{h}"].to_numpy(dtype=float) if len(gap_ev) else np.array([])
        ss = summarize(srev, cost_bps=cost_bps, borrow_bps=borrow_bps)

        br_h = pool_base_rate(panel, horizons=(h,), seed=seed)
        base = br_h[f"rev_{h}"].to_numpy(dtype=float) if len(br_h) else np.array([])
        b = summarize(base)
        delta = (s["mean_bps"] - b["mean_bps"]
                 if np.isfinite(s["mean_bps"]) and np.isfinite(b["mean_bps"]) else float("nan"))
        welch = welch_t(rev, base)  # the decisive tri-vs-base significance test

        pool_vals = base[np.isfinite(base)]
        real_mean = float(np.nanmean(rev)) if rev.size else float("nan")
        if pool_vals.size >= n_tri and n_tri >= 1 and np.isfinite(real_mean):
            seed_h = int(rng_master.integers(0, 1 << 31))
            r2 = np.random.default_rng(seed_h)
            placebo = np.array([r2.choice(pool_vals, size=n_tri, replace=False).mean()
                                for _ in range(n_draws)])
            p = float((placebo >= real_mean).mean())
        else:
            placebo = np.array([])
            p = float("nan")

        out["per_horizon"][h] = {
            "tri": s, "base": b, "delta_bps": float(delta), "welch_t": welch,
            "placebo_p": p,
            "placebo_mean_bps": float(placebo.mean() * 1e4) if placebo.size else float("nan"),
        }
        out["strict_per_horizon"][h] = {"tri": ss}
    return out


# ---------------------------------------------------------------------------
# Synthetic-control detector (the machinery proof)
# ---------------------------------------------------------------------------
def synthetic_detect(data: dict[str, pd.DataFrame], horizon: int = 20,
                     body_frac: float = 0.10, lookback: int = 10,
                     seed: int = 685) -> dict:
    """Run the headline tri-star-vs-base-rate split on a synthetic panel.

    The DECISIVE statistic is the **Welch t of tri-star reversal mean vs the
    base-rate reversal mean** (``welch_t``) -- the same delta the real-tape headline
    is graded on. A raw one-sample t against zero is also reported for reference, but
    (as sibling study 405 demonstrates on the real tape) it is confounded by drift and
    is never the certifying number.
    """
    ev = pool_events(data, body_frac=body_frac, horizons=(horizon,), lookback=lookback)
    br = pool_base_rate(data, horizons=(horizon,), lookback=lookback, seed=seed)
    n = len(ev)
    rev = ev[f"rev_{horizon}"].to_numpy(dtype=float) if n else np.array([])
    base = br[f"rev_{horizon}"].to_numpy(dtype=float) if len(br) else np.array([])
    mean_bps = float(np.nanmean(rev) * 1e4) if n else float("nan")
    base_bps = float(np.nanmean(base) * 1e4) if len(base) else float("nan")
    return {"n": n, "mean_bps": mean_bps, "base_bps": base_bps,
            "delta_bps": mean_bps - base_bps if n and len(base) else float("nan"),
            "tstat": hac_tstat(rev) if n >= MIN_N_FOR_TEST else None,
            "welch_t": welch_t(rev, base)}
