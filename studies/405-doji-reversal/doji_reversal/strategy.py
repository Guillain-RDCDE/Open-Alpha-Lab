"""The doji detector, the event study, and its honest controls — Study 405.

A **doji** is the candlestick of indecision: the session opens and closes at (almost) the
same price, so the *real body* |close − open| is tiny relative to the bar's total range
high − low.  The folk recipe (Nison, *Japanese Candlestick Charting Techniques*) says a
doji marks a stand-off between buyers and sellers and therefore **precedes a reversal** of
the move that led into it — a down-move into a doji should bounce; an up-move into a doji
should fade.

What we test, announced before we run it:

* **Detector.** A bar at *t* is a doji if ``|close − open| <= body_frac * (high − low)``
  and the range is non-trivial.  ``body_frac = 0.1`` (a 10%-of-range body) is the standard
  textbook cut; we sweep it for robustness.

* **The reversal event.** The folk claim is *directional and conditional on the prior
  move*.  We tag each doji with the sign of the 2-day move into it (``prior``) and define
  the **reversal trade** as taking a position *against* that prior move at the next
  session's open: down-into-doji → go long, up-into-doji → go short.  Forward returns are
  measured at **1 / 3 / 5 / 10** days, entered at *t+1*'s open (one documented execution
  lag).

* **The base rate.** The reversal return is compared against the *unconditional* base rate
  — the same horizon return measured on **all** sessions (direction-matched by the prior
  move), so we ask "does a doji add reversal information beyond what any bar would?"

* **Arbiters.** A one-sample HAC (Newey–West) *t* on the reversal returns against zero; a
  **label-shuffle placebo** that re-draws which bars are 'dojis' at the same rate and keeps
  the best of nothing; costs (one-way bps, shorts pay borrow); and a **myth-check** that
  adds a trend/volume filter to see whether conditioning rescues the rule.

No look-ahead: the doji and the prior-move sign are known at the close of *t*; the trade
is entered at *t+1*'s open and held a fixed horizon.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Doji detector
# ---------------------------------------------------------------------------
def doji_flags(bars: pd.DataFrame, body_frac: float = 0.10) -> pd.Series:
    """Boolean Series: True on bars whose real body is <= ``body_frac`` of the range.

    A doji needs (a) a tiny body relative to its high-low range and (b) a non-degenerate
    range (we drop bars where high == low).  Returns a Series aligned to ``bars``.
    """
    rng = (bars["high"] - bars["low"]).to_numpy(dtype=float)
    body = np.abs(bars["close"].to_numpy(dtype=float) - bars["open"].to_numpy(dtype=float))
    with np.errstate(divide="ignore", invalid="ignore"):
        frac = np.where(rng > 0, body / rng, np.inf)
    flags = (frac <= body_frac) & (rng > 0)
    return pd.Series(flags, index=bars.index, name="doji")


# ---------------------------------------------------------------------------
# Event table — one row per (ticker, doji session)
# ---------------------------------------------------------------------------
def doji_events(bars: pd.DataFrame, body_frac: float = 0.10, lookback: int = 2,
                horizons=(1, 3, 5, 10), trend_win: int = 20,
                vol_win: int = 20) -> pd.DataFrame:
    """One row per doji bar with the prior-move sign and forward reversal returns.

    For each doji at integer position ``i``:

    * ``prior`` = sign of ``close[i] / close[i - lookback] - 1`` (the move into the doji).
    * ``rev_h`` = the **reversal** return over horizon ``h``, entered at *i+1*'s open and
      exited at *i+h*'s close, signed *against* the prior move
      (``rev = -prior * (close[i+h] / open[i+1] - 1)``).  A positive ``rev_h`` means the
      doji *did* precede a reversal.
    * Context for the myth-check: ``above_sma`` (close[i] vs its ``trend_win`` SMA) and
      ``vol_ratio`` (the bar's volume vs its ``vol_win`` average).

    Bars without a full forward window are dropped.  No look-ahead: everything in a row is
    knowable at the close of *i* except the forward path, which starts at *i+1*'s open.
    """
    o = bars["open"].to_numpy(dtype=float)
    c = bars["close"].to_numpy(dtype=float)
    v = bars["volume"].to_numpy(dtype=float)
    n = len(bars)
    flags = doji_flags(bars, body_frac=body_frac).to_numpy()
    sma = bars["close"].rolling(trend_win).mean().to_numpy()
    vbar = bars["volume"].rolling(vol_win).mean().to_numpy()
    hmax = max(horizons)

    rows = []
    for i in range(lookback, n - hmax - 1):
        if not flags[i]:
            continue
        if c[i - lookback] <= 0:
            continue
        prior = np.sign(c[i] / c[i - lookback] - 1.0)
        if prior == 0:
            continue
        entry = o[i + 1]
        if not np.isfinite(entry) or entry <= 0:
            continue
        row = {"pos": i, "prior": int(prior),
               "above_sma": bool(c[i] > sma[i]) if np.isfinite(sma[i]) else False,
               "vol_ratio": float(v[i] / vbar[i]) if np.isfinite(vbar[i]) and vbar[i] > 0
               else np.nan}
        for h in horizons:
            raw = c[i + h] / entry - 1.0
            row[f"rev_{h}"] = float(-prior * raw)   # reversal return (against prior move)
            row[f"raw_{h}"] = float(raw)            # raw forward move (for diagnostics)
        rows.append(row)
    return pd.DataFrame(rows)


def base_rate_events(bars: pd.DataFrame, lookback: int = 2, horizons=(1, 3, 5, 10),
                     sample: int | None = None, seed: int = 405) -> pd.DataFrame:
    """The unconditional base rate: the same reversal trade on *every* eligible bar.

    Identical construction to :func:`doji_events` but without the doji condition — every
    bar with a defined prior move and forward window contributes a row.  Optionally
    sub-samples ``sample`` bars (seeded) to match the doji count for a like-for-like
    comparison.  This is "what would the same against-the-prior-move bet earn on a random
    day?"
    """
    o = bars["open"].to_numpy(dtype=float)
    c = bars["close"].to_numpy(dtype=float)
    n = len(bars)
    hmax = max(horizons)
    rows = []
    for i in range(lookback, n - hmax - 1):
        if c[i - lookback] <= 0:
            continue
        prior = np.sign(c[i] / c[i - lookback] - 1.0)
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
                horizons=(1, 3, 5, 10), **kw) -> pd.DataFrame:
    """Concatenate :func:`doji_events` across every ticker in ``panel``."""
    frames = []
    for t, bars in panel.items():
        ev = doji_events(bars, body_frac=body_frac, horizons=horizons, **kw)
        if len(ev):
            ev = ev.copy()
            ev["ticker"] = t
            frames.append(ev)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def pool_base_rate(panel: dict[str, pd.DataFrame], horizons=(1, 3, 5, 10),
                   sample_per: int | None = None, seed: int = 405) -> pd.DataFrame:
    frames = []
    for t, bars in panel.items():
        br = base_rate_events(bars, horizons=horizons, sample=sample_per, seed=seed)
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
    """One-sample Newey–West HAC *t*-statistic on the mean of ``x`` against zero."""
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


def summarize(returns: np.ndarray, cost_bps: float = 0.0,
              borrow_bps: float = 0.0, short_frac: float = 0.5) -> dict:
    """Headline stats for a vector of (reversal) returns.

    ``cost_bps`` is a one-way round-trip cost (entry+exit) deducted from every trade;
    ``borrow_bps`` is a per-trade borrow charge applied to the short fraction (here the
    reversal book is ~half short by construction).  Returns count, win-rate, mean (bps),
    per-trade Sharpe, and the HAC *t* on the gross mean.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n == 0:
        return {k: float("nan") for k in
                ["n", "win_rate", "mean_bps", "sharpe", "tstat", "net_bps"]}
    net = r - cost_bps * 1e-4 - borrow_bps * 1e-4 * short_frac
    return {
        "n": int(n),
        "win_rate": float((r > 0).mean()),
        "mean_bps": float(r.mean() * 1e4),
        "sharpe": float(r.mean() / r.std(ddof=1)) if n > 1 and r.std() > 0 else float("nan"),
        "tstat": hac_tstat(r),
        "net_bps": float(net.mean() * 1e4),
        "net_t": hac_tstat(net),
    }


def label_shuffle_placebo(bars: pd.DataFrame, n_dojis: int, horizon: int = 5,
                          n_draws: int = 2000, lookback: int = 2,
                          seed: int = 405) -> dict:
    """Placebo: re-draw ``n_dojis`` 'fake doji' bars at random and recompute the mean.

    Builds the full base-rate event table once, then on each of ``n_draws`` draws picks
    ``n_dojis`` random rows and records their mean reversal return at ``horizon``.  The
    returned ``p`` is the fraction of placebo means at least as large as the real doji
    mean — a non-parametric null that respects "what does a same-size random set of bars
    give?"  Requires the real doji mean be passed via :func:`run_experiment`.
    """
    br = base_rate_events(bars, lookback=lookback, horizons=(horizon,))
    col = f"rev_{horizon}"
    vals = br[col].to_numpy(dtype=float)
    vals = vals[np.isfinite(vals)]
    rng = np.random.default_rng(seed)
    if len(vals) < n_dojis or n_dojis < 1:
        return {"placebo_means": np.array([]), "pool": vals}
    means = np.array([rng.choice(vals, size=n_dojis, replace=False).mean()
                      for _ in range(n_draws)])
    return {"placebo_means": means, "pool": vals}


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def run_experiment(panel: dict[str, pd.DataFrame], body_frac: float = 0.10,
                   horizons=(1, 3, 5, 10), cost_bps: float = 2.0,
                   borrow_bps: float = 1.0, n_draws: int = 2000,
                   seed: int = 405) -> dict:
    """Run the full doji-reversal teardown on a panel and return a results dict.

    Computes, per horizon: the pooled doji reversal stats (gross + net), the direction-
    matched unconditional base rate, the doji-minus-base delta, and a pooled label-shuffle
    placebo *p*-value.  Also reports the total doji count and the per-year rate.
    """
    ev = pool_events(panel, body_frac=body_frac, horizons=horizons)
    br = pool_base_rate(panel, horizons=horizons)
    n_dojis = len(ev)
    out = {"n_dojis": int(n_dojis), "horizons": list(horizons),
           "body_frac": body_frac, "per_horizon": {}}

    # crude trades/yr estimate from the longest tape
    spans = [(b.index[-1] - b.index[0]).days / 365.25 for b in panel.values() if len(b) > 1]
    yrs = max(spans) if spans else float("nan")
    out["years"] = float(yrs)
    out["dojis_per_name_per_yr"] = float(n_dojis / max(len(panel), 1) / yrs) if yrs else float("nan")

    rng_master = np.random.default_rng(seed)
    for h in horizons:
        rev = ev[f"rev_{h}"].to_numpy(dtype=float) if n_dojis else np.array([])
        base = br[f"rev_{h}"].to_numpy(dtype=float) if len(br) else np.array([])
        s = summarize(rev, cost_bps=cost_bps, borrow_bps=borrow_bps)
        b = summarize(base)
        delta = s["mean_bps"] - b["mean_bps"] if np.isfinite(s["mean_bps"]) and np.isfinite(b["mean_bps"]) else float("nan")

        # pooled placebo: draw n_dojis bars from the pooled base-rate distribution
        pool_vals = base[np.isfinite(base)]
        real_mean = float(np.nanmean(rev)) if rev.size else float("nan")
        if pool_vals.size >= n_dojis and n_dojis >= 1 and np.isfinite(real_mean):
            seed_h = int(rng_master.integers(0, 1 << 31))
            r2 = np.random.default_rng(seed_h)
            placebo = np.array([r2.choice(pool_vals, size=n_dojis, replace=False).mean()
                                for _ in range(n_draws)])
            p = float((placebo >= real_mean).mean())
        else:
            placebo = np.array([])
            p = float("nan")

        out["per_horizon"][h] = {
            "doji": s, "base": b, "delta_bps": float(delta),
            "placebo_p": p,
            "placebo_mean_bps": float(placebo.mean() * 1e4) if placebo.size else float("nan"),
        }
    return out


# ---------------------------------------------------------------------------
# Myth-check: does a trend / volume filter rescue the rule?
# ---------------------------------------------------------------------------
def filtered_stats(panel: dict[str, pd.DataFrame], horizon: int = 5,
                   body_frac: float = 0.10, mode: str = "trend") -> dict:
    """Re-run the horizon-``h`` reversal stats on filtered subsets of dojis.

    ``mode='trend'`` splits dojis by whether the close sits above or below its 20-day SMA
    (the folk filter "reversal only against the trend"); ``mode='volume'`` splits by
    whether the doji printed on above- or below-average volume (a 'confirmed' doji).
    Returns one summarize() dict per subset.
    """
    ev = pool_events(panel, body_frac=body_frac, horizons=(horizon,))
    col = f"rev_{horizon}"
    out = {}
    if not len(ev):
        return out
    if mode == "trend":
        out["above_sma"] = summarize(ev.loc[ev["above_sma"], col].to_numpy())
        out["below_sma"] = summarize(ev.loc[~ev["above_sma"], col].to_numpy())
    elif mode == "volume":
        hi = ev["vol_ratio"] >= 1.0
        out["high_vol"] = summarize(ev.loc[hi.fillna(False), col].to_numpy())
        out["low_vol"] = summarize(ev.loc[~hi.fillna(True), col].to_numpy())
    return out
