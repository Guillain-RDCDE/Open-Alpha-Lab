"""The stick-sandwich detector, the event study, and its honest controls — Study 686.

The **bullish stick sandwich** (Nison): a bearish candle, a bullish candle that rallies
above it, and a second bearish candle that gives the whole rally back and closes at
~the same level as the first — mechanically, three bars ``t-2, t-1, t``:

1. **Candle t-2** — bearish (``close < open``), the first slice of "bread".
2. **Candle t-1** — bullish (``close > open``) *and* closes above candle t-2's close (the
   "filling": a genuine, if temporary, rally).
3. **Candle t** — bearish (``close < open``) *and* closes back within ``tol`` of candle
   t-2's close (the second slice of "bread" — the rally is completely erased and the tape
   prints the same level twice).
4. **Context** — the leg into candle t-2 is a genuine downtrend (``close[t-3] <
   close[t-3-trend_lookback]``), steelmanning "reversal": the pattern is supposed to *end*
   a decline, not appear at random.

What we test, announced before we run it:

* **The signal.** Forward ``H``-day returns from a long entered at the **next close**
  (one documented lag) after the sandwich completes, for ``H`` in 5/10/20/60 days.
* **The base rate.** The *same* long-only forward-return distribution measured on **every**
  other eligible bar in the panel (the drift-matched, direction-unconditional baseline) —
  a stick sandwich adds value only if it beats what any random long entry on the same tape
  already earns. The decisive number is the **Welch t of sandwich-minus-base-rate**.
* **Multiple comparisons.** Four horizons is four simultaneous looks at the same question;
  the certifying bar is a **Bonferroni-corrected** critical |t| (``k=4``), not the naive
  |t| >= 2.
* **The geometry placebo.** Candidates that satisfy the down/up/down *context* (bearish,
  bullish-and-rallies, bearish) but **not** the equal-close tolerance — "almost sandwiches"
  that share everything except the defining geometry. If the *equal close* itself carries
  no information, a same-size random draw from that candidate pool should do just as well
  as the real sandwiches.
* **Costs.** One-way bps charged on entry and exit (round trip), long-only, no borrow.

No look-ahead: the sandwich (and its downtrend context) is fully knowable at the close of
bar *t*; the trade enters at *t+1*'s close and is held a fixed horizon.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

HORIZONS = (5, 10, 20, 60)
DEFAULT_TOL = 0.0015          # closes "meet" within 15 bps — same convention as 460
DEFAULT_TREND_LOOKBACK = 10   # bars defining the prior down leg into the pattern
MIN_N_FOR_TEST = 8            # below this, report raw numbers only — no t-stat theatre
BONFERRONI_K = len(HORIZONS)  # four horizons -> four simultaneous looks


# --------------------------------------------------------------------------- #
# Pattern detection (bullish stick sandwich)
# --------------------------------------------------------------------------- #
def _sandwich_mask(bars: pd.DataFrame, tol: float = DEFAULT_TOL,
                   trend_lookback: int = DEFAULT_TREND_LOOKBACK) -> np.ndarray:
    """Boolean array: True at position *t* if a bullish stick sandwich completes at *t*.

    All conditions are read from data at or before *t* (no look-ahead), vectorized over the
    whole tape at once:
      * down leg into the pattern: close[t-3] < close[t-3-trend_lookback]
      * candle t-2 (bread 1) bearish: close[t-2] < open[t-2]
      * candle t-1 (filling)  bullish, rallies: close[t-1] > open[t-1] and close[t-1] > close[t-2]
      * candle t   (bread 2)  bearish: close[t] < open[t]
      * closes meet: |close[t] - close[t-2]| / close[t-2] <= tol
    """
    hits, idx = _pattern_hits(bars, tol=tol, trend_lookback=trend_lookback, require_meet=True)
    mask = np.zeros(len(bars), dtype=bool)
    if idx.size:
        mask[idx] = hits
    return mask


def _candidate_mask(bars: pd.DataFrame, trend_lookback: int = DEFAULT_TREND_LOOKBACK) -> np.ndarray:
    """Bars satisfying the down/up/down CONTEXT of a sandwich but NOT the equal close.

    A candidate = down leg + bearish bread1 + bullish-and-rallying filling + bearish bread2.
    These are "almost sandwiches" sharing everything with a real sandwich except the defining
    equal-close geometry. The geometry placebo draws its entries from this pool.
    """
    hits, idx = _pattern_hits(bars, trend_lookback=trend_lookback, require_meet=False)
    mask = np.zeros(len(bars), dtype=bool)
    if idx.size:
        mask[idx] = hits
    return mask


def _pattern_hits(bars: pd.DataFrame, trend_lookback: int, tol: float = DEFAULT_TOL,
                  require_meet: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Vectorized shared core for :func:`_sandwich_mask` / :func:`_candidate_mask`.

    Returns ``(hits, idx)`` where ``idx`` are the candidate bar positions >= ``3+trend_lookback``
    and ``hits`` is the boolean condition array aligned to ``idx``.
    """
    o = bars["open"].to_numpy(dtype=float)
    c = bars["close"].to_numpy(dtype=float)
    n = c.size
    L = trend_lookback
    if n <= 3 + L:
        return np.asarray([], dtype=bool), np.asarray([], dtype=int)
    idx = np.arange(3 + L, n)
    c_t2, c_t1, c_t0 = c[idx - 2], c[idx - 1], c[idx]
    o_t2, o_t1, o_t0 = o[idx - 2], o[idx - 1], o[idx]
    bearish_bread1 = c_t2 < o_t2
    bullish_fill = (c_t1 > o_t1) & (c_t1 > c_t2)
    bearish_bread2 = c_t0 < o_t0
    down_leg = c[idx - 3] < c[idx - 3 - L]
    hits = bearish_bread1 & bullish_fill & bearish_bread2 & down_leg
    if require_meet:
        with np.errstate(divide="ignore", invalid="ignore"):
            meet = np.where(c_t2 > 0, np.abs(c_t0 - c_t2) / c_t2 <= tol, False)
        hits = hits & meet
    return hits, idx


def sandwich_entries(bars: pd.DataFrame, tol: float = DEFAULT_TOL,
                     trend_lookback: int = DEFAULT_TREND_LOOKBACK) -> pd.DatetimeIndex:
    """Bars completing a bullish stick sandwich — the buy signals.

    Each True bar is an independent 3-bar completion (not a run), so no de-duplication is
    needed beyond what the mask already encodes. Entry is executed at the next close by
    :func:`forward_returns`.
    """
    mask = _sandwich_mask(bars, tol=tol, trend_lookback=trend_lookback)
    return bars.index[mask]


# --------------------------------------------------------------------------- #
# Forward-return engine
# --------------------------------------------------------------------------- #
def forward_returns(close: pd.Series, entries, horizon: int, cost_bps: float = 0.0,
                    pos: dict | None = None) -> np.ndarray:
    """Forward ``horizon``-day return for each entry, entered at the *next* close (one lag).

    ``cost_bps`` is a one-way cost (charged twice: in + out) subtracted from each trade's
    return. Trades whose window overruns the tape are dropped. Pass a precomputed
    ``pos`` (``{date: integer position}``) to skip rebuilding it on every call — the
    geometry placebo's repeated draws lean on this.
    """
    if pos is None:
        pos = {d: i for i, d in enumerate(close.index)}
    p = close.to_numpy(dtype=float)
    n = p.size
    out = []
    for d in entries:
        i = pos.get(d)
        if i is None or i + 1 + horizon >= n:
            continue
        e = i + 1                      # enter at next close
        r = p[e + horizon] / p[e] - 1.0
        out.append(r - 2.0 * cost_bps * 1e-4)
    return np.asarray(out, dtype=float)


def base_rate_returns(close: pd.Series, horizon: int, warmup: int = DEFAULT_TREND_LOOKBACK + 3,
                      cost_bps: float = 0.0) -> np.ndarray:
    """The unconditional base rate: the SAME long, entered the next close, on EVERY eligible
    bar (after warm-up) — "what does any random long earn on this tape at this horizon?"
    """
    idx = close.index[warmup:]
    return forward_returns(close, idx, horizon, cost_bps=cost_bps)


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def hac_t(x: np.ndarray) -> float:
    """Newey-West (HAC) one-sample t-stat of the mean against zero."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 6:
        return float("nan")
    mu = x.mean()
    e = x - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for kk in range(1, lags + 1):
        w = 1.0 - kk / (lags + 1.0)
        lrv += 2.0 * w * float(e[kk:] @ e[:-kk]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def welch_t(a: np.ndarray, b: np.ndarray, min_n: int = MIN_N_FOR_TEST) -> float | None:
    """Welch t of mean(a) - mean(b) (unequal variances). ``None`` if either side is too small."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) < min_n or len(b) < min_n:
        return None
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else None


def bonferroni_critical(k: int = BONFERRONI_K, alpha: float = 0.05) -> float:
    """Two-sided Bonferroni-corrected |t| critical value for ``k`` simultaneous tests."""
    from scipy import stats

    return float(stats.norm.ppf(1.0 - alpha / (2.0 * k)))


def summarize(returns: np.ndarray, min_n: int = MIN_N_FOR_TEST) -> dict:
    """Headline per-trade stats: count, win-rate, mean (bps), per-trade Sharpe, HAC t.

    Below ``min_n`` observations the t-stat is deliberately not computed (``tested=False``,
    ``t=None``) — a t on a handful of points is theatre, not evidence.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n == 0:
        return {"n": 0, "win": float("nan"), "mean_bps": float("nan"),
                "sharpe": float("nan"), "t": None, "tested": False}
    tested = n >= min_n
    return {
        "n": int(n),
        "win": float((r > 0).mean()),
        "mean_bps": float(r.mean() * 1e4),
        "sharpe": float(r.mean() / r.std(ddof=1)) if n > 1 and r.std() > 0 else float("nan"),
        "t": hac_t(r) if tested else None,
        "tested": tested,
    }


# --------------------------------------------------------------------------- #
# Pooling across the basket
# --------------------------------------------------------------------------- #
def pool_entries(panel: dict[str, pd.DataFrame], tol: float = DEFAULT_TOL,
                 trend_lookback: int = DEFAULT_TREND_LOOKBACK) -> dict[str, pd.DatetimeIndex]:
    """``{ticker: sandwich entry dates}`` across the whole basket."""
    return {t: sandwich_entries(b, tol=tol, trend_lookback=trend_lookback) for t, b in panel.items()}


def pool_forward(panel: dict[str, pd.DataFrame], entries: dict[str, pd.DatetimeIndex],
                 horizon: int, cost_bps: float = 0.0) -> np.ndarray:
    """Concatenate forward returns for a per-ticker entries dict across the whole basket."""
    out = [forward_returns(panel[t]["close"], ent, horizon, cost_bps=cost_bps)
           for t, ent in entries.items() if len(ent)]
    return np.concatenate(out) if out else np.asarray([], dtype=float)


def pool_base_rate(panel: dict[str, pd.DataFrame], horizon: int,
                   warmup: int = DEFAULT_TREND_LOOKBACK + 3, cost_bps: float = 0.0) -> np.ndarray:
    out = [base_rate_returns(b["close"], horizon, warmup=warmup, cost_bps=cost_bps)
           for b in panel.values()]
    return np.concatenate(out) if out else np.asarray([], dtype=float)


# --------------------------------------------------------------------------- #
# Geometry placebo — destroy the equal close, keep the down/up/down context
# --------------------------------------------------------------------------- #
def geometry_placebo(bars: pd.DataFrame, horizon: int, tol: float = DEFAULT_TOL,
                     trend_lookback: int = DEFAULT_TREND_LOOKBACK,
                     n_draws: int = 1000, seed: int = 686) -> dict:
    """Placebo: keep the down/up/down context, scramble the equal-close test.

    Draws, from the context-matched candidate pool, a random subset of the same size as the
    real sandwich set — i.e. fires the same number of "failed-rally after a down leg"
    entries but ignores whether the two bearish closes actually meet. Returns the share of
    placebo runs whose mean forward return **beats** the real sandwich return — the honest
    "is the equal-close geometry load-bearing, or is any failed-rally-after-a-decline just as
    good?" p-value, plus the observed mean.
    """
    close = bars["close"]
    pos = {d: i for i, d in enumerate(close.index)}
    real_ent = sandwich_entries(bars, tol=tol, trend_lookback=trend_lookback)
    obs_r = forward_returns(close, real_ent, horizon, pos=pos)
    obs = float(np.mean(obs_r)) if obs_r.size else float("nan")
    cand = _candidate_mask(bars, trend_lookback=trend_lookback)
    cand_idx = bars.index[cand]
    n_sand = len(real_ent)
    if n_sand == 0 or len(cand_idx) < n_sand or not np.isfinite(obs):
        return {"obs": obs, "p_value": float("nan"), "n_draws": 0, "n_candidates": len(cand_idx)}
    rng = np.random.default_rng(seed)
    cand_arr = np.asarray(cand_idx)
    beats = 0
    valid = 0
    for _ in range(n_draws):
        pick = rng.choice(cand_arr, size=n_sand, replace=False)
        rr = forward_returns(close, pd.DatetimeIndex(sorted(pick)), horizon, pos=pos)
        if rr.size == 0:
            continue
        valid += 1
        if rr.mean() >= obs:
            beats += 1
    p = (beats + 1) / (valid + 1) if valid else float("nan")
    return {"obs": obs, "p_value": float(p), "n_draws": valid, "n_candidates": len(cand_idx)}


# --------------------------------------------------------------------------- #
# Orchestrator
# --------------------------------------------------------------------------- #
def run_experiment(panel: dict[str, pd.DataFrame], tol: float = DEFAULT_TOL,
                   trend_lookback: int = DEFAULT_TREND_LOOKBACK, cost_bps: float = 5.0,
                   placebo_ticker: str = "SPY", placebo_horizon: int = 20,
                   n_draws: int = 1000) -> dict:
    """Run the full stick-sandwich gauntlet on a panel: sandwich vs base-rate, all horizons.

    Per horizon: pooled sandwich stats (gross + net), the pooled unconditional base rate,
    the sandwich-minus-base-rate Welch t, and the Bonferroni-corrected critical |t| for
    ``len(HORIZONS)`` simultaneous tests. Also runs the SPY geometry placebo at
    ``placebo_horizon``.
    """
    entries = pool_entries(panel, tol=tol, trend_lookback=trend_lookback)
    n_sand = sum(len(e) for e in entries.values())
    per_ticker_n = {t: len(e) for t, e in entries.items()}

    out = {"n_sandwiches": int(n_sand), "per_ticker_n": per_ticker_n,
           "n_names": len(panel), "bonferroni_crit": bonferroni_critical(len(HORIZONS)),
           "by_h": {}}

    for h in HORIZONS:
        gross = pool_forward(panel, entries, h, cost_bps=0.0)
        net = pool_forward(panel, entries, h, cost_bps=cost_bps)
        base = pool_base_rate(panel, h)
        g = summarize(gross)
        ns = summarize(net)
        b = summarize(base)
        delta = (g["mean_bps"] - b["mean_bps"]
                 if np.isfinite(g["mean_bps"]) and np.isfinite(b["mean_bps"]) else float("nan"))
        wt = welch_t(gross, base)
        out["by_h"][h] = {"gross": g, "net": ns, "base": b, "delta_bps": delta, "welch_t": wt}

    pb = geometry_placebo(panel[placebo_ticker], placebo_horizon, tol=tol,
                          trend_lookback=trend_lookback, n_draws=n_draws)
    out["placebo"] = {"ticker": placebo_ticker, "horizon": placebo_horizon, **pb}
    return out


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(panel: dict[str, pd.DataFrame], horizon: int = 20) -> dict:
    """Run the headline sandwich-vs-base-rate Welch split on a synthetic panel."""
    entries = pool_entries(panel)
    n = sum(len(e) for e in entries.values())
    sand = pool_forward(panel, entries, horizon)
    base = pool_base_rate(panel, horizon)
    mean_bps = float(np.nanmean(sand) * 1e4) if sand.size else float("nan")
    base_bps = float(np.nanmean(base) * 1e4) if base.size else float("nan")
    return {"n": int(n), "mean_bps": mean_bps, "base_bps": base_bps,
            "delta_bps": mean_bps - base_bps if sand.size and base.size else float("nan"),
            "welch_t": welch_t(sand, base)}
