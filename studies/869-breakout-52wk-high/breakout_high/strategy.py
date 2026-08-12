"""Strategy + inference for Study 869 — 52-Week-High Breakout Drift.

The claim: a **fresh 52-week-high breakout** is an *event*. When a name closes at a new
52-week high for the first time (its close tops the trailing 252-day maximum), does it
**drift up** over the next 5/20 days (breakout momentum) or **fade** (resistance /
anchoring)? We measure the forward return of the just-broke-out book against the rest of
the cross-section.

This is distinct from:

* [236-fifty-two-week-high](../../236-fifty-two-week-high/) — George-Hwang **nearness**
  to the 52-week high: a *continuous level* (close / 252-day high) ranked cross-
  sectionally. This study tests the discrete **breakout event** (the first close *above*
  the prior high), not the standing proximity.
* [202-fifty-two-week-low](../../202-fifty-two-week-low/) — the symmetric **low**-side
  anchor, not a high breakout.
* [331-fifty-two-week-range](../../331-fifty-two-week-range/) — position **within** the
  52-week high-low *range* (a normalised level), not the high-breakout event.
* [437-donchian-breakout](../../437-donchian-breakout/) — the classic **Donchian**
  channel breakout (a shorter n-day high, a trend-following entry rule), not the
  specific 52-week-high cross-sectional event studied here.

Method:

* **Close prices.** Build a per-name daily adjusted-close panel.
* **Point-in-time breakout flag.** ``flag[t]`` is True when ``Close[t]`` strictly tops
  the rolling ``lookback``-day maximum of *prior* closes (``.shift(1)``, so today is
  excluded) — a fresh 52-week high known at the close of ``t``, zero look-ahead.
* **Forward-window event sort.** With one documented execution lag, enter at the close
  of ``t+1`` and hold ``horizon`` trading days: ``fwd[t] = Close[t+1+h]/Close[t+1] − 1``.
  On each day ``t`` the **breakout book** is the mean forward-``h`` return of names that
  just broke out; the **rest** book is the mean of the names that did not. The daily
  ``spread = breakout − rest`` series (long just-broke-out, short the rest) is the
  headline — positive = breakout drift, negative = fade.
* **Inference.** Newey-West (HAC) *t* on the daily spread with lags scaled to the
  overlap horizon (forward windows overlap, so a plain *t* would overstate); a
  one-sample *t* and a pooled Welch *t* (breakout vs rest events) cross-check; a
  permutation placebo breaks the event->outcome link; a costed timer charges the
  round-trip friction and short borrow.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

TRADING_DAYS = 252


# --------------------------------------------------------------------------- #
# Return panel + breakout signal
# --------------------------------------------------------------------------- #
def closes_frame(panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Aligned adjusted-close frame (index=date, columns=ticker)."""
    return pd.DataFrame({s: panel[s]["Close"] for s in panel}).sort_index()


def breakout_flags(closes: pd.DataFrame, lookback: int = 252) -> pd.DataFrame:
    """Point-in-time fresh-``lookback``-day-high flag, per name.

    ``flag[t]`` is True when ``Close[t]`` strictly exceeds the maximum of the *prior*
    ``lookback`` closes (``rolling(lookback).max().shift(1)`` — today excluded), i.e. a
    fresh 52-week high. Known at the close of ``t``; the sort adds one execution lag.
    """
    prior_max = closes.rolling(lookback, min_periods=lookback).max().shift(1)
    return (closes > prior_max) & prior_max.notna()


def forward_returns(closes: pd.DataFrame, horizon: int = 5, lag: int = 1) -> pd.DataFrame:
    """Forward ``horizon``-day simple return entered ``lag`` days after the signal.

    ``fwd[t] = Close[t+lag+horizon]/Close[t+lag] − 1`` — the return of a position opened
    at the close of ``t+lag`` (one documented execution lag past the breakout at ``t``)
    and held ``horizon`` trading days.
    """
    entry = closes.shift(-lag)
    exit_ = closes.shift(-(lag + horizon))
    return exit_ / entry - 1.0


# --------------------------------------------------------------------------- #
# The event sort -> long-breakout / short-the-rest forward-return spread
# --------------------------------------------------------------------------- #
def breakout_spreads(
    closes: pd.DataFrame,
    lookback: int = 252,
    horizon: int = 5,
    lag: int = 1,
    min_rest: int = 5,
) -> pd.DataFrame:
    """Daily equal-weight breakout-minus-rest forward-``horizon`` return spread.

    On each day ``t`` the **breakout book** = mean forward-``h`` return of names whose
    ``flag[t]`` is True (fresh 52-week high at close ``t``); the **rest book** = mean of
    names with a valid forward return that did *not* break out. ``spread = brk − rest``
    (long just-broke-out, short the rest). Days with no breakout, or fewer than
    ``min_rest`` valid names in the rest book, are dropped. Fully vectorised.
    """
    flags = breakout_flags(closes, lookback)
    fwd = forward_returns(closes, horizon, lag)
    F = flags.to_numpy(dtype=bool)
    W = fwd.to_numpy(dtype=float)
    valid = ~np.isnan(W)
    brk_mask = F & valid
    rest_mask = (~F) & valid

    n_brk = brk_mask.sum(axis=1)
    n_rest = rest_mask.sum(axis=1)

    Wb = np.where(brk_mask, W, np.nan)
    Wr = np.where(rest_mask, W, np.nan)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        brk_mean = np.nanmean(Wb, axis=1)
        rest_mean = np.nanmean(Wr, axis=1)

    keep = (n_brk >= 1) & (n_rest >= min_rest)
    idx = closes.index[keep]
    return pd.DataFrame(
        {
            "spread": (brk_mean - rest_mean)[keep],
            "brk": brk_mean[keep],
            "rest": rest_mean[keep],
            "n_brk": n_brk[keep],
            "n_rest": n_rest[keep],
        },
        index=idx,
    ).sort_index()


# --------------------------------------------------------------------------- #
# Inference primitives  (copied from Study 803 — the canonical desk set)
# --------------------------------------------------------------------------- #
def one_sample_t(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def welch_t(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else float("nan")


def newey_west_t(x: np.ndarray, lags: int = 10) -> float:
    """HAC (Newey-West, Bartlett kernel) t of mean(x) vs 0."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float("nan")
    mu = x.mean()
    u = x - mu
    gamma0 = float(u @ u) / n
    var = gamma0
    for l in range(1, min(lags, n - 1) + 1):
        w = 1.0 - l / (lags + 1.0)
        cov = float(u[l:] @ u[:-l]) / n
        var += 2.0 * w * cov
    if var <= 0:
        return float("nan")
    se = np.sqrt(var / n)
    return float(mu / se) if se > 0 else float("nan")


def wilson_interval(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z2 = z * z
    mid = (p + z2 / (2 * n)) / (1 + z2 / n)
    half = z * np.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / (1 + z2 / n)
    return (mid - half, mid + half)


# --------------------------------------------------------------------------- #
# Headline stats
# --------------------------------------------------------------------------- #
def breakout_stats(spreads: pd.DataFrame, nw_lags: int = 10) -> dict:
    sp = spreads["spread"].to_numpy(dtype=float)
    win = int(np.sum(sp > 0))
    lo, hi = wilson_interval(win, len(sp))
    return {
        "n_days": int(len(spreads)),
        "spread_bps": float(np.nanmean(sp) * 1e4),
        "t_nw": newey_west_t(sp, nw_lags),
        "t_1s": one_sample_t(sp),
        "brk_bps": float(np.nanmean(spreads["brk"].to_numpy()) * 1e4),
        "rest_bps": float(np.nanmean(spreads["rest"].to_numpy()) * 1e4),
        "welch_t": welch_t(spreads["brk"].to_numpy(), spreads["rest"].to_numpy()),
        "n_breakouts": int(spreads["n_brk"].sum()),
        "hit_rate": win / len(sp) if len(sp) else float("nan"),
        "hit_lo": lo,
        "hit_hi": hi,
    }


# --------------------------------------------------------------------------- #
# Placebo — is the spread real, or a lucky alignment of the event flags?
# --------------------------------------------------------------------------- #
def placebo_pvalue(
    closes: pd.DataFrame,
    lookback: int = 252,
    horizon: int = 5,
    lag: int = 1,
    min_rest: int = 5,
    n_seeds: int = 20,
    n_draws_per_seed: int = 50,
    base_seed: int = 869,
) -> dict:
    """Keep the breakout event flags but read each day's forward returns from a
    **column-permuted** panel (event->outcome link broken, each day's cross-sectional
    forward-return distribution preserved). p = share of permuted worlds whose spread
    mean is >= observed (right-tail test on the long-breakout / short-rest spread)."""
    flags = breakout_flags(closes, lookback)
    fwd = forward_returns(closes, horizon, lag)
    cols = list(closes.columns)
    ncol = len(cols)
    obs = float(breakout_spreads(closes, lookback, horizon, lag, min_rest)["spread"].mean())

    F = flags.to_numpy(dtype=bool)
    W = fwd.to_numpy(dtype=float)
    valid = ~np.isnan(W)
    brk_mask = F & valid
    n_brk = brk_mask.sum(axis=1)
    n_rest = ((~F) & valid).sum(axis=1)
    keep = np.where((n_brk >= 1) & (n_rest >= min_rest))[0]

    means = []
    if len(keep):
        Fk = F[keep]
        Wk = W[keep]
        validk = valid[keep]
        for seed in range(n_seeds):
            rng = np.random.default_rng(base_seed + seed)
            for _ in range(n_draws_per_seed):
                perm = rng.permutation(ncol)
                Wp = Wk[:, perm]              # permute which name's forward return lands where
                vp = validk[:, perm]
                brk = Fk & vp
                rest = (~Fk) & vp
                Wb = np.where(brk, Wp, np.nan)
                Wr = np.where(rest, Wp, np.nan)
                with np.errstate(invalid="ignore"), warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    bm = np.nanmean(Wb, axis=1)
                    rm = np.nanmean(Wr, axis=1)
                d = bm - rm
                means.append(np.nanmean(d))
    means = np.asarray(means)
    return {
        "obs_bps": obs * 1e4,
        "placebo_mean_bps": float(means.mean() * 1e4) if len(means) else float("nan"),
        "placebo_sd_bps": float(means.std(ddof=1) * 1e4) if len(means) > 1 else float("nan"),
        "p_value": float((means >= obs).mean()) if len(means) else float("nan"),
        "n_draws": len(means),
        "draws_bps": means * 1e4,
    }


# --------------------------------------------------------------------------- #
# The costed timer
# --------------------------------------------------------------------------- #
def timer_stats(
    spreads: pd.DataFrame,
    horizon: int = 5,
    cost_bps: float = 5.0,
    borrow_bps_yr: float = 50.0,
) -> dict:
    """Cost the long-breakout / short-the-rest book.

    Each event is a ``horizon``-day round-trip long-short position: we charge two sides
    (long breakout + short rest) × a round-trip (in-and-out) one-way cost, plus borrow on
    the short leg over the holding window. Per-event returns are annualised with the
    ``TRADING_DAYS / horizon`` non-overlapping-period approximation (overlapping windows
    inflate the raw count, so this is the conservative independent-period Sharpe)."""
    sp = spreads["spread"].to_numpy(dtype=float)
    sp = sp[~np.isnan(sp)]
    n = len(sp)
    round_trip_cost = 2.0 * 2.0 * cost_bps / 1e4          # 2 sides × in+out
    borrow_hold = (borrow_bps_yr / 1e4) * (horizon / 365.0)
    net = sp - round_trip_cost - borrow_hold
    gross_mean = float(sp.mean())
    net_mean = float(net.mean())
    sd = float(net.std(ddof=1)) if n > 1 else float("nan")
    periods_yr = TRADING_DAYS / horizon
    sharpe = net_mean / sd * np.sqrt(periods_yr) if sd and sd > 0 else float("nan")
    return {
        "n_events": n,
        "horizon": horizon,
        "gross_bps": gross_mean * 1e4,
        "net_bps": net_mean * 1e4,
        "cost_bps_per_event": (round_trip_cost + borrow_hold) * 1e4,
        "ann_net_pct": net_mean * periods_yr * 100,
        "sharpe_net": sharpe,
        "t_net": one_sample_t(net),
    }


# --------------------------------------------------------------------------- #
# Synthetic-control detector (the machinery proof)
# --------------------------------------------------------------------------- #
def synthetic_detect(panel: dict[str, pd.DataFrame], lookback: int = 252,
                     horizon: int = 5, nw_lags: int = 20) -> dict:
    """Run the headline breakout stats on a synthetic panel."""
    closes = closes_frame(panel)
    sp = breakout_spreads(closes, lookback, horizon)
    ts = breakout_stats(sp, nw_lags)
    return {"spread_bps": ts["spread_bps"], "t_nw": ts["t_nw"],
            "welch_t": ts["welch_t"], "n_days": ts["n_days"],
            "n_breakouts": ts["n_breakouts"]}
