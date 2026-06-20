"""The volatility event-study engine and its honest controls — Study 318.

Three questions, three tools:

1. **Does realized vol spike into the election?** ``realized_vol`` builds a rolling
   annualised vol; ``vol_ratio_around`` measures, per election, the ratio of realized
   vol in the ``pre`` sessions *before* election day to a normal baseline window. A ratio
   > 1 means vol is elevated. ``placebo_vol_ratio`` re-runs the same measurement on
   random non-election dates — the falsification distribution — so a "spike" must sit in
   its tail to count.

2. **Does implied vol over-price it (a variance-risk premium)?** ``vrp_around`` compares
   the implied-vol level (VIX) just *before* the election to the realized vol that
   actually shows up *over* the event window. Implied-minus-realized is the premium a
   vol seller collects; a positive average means the option market over-charged for
   election risk.

3. **Could you trade it?** ``short_vol_carry`` is the tradable overlay: at the close of
   a pre-election entry session (election day is a *calendar fact*, known in advance) it
   takes a short-vol position proxied by a short straddle-like payoff — it collects the
   implied variance priced in and pays out the realized variance that follows, over a
   ``hold``-session window. One execution lag, applied once; one-way costs charged twice
   (open + close) against NAV; selling vol is the short leg, so it pays a carry/borrow-
   like haircut. We report gross and net side by side, and race the net P&L against a
   plain *non-election* short-vol carry (sell vol on random dates) so the verdict is
   excess-of-the-always-on-carry, not the carry itself.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Realized volatility
# ---------------------------------------------------------------------------
def daily_log_returns(bars: pd.DataFrame) -> pd.Series:
    """Close-to-close log returns."""
    return np.log(bars["close"]).diff()


def realized_vol(bars: pd.DataFrame, window: int = 21) -> pd.Series:
    """Trailing annualised realized volatility (%), a rolling std of log returns."""
    r = daily_log_returns(bars)
    return r.rolling(window, min_periods=max(5, window // 2)).std() * np.sqrt(
        TRADING_DAYS_PER_YEAR
    ) * 100.0


# ---------------------------------------------------------------------------
# 1) The realized-vol spike around an election
# ---------------------------------------------------------------------------
def _pos(idx: pd.DatetimeIndex, d: pd.Timestamp) -> int | None:
    """First session position on or after ``d``; None if past the end."""
    p = idx.searchsorted(d)
    return int(p) if p < len(idx) else None


def vol_ratio_around(
    bars: pd.DataFrame,
    election_dates,
    pre: int = 10,
    baseline: int = 63,
    gap: int = 21,
) -> np.ndarray:
    """Per-election ratio of pre-election realized vol to a normal baseline.

    For each election we take the std of daily log returns over the ``pre`` sessions
    ending at election day, and divide by the std over a ``baseline``-session window that
    ends ``gap`` sessions *before* the pre-window starts (so the baseline is "normal
    times", not contaminated by the run-up). A ratio > 1 means vol was elevated into the
    election. Returns one number per election with a complete window.
    """
    r = daily_log_returns(bars)
    vals = r.to_numpy()
    idx = bars.index
    out = []
    for d in pd.to_datetime(pd.Series(election_dates)):
        p = _pos(idx, d)
        if p is None:
            continue
        pre_lo, pre_hi = p - pre, p
        base_hi = pre_lo - gap
        base_lo = base_hi - baseline
        if base_lo < 1 or pre_hi > len(vals):
            continue
        pre_sd = np.nanstd(vals[pre_lo:pre_hi], ddof=1)
        base_sd = np.nanstd(vals[base_lo:base_hi], ddof=1)
        if base_sd > 0:
            out.append(pre_sd / base_sd)
    return np.asarray(out, dtype=float)


def placebo_vol_ratio(
    bars: pd.DataFrame,
    n_events: int,
    pre: int = 10,
    baseline: int = 63,
    gap: int = 21,
    n_draws: int = 2000,
    seed: int = 318,
) -> np.ndarray:
    """Falsification: mean vol-ratio on ``n_draws`` sets of random non-election dates.

    A real election vol-spike must put the observed mean ratio in the upper tail of this
    distribution. This is a *machinery/placebo* control — it shows the harness finds
    nothing where nothing is special; it can never, alone, certify a real-market effect.
    """
    r = daily_log_returns(bars)
    vals = r.to_numpy()
    n = len(vals)
    lo = baseline + gap + pre + 2
    hi = n - 1
    if hi <= lo or n_events <= 0:
        return np.array([])
    rng = np.random.default_rng(seed)
    out = np.empty(n_draws)
    for d in range(n_draws):
        locs = rng.integers(lo, hi, size=n_events)
        ratios = []
        for p in locs:
            pre_sd = np.nanstd(vals[p - pre:p], ddof=1)
            base_hi = p - pre - gap
            base_sd = np.nanstd(vals[base_hi - baseline:base_hi], ddof=1)
            if base_sd > 0:
                ratios.append(pre_sd / base_sd)
        out[d] = np.nanmean(ratios) if ratios else np.nan
    return out[np.isfinite(out)]


# ---------------------------------------------------------------------------
# 2) The variance-risk premium around an election (implied vs subsequent realized)
# ---------------------------------------------------------------------------
def vrp_around(
    bars: pd.DataFrame,
    vix: pd.Series,
    election_dates,
    pre: int = 1,
    post: int = 21,
) -> pd.DataFrame:
    """Per-election implied-minus-realized vol over the event window (in vol points).

    Implied vol is the VIX level ``pre`` sessions before the election (what the option
    market charged going in). Realized vol is the annualised std of log returns over the
    ``post`` sessions *after* the entry session (what actually happened). Implied minus
    realized is the variance-risk premium a vol seller collected (positive = over-priced).

    Returns a frame: ``election_date, implied, realized, vrp`` (one row per usable event).
    """
    r = daily_log_returns(bars)
    vals = r.to_numpy()
    idx = bars.index
    vix = vix.reindex(idx).ffill()
    rows = []
    for d in pd.to_datetime(pd.Series(election_dates)):
        p = _pos(idx, d)
        if p is None:
            continue
        entry = p - pre
        if entry < 1 or entry + post >= len(vals):
            continue
        implied = float(vix.iloc[entry])
        rv = np.nanstd(vals[entry + 1: entry + 1 + post], ddof=1) * np.sqrt(
            TRADING_DAYS_PER_YEAR
        ) * 100.0
        if not (np.isfinite(implied) and np.isfinite(rv)):
            continue
        rows.append(
            {
                "election_date": idx[p],
                "implied": implied,
                "realized": float(rv),
                "vrp": implied - float(rv),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 3) The tradable overlay — short-vol carry into the election
# ---------------------------------------------------------------------------
def short_vol_carry(
    bars: pd.DataFrame,
    vix: pd.Series,
    entry_dates,
    pre: int = 5,
    hold: int = 21,
    cost_vol_pts: float = 0.0,
) -> pd.DataFrame:
    """Short-vol carry P&L per event, in *vol points*.

    At the close of the session ``pre`` sessions before each ``entry_date`` (election day
    is a calendar fact, known in advance — no information lag), we *sell* volatility at
    the implied level and buy it back at the realized level that shows up over the next
    ``hold`` sessions. The per-trade P&L of a short-vol position is therefore

        pnl_vol_pts = implied - realized_over_hold

    expressed in vol points — a clean, units-stable proxy for a delta-hedged short
    straddle / short-variance-swap payoff (positive when implied over-priced realized).
    ``cost_vol_pts`` is a round-trip transaction/slippage cost in vol points charged once
    per trade (open + close), and short vol pays it — so net = gross - cost_vol_pts.

    One execution lag, applied once: realized is measured over sessions *after* the entry
    close. Returns a per-event ledger: ``entry_date, implied, realized, pnl_gross,
    pnl_net``.
    """
    r = daily_log_returns(bars)
    vals = r.to_numpy()
    idx = bars.index
    vix = vix.reindex(idx).ffill()
    rows = []
    for d in pd.to_datetime(pd.Series(entry_dates)):
        p = _pos(idx, d)
        if p is None:
            continue
        entry = p - pre
        if entry < 1 or entry + hold >= len(vals):
            continue
        implied = float(vix.iloc[entry])
        realized = np.nanstd(vals[entry + 1: entry + 1 + hold], ddof=1) * np.sqrt(
            TRADING_DAYS_PER_YEAR
        ) * 100.0
        if not (np.isfinite(implied) and np.isfinite(realized)):
            continue
        gross = implied - realized
        rows.append(
            {
                "entry_date": idx[entry],
                "implied": implied,
                "realized": float(realized),
                "pnl_gross": float(gross),
                "pnl_net": float(gross - cost_vol_pts),
            }
        )
    return pd.DataFrame(rows)


def random_carry_dates(
    bars: pd.DataFrame,
    n: int,
    pre: int = 5,
    hold: int = 21,
    seed: int = 318,
) -> pd.DatetimeIndex:
    """``n`` random non-election session dates usable as short-vol entries (the control)."""
    idx = bars.index
    lo, hi = pre + 2, len(idx) - hold - 2
    if hi <= lo:
        return pd.DatetimeIndex([])
    rng = np.random.default_rng(seed)
    return idx[rng.integers(lo, hi, size=n)]


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------
def tstat(x: np.ndarray, mu0: float = 0.0) -> float:
    """Cross-event t-stat of the mean vs ``mu0`` (events ~independent in time)."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2 or x.std(ddof=1) == 0:
        return float("nan")
    return float((x.mean() - mu0) / (x.std(ddof=1) / np.sqrt(n)))


def bootstrap_ci(x: np.ndarray, n_boot: int = 5000, alpha: float = 0.05,
                 seed: int = 318) -> tuple[float, float]:
    """Percentile bootstrap CI on the mean (resampling events with replacement).

    Each event is summarised to one number, so there is no within-event serial structure
    to preserve; the i.i.d.-over-events bootstrap is the right resampling unit.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.array([x[rng.integers(0, n, size=n)].mean() for _ in range(n_boot)])
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def percentile_in(value: float, dist: np.ndarray) -> float:
    """Percentile (0..100) of ``value`` within ``dist`` — the placebo-tail position."""
    dist = np.asarray(dist, dtype=float)
    dist = dist[np.isfinite(dist)]
    if dist.size == 0 or not np.isfinite(value):
        return float("nan")
    return float((dist <= value).mean() * 100.0)


def summarize(x: np.ndarray) -> dict:
    """n, mean, std, win-rate (>0), and cross-event t for a per-event vector."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = x.size
    if n == 0:
        return {"n": 0, "mean": float("nan"), "std": float("nan"),
                "win_rate": float("nan"), "tstat": float("nan")}
    return {
        "n": int(n),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if n > 1 else float("nan"),
        "win_rate": float((x > 0).mean()),
        "tstat": tstat(x),
    }
