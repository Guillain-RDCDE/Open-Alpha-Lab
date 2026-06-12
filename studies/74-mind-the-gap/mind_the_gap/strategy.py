"""The strategy and its honest controls -- Study 74 (Mind-the-Gap).

The folk recipe: 'An opening gap always fills.'  Enter at the open in the direction
opposite to the gap (fade it, targeting the prior close), with a stop for risk
management.  The study separates two distinct questions:

1. **Mechanical fill rate** -- how often does the intraday range include the prior
   close, regardless of direction?  A small gap can fill trivially (the day's range
   brackets it by pure noise).  The base rate is high by construction for tiny gaps
   and says nothing about forecasting skill.
2. **Forecastable edge** -- after controlling for the base rate, does *knowing there
   is a gap* let you trade the fade profitably?  We measure this with:
   (a) a conditional P(fill | gap size bucket) vs an unconditional base rate, and
   (b) a **symmetric** barrier backtest -- enter at the open, target = ``tp_R * ATR``
       in the fade direction, stop = ``stop_R * ATR`` in the adverse direction --
       pinned against a random-direction control so the verdict is "vs a coin",
       never an absolute backtest.  The symmetric-ATR exit is the only payoff that
       is direction-fair: a coin earns approximately 0, just like in study 72.

No look-ahead: the gap is known at the open (prior close vs today open); the intraday
range (high/low) is only known at the close.  We enter at the open.

Exit regimes share one engine (:func:`run_trades`):
- Direction ``d = -sign(gap_pct)`` for the fade; overridden by ``directions`` for the
  random-control arm.
- Take-profit at ``tp_R * ATR(atr_n)`` in the trade direction.
- Stop at ``stop_R * ATR(atr_n)`` in the adverse direction.
- End-of-day close if neither barrier is hit.  On daily bars both barriers live in
  the high/low range; a bar that straddles both is treated as a stop (conservative,
  same discipline as study 72).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Gap classification helpers
# ---------------------------------------------------------------------------
def gap_size_bucket(gap_pct: pd.Series) -> pd.Series:
    """Label each gap with a size bucket: 'small' | 'medium' | 'large'.

    - ``small``  : |gap| < 0.25% -- within the bid-ask and overnight noise band;
                   fills trivially most of the time (mechanical, not forecastable).
    - ``medium`` : 0.25% <= |gap| < 1.00% -- the typical overnight drift range.
    - ``large``  : |gap| >= 1.00% -- news-driven; fills are rarer and more costly
                   to fade because the stop must be wide.
    """
    abs_g = gap_pct.abs()
    bucket = pd.Series("medium", index=gap_pct.index, dtype="object")
    bucket[abs_g < 0.25]  = "small"
    bucket[abs_g >= 1.00] = "large"
    return bucket


def atr(daily: pd.DataFrame, n: int = 20) -> pd.Series:
    """Average True Range over ``n`` sessions, in price units.

    True range = max(high-low, |high-prev_close|, |low-prev_close|).
    Simple rolling average (Wilder-style).
    """
    prev_close = daily["close"].shift(1)
    tr = pd.concat(
        [
            daily["high"] - daily["low"],
            (daily["high"] - prev_close).abs(),
            (daily["low"]  - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


# ---------------------------------------------------------------------------
# Fill-rate decomposition (the "mechanical vs forecastable" split)
# ---------------------------------------------------------------------------
def fill_rate_by_bucket(daily: pd.DataFrame) -> pd.DataFrame:
    """Conditional fill rates by gap-size bucket, with Wilson 95% CI.

    Returns a DataFrame with columns:
        bucket, n_gaps, n_fills, fill_rate, ci_lo, ci_hi

    Wilson intervals are used throughout because the cell sizes can be small
    (large gaps are rare) and the normal approximation fails at the extremes.
    """
    from scipy.stats import norm  # lazy import -- only this function needs scipy

    z = norm.ppf(0.975)  # 95% two-sided
    daily = daily.dropna(subset=["gap_pct", "gap_fill"])
    daily = daily[daily["gap_pct"].abs() > 0]  # skip zero-gap sessions

    daily = daily.copy()
    daily["bucket"] = gap_size_bucket(daily["gap_pct"])

    rows = []
    for bucket in ["small", "medium", "large"]:
        sub = daily[daily["bucket"] == bucket]
        n = len(sub)
        k = int(sub["gap_fill"].sum())
        if n == 0:
            continue
        p = k / n
        denom  = 1.0 + z**2 / n
        centre = (p + z**2 / (2 * n)) / denom
        margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
        rows.append({
            "bucket":    bucket,
            "n_gaps":    n,
            "n_fills":   k,
            "fill_rate": p,
            "ci_lo":     max(0.0, centre - margin),
            "ci_hi":     min(1.0, centre + margin),
        })
    return pd.DataFrame(rows)


def random_directions(n: int, seed: int = 0) -> np.ndarray:
    """A reproducible vector of +/-1 -- the control arm's coin."""
    rng = np.random.default_rng(seed)
    return rng.choice([-1, 1], size=n)


# ---------------------------------------------------------------------------
# Barrier backtest -- daily bars, intraday high/low as the path proxy
# ---------------------------------------------------------------------------
def run_trades(
    daily: pd.DataFrame,
    min_gap_pct: float = 0.25,
    max_gap_pct: float = 10.0,
    tp_R: float = 1.0,
    stop_R: float = 1.0,
    atr_n: int = 20,
    cost_bps: float = 2.0,
    directions: np.ndarray | None = None,
) -> pd.DataFrame:
    """Gap-fade symmetric barrier backtest on daily bars.

    For every session where ``|gap_pct|`` is in [``min_gap_pct``, ``max_gap_pct``]
    and ATR is available, a fade trade is opened at the **open**:

    - Direction ``d = -sign(gap_pct)`` (fade the gap -- go toward prior close).
    - Take-profit at ``tp_R * ATR(atr_n)`` in the trade direction.
    - Stop at ``stop_R * ATR(atr_n)`` in the adverse direction.
    - The day's high/low bracket is used as the path proxy.  A bar that hits both
      barriers is treated as a stop (conservative; order unknowable on daily bars).
    - End-of-day close if neither barrier is hit.

    ``directions`` overrides the trade direction (the random-control arm passes a +-1
    vector aligned to the entry rows, so the payoff is direction-symmetric and a coin
    earns approximately 0).  ``cost_bps`` is a one-way cost; net return subtracts
    2 * cost_bps (round trip).

    Columns: date, gap_pct, gap_size, dir, entry, tp_px, stop_px,
             exit, exit_reason, ret_gross, ret_net.
    """
    daily = daily.dropna(subset=["gap_pct"]).copy()
    daily["atr_val"] = atr(daily, atr_n)

    mask = (
        (daily["gap_pct"].abs() >= min_gap_pct) &
        (daily["gap_pct"].abs() <= max_gap_pct) &
        daily["atr_val"].notna()
    )
    entries = daily[mask].copy()

    dirs_arr = (
        np.asarray(directions, dtype=float)
        if directions is not None
        else -np.sign(entries["gap_pct"].to_numpy())
    )

    # --- vectorized barrier engine ------------------------------------------
    entry_px  = entries["open"].to_numpy()
    R_val     = entries["atr_val"].to_numpy()
    hi        = entries["high"].to_numpy()
    lo        = entries["low"].to_numpy()
    close_px  = entries["close"].to_numpy()
    gap_arr   = entries["gap_pct"].to_numpy()

    tp_px    = entry_px + dirs_arr * tp_R   * R_val
    stop_px  = entry_px - dirs_arr * stop_R * R_val

    # Hit conditions: long (d>0) hits TP if hi>=tp, stop if lo<=stop.
    hit_tp_long  = (dirs_arr > 0) & (hi >= tp_px)
    hit_tp_short = (dirs_arr < 0) & (lo <= tp_px)
    hit_sl_long  = (dirs_arr > 0) & (lo <= stop_px)
    hit_sl_short = (dirs_arr < 0) & (hi >= stop_px)

    hit_tp = hit_tp_long | hit_tp_short
    hit_sl = hit_sl_long | hit_sl_short

    # Conservative: stop wins when both triggered on same daily bar.
    both = hit_tp & hit_sl
    exit_px = np.where(
        both,      stop_px,
        np.where(hit_sl,  stop_px,
        np.where(hit_tp,  tp_px,
                 close_px))
    )
    exit_reason = np.where(both, "sl",
                  np.where(hit_sl, "sl",
                  np.where(hit_tp, "tp", "eod")))

    ret_gross = dirs_arr * (exit_px - entry_px) / entry_px

    out = pd.DataFrame({
        "date":        entries.index,
        "gap_pct":     gap_arr,
        "gap_size":    gap_size_bucket(pd.Series(gap_arr)).values,
        "dir":         dirs_arr.astype(int),
        "entry":       entry_px,
        "tp_px":       tp_px,
        "stop_px":     stop_px,
        "exit":        exit_px,
        "exit_reason": exit_reason,
        "ret_gross":   ret_gross,
        "ret_net":     ret_gross - 2.0 * cost_bps * 1e-4,
    })
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Trade-ledger summary
# ---------------------------------------------------------------------------
def summarize(ledger: pd.DataFrame, col: str = "ret_net") -> dict:
    """Headline per-trade statistics with a Newey-West HAC t-stat.

    Returns trade count, win-rate, mean return (bps/trade), per-trade Sharpe, P&L
    skew, and the HAC t-stat that drives the Signal verdict.  The HAC bandwidth
    uses the Newey-West (1994) automatic rule (4 * (n/100)^(2/9) lags) -- the same
    inline implementation as study 72 so the engine has no hard dependency on
    quantlab being importable in tests.
    """
    r = ledger[col].to_numpy(dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    out = {
        "n_trades": int(n),
        "win_rate": float((r > 0).mean()) if n else float("nan"),
        "mean_bps": float(r.mean() * 1e4) if n else float("nan"),
        "sharpe_per_trade": (
            float(r.mean() / r.std(ddof=1)) if n > 1 and r.std() > 0 else float("nan")
        ),
        "skew": float(pd.Series(r).skew()) if n > 2 else float("nan"),
        "tstat": float("nan"),
    }
    if n > 5:
        mu   = r.mean()
        e    = r - mu
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
        lrv  = float(e @ e) / n
        for k in range(1, lags + 1):
            w    = 1.0 - k / (lags + 1.0)
            lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
        se = np.sqrt(max(lrv, 0.0) / n)
        out["tstat"] = float(mu / se) if se > 0 else float("nan")
    return out
