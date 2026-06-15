"""Strategy and analysis layer for Study 162 (Rosh-Hashanah).

The folk adage: "Sell Rosh Hashanah, buy Yom Kippur" — exit equities at the close before
Rosh Hashanah, re-enter at the close on Yom Kippur.  The ~8-day window is supposedly weak
for stocks, perhaps because Jewish traders (disproportionately represented among institutional
participants) reduce risk before solemn holidays.

We test it honestly against two controls:

1. **Matched random windows** — for each holiday window we draw 40 non-overlapping random
   8-day (trading-day count) windows from the *same calendar month ± 3 weeks*, measure the
   S&P return, and ask whether the holiday window is *worse* than the random draw.  HAC
   t-stat on the difference.

2. **Unconditional baseline** — is the holiday window return negative on average, or merely
   smaller than the rest of the year?  A window that earns +0.3% vs +0.5% unconditional is
   *not* a "sell" signal.

Both controls share the same :func:`summarize` interface.  The inference bar: |t| >= 2 on
the real tape with ~40 usable observations (1980–2024).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Matched random-window baseline
# ---------------------------------------------------------------------------
def random_windows(
    prices: pd.DataFrame,
    holiday_windows_df: pd.DataFrame,
    n_draws: int = 40,
    seed: int = 162,
) -> pd.DataFrame:
    """Draw matched random windows for comparison with each holiday window.

    For each holiday year, we draw ``n_draws`` windows of the same trading-day length as the
    corresponding holiday window, starting from a random trading day within ±21 calendar days
    of the Rosh Hashanah date (but not overlapping the holiday window itself).  The median
    return across the draws is assigned as the "random baseline" for that year.

    Returns a DataFrame indexed by year with columns:
    - ``rand_median``  — median of the sampled random window returns.
    - ``rand_mean``    — mean across all draws for that year.
    - ``rand_std``     — std across draws (within-year uncertainty).
    - ``n_draws``      — number of valid draws obtained.
    """
    rng = np.random.default_rng(seed)
    close = prices["close"].sort_index()
    trading_days = np.array(close.index)

    rows = []
    for year, hw_row in holiday_windows_df.iterrows():
        n_td = max(int(hw_row["n_trading"]), 1)
        entry_date = pd.Timestamp(hw_row["entry_date"])
        exit_date = pd.Timestamp(hw_row["exit_date"])
        rh = pd.Timestamp(hw_row["rosh_hashanah"])

        # Candidate starts: trading days within a ±21-day neighbourhood of RH
        # but not overlapping [entry_date, exit_date].
        lo = rh - pd.Timedelta(days=21)
        hi = rh + pd.Timedelta(days=21)
        candidates = trading_days[
            (trading_days >= lo) & (trading_days <= hi)
        ]
        # exclude days that would put the window in the holiday period
        candidates = [
            d for d in candidates
            if not (entry_date <= d <= exit_date)
            and not (entry_date <= trading_days[
                min(list(trading_days).index(d) + n_td, len(trading_days) - 1)
            ] <= exit_date)
        ]

        if len(candidates) < 3:
            continue

        rets = []
        for _ in range(n_draws):
            start = rng.choice(candidates)
            idx_start = list(trading_days).index(start)
            idx_end = idx_start + n_td
            if idx_end >= len(trading_days):
                continue
            end = trading_days[idx_end]
            ret = close.loc[end] / close.loc[start] - 1.0
            rets.append(float(ret))

        if len(rets) < 3:
            continue

        rows.append(
            {
                "year": int(year),
                "rand_median": float(np.median(rets)),
                "rand_mean": float(np.mean(rets)),
                "rand_std": float(np.std(rets, ddof=1)),
                "n_draws": int(len(rets)),
            }
        )

    return pd.DataFrame(rows).set_index("year")


# ---------------------------------------------------------------------------
# HAC t-stat (Newey-West) — local copy so the engine is self-contained
# ---------------------------------------------------------------------------
def _hac_tstat(x: np.ndarray) -> float:
    """Newey-West HAC t-stat for the mean of a 1-D array.

    Falls back to plain OLS SE when n < 6.
    """
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 2:
        return float("nan")
    mu = x.mean()
    e = x - mu
    if n < 6:
        se = x.std(ddof=1) / np.sqrt(n)
        return float(mu / se) if se > 0 else float("nan")
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


# ---------------------------------------------------------------------------
# Per-window timing P&L (the "trade" version)
# ---------------------------------------------------------------------------
def timing_pnl(
    holiday_windows_df: pd.DataFrame,
    prices: pd.DataFrame,
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """Simulate the "Sell Rosh Hashanah, buy Yom Kippur" timing trade, net of costs.

    For each holiday year:
    - **Signal arm**: short (or cash) during the window — captures the *negative* of the
      holiday return as a gain.  Net return = ``-window_ret - cost_bps * 2 * 1e-4``.
    - **Buy-and-hold** over the same window: ``window_ret``.

    Returns a DataFrame indexed by year with columns ``signal_ret``, ``bah_ret``,
    ``cost``, ``signal_net``.
    """
    rows = []
    for year, row in holiday_windows_df.iterrows():
        w = float(row["window_ret"])
        cost = 2 * cost_bps * 1e-4  # round trip
        rows.append(
            {
                "year": int(year),
                "bah_ret": w,
                "signal_ret": -w,  # short the window
                "cost": cost,
                "signal_net": -w - cost,
            }
        )
    return pd.DataFrame(rows).set_index("year")


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def summarize(windows: pd.DataFrame, col: str = "window_ret") -> dict:
    """Headline statistics for the holiday-window return series.

    - ``n``           — number of observations.
    - ``mean_bps``    — mean return in basis points.
    - ``win_rate``    — fraction of windows with negative return (for the "sell" framing).
    - ``tstat``       — HAC t-stat on the mean.
    - ``pct_negative``— fraction of windows with negative return.
    - ``median_bps``  — median in bps.
    """
    r = windows[col].dropna().to_numpy(dtype=float)
    n = len(r)
    if n == 0:
        return {k: float("nan") for k in ("n", "mean_bps", "win_rate", "tstat", "pct_negative", "median_bps")}
    return {
        "n": int(n),
        "mean_bps": float(r.mean() * 1e4),
        "win_rate": float((r > 0).mean()),   # fraction of POSITIVE windows
        "pct_negative": float((r < 0).mean()),
        "median_bps": float(np.median(r) * 1e4),
        "tstat": float(_hac_tstat(r)),
    }


def compare_holiday_vs_random(
    holiday_windows_df: pd.DataFrame,
    random_windows_df: pd.DataFrame,
) -> dict:
    """Difference between the holiday-window return and the matched random baseline.

    Aligns the two series by year and computes the HAC t-stat on the per-year gap
    ``holiday_ret - rand_median``.  If this t-stat is significantly negative (|t| > 2
    in the negative direction) we have evidence the holiday window is weaker than an
    average same-month window.
    """
    merged = holiday_windows_df[["window_ret"]].join(
        random_windows_df[["rand_median"]], how="inner"
    )
    gap = (merged["window_ret"] - merged["rand_median"]).to_numpy(dtype=float)
    n = len(gap)
    return {
        "n": int(n),
        "mean_gap_bps": float(gap.mean() * 1e4) if n else float("nan"),
        "tstat_gap": float(_hac_tstat(gap)) if n >= 2 else float("nan"),
        "pct_holiday_worse": float((gap < 0).mean()) if n else float("nan"),
    }
