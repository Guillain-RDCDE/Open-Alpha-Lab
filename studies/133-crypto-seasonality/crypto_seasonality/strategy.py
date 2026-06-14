"""The strategy and its honest controls — Study 133 (Crypto-Seasonality).

The folk recipe: some calendar months are structurally good or bad for Bitcoin.
"Uptober" (October bullish), "Rektember" (September bearish), and the Q1 rally are
the most-cited variants. We test *all 12 months impartially* against a single honest
baseline: the buy-and-hold monthly return distribution (what you'd get by picking
months at random). Separately, we run a seasonal long/flat rule — long only in the
allegedly good months — and compare it to buy-and-hold.

The three inference hurdles that decide the verdict:

1. **HAC t-stat on each month's mean return** vs the all-month average. With ~11
   observations per month the individual t-stats will be very low-powered; we say so.
2. **Multiple-comparisons caveat**: testing 12 months means the best-looking month has
   ~50% chance of clearing |t| >= 2 purely by chance (family-wise error rate). We
   apply a Bonferroni-corrected threshold (|t| >= 3.1 for alpha=0.05/12) and note it.
3. **Long/flat rule vs buy-and-hold**: cumulative returns, annualised Sharpe, HAC
   t-stat on the long/flat minus B&H monthly spread — the only fair tradability test.

No look-ahead: the month label is known from the calendar (the signal is "it is
currently October"). The trade enters at the first day of the month. For the long/flat
strategy we form the signal at the end of month *m-1* (the month is known in advance)
and enter at the close of the last day of *m-1* / open of the first day of *m*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import MONTH_NAMES


# ---------------------------------------------------------------------------
# Monthly return table
# ---------------------------------------------------------------------------
def monthly_returns(bars: pd.DataFrame, col: str = "close") -> pd.DataFrame:
    """Compute monthly log-returns from a daily OHLCV tape.

    Each month's return is log(last_close / first_open_of_month). Using open of the
    first day as the entry price avoids look-ahead (we could not have bought at the
    prior close without knowing the future).

    Returns a DataFrame indexed by (year, month) with columns:
    ``year, month, month_name, log_ret, pct_ret, n_days``.
    """
    bars = bars.copy()
    bars["ym"] = bars.index.to_series().dt.to_period("M")

    rows = []
    for ym, grp in bars.groupby("ym"):
        grp = grp.sort_index()
        entry_px = grp["open"].iloc[0]   # first day's open — the no-look-ahead entry
        exit_px = grp[col].iloc[-1]      # last day's close
        if entry_px <= 0 or exit_px <= 0:
            continue
        log_r = np.log(exit_px / entry_px)
        rows.append(
            {
                "year": ym.year,
                "month": ym.month,
                "month_name": MONTH_NAMES[ym.month - 1],
                "log_ret": log_r,
                "pct_ret": np.expm1(log_r),
                "n_days": len(grp),
            }
        )
    return pd.DataFrame(rows).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Per-month statistics and HAC t-stats
# ---------------------------------------------------------------------------
def _hac_tstat(r: np.ndarray, baseline_mean: float = 0.0) -> float:
    """HAC (Newey-West) t-stat for the mean of ``r`` relative to ``baseline_mean``."""
    x = r - baseline_mean
    n = x.size
    if n < 2:
        return float("nan")
    mu = x.mean()
    e = x - mu
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, min(lags + 1, n)):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")


def month_stats(mret: pd.DataFrame) -> pd.DataFrame:
    """Per-month summary: mean, std, median, n, HAC t-stat vs the all-month mean.

    The baseline for each month's t-stat is the pooled all-month mean log-return.
    A month "beats the baseline" only if its t-stat clears the Bonferroni-corrected
    threshold for 12 simultaneous tests (|t| >= ~3.1 for alpha=0.05/12).

    Returns a DataFrame with one row per calendar month (1..12).
    """
    all_mean = mret["log_ret"].mean()  # the all-month baseline
    rows = []
    for m in range(1, 13):
        sub = mret[mret["month"] == m]["log_ret"].to_numpy()
        n = len(sub)
        t = _hac_tstat(sub, baseline_mean=all_mean) if n >= 2 else float("nan")
        rows.append(
            {
                "month": m,
                "month_name": MONTH_NAMES[m - 1],
                "n": n,
                "mean_pct": float(np.expm1(sub.mean())) * 100 if n else float("nan"),
                "mean_log": float(sub.mean()) if n else float("nan"),
                "std_log": float(sub.std(ddof=1)) if n > 1 else float("nan"),
                "median_pct": float(np.expm1(np.median(sub))) * 100 if n else float("nan"),
                "tstat_vs_baseline": t,
            }
        )
    df = pd.DataFrame(rows)
    # Bonferroni threshold for 12 tests: p < 0.05/12 ≈ p < 0.00417
    # Approximate t critical at df≈10, two-sided: ~3.17
    df["bonferroni_sig"] = df["tstat_vs_baseline"].abs() >= 3.1
    df["naive_sig"] = df["tstat_vs_baseline"].abs() >= 2.0
    return df


# ---------------------------------------------------------------------------
# Seasonal long/flat rule vs buy-and-hold
# ---------------------------------------------------------------------------
def seasonal_strategy(
    mret: pd.DataFrame,
    long_months: list[int] | None = None,
) -> pd.DataFrame:
    """Monthly P&L of a long/flat rule vs buy-and-hold.

    In ``long_months`` (default: {10} for "Uptober only") the strategy holds BTC; in
    all other months it stays flat (cash). Buy-and-hold holds every month.

    Returns a DataFrame with columns:
    ``year, month, bh_log_ret, strategy_log_ret, diff`` (strategy minus B&H).
    """
    if long_months is None:
        long_months = [10]  # "Uptober" as the canonical test case
    long_set = set(long_months)

    df = mret[["year", "month", "log_ret"]].copy().rename(columns={"log_ret": "bh_log_ret"})
    df["in_long"] = df["month"].isin(long_set)
    df["strategy_log_ret"] = df["bh_log_ret"] * df["in_long"].astype(float)
    df["diff"] = df["strategy_log_ret"] - df["bh_log_ret"]
    return df.reset_index(drop=True)


def summarize(returns: np.ndarray | pd.Series, label: str = "") -> dict:
    """Headline statistics for a monthly return series.

    Returns n, mean (%), annualised Sharpe, and HAC t-stat on the mean. The Sharpe
    is annualised by sqrt(12) (monthly data). With ~11 years of data the t-stat is
    the decisive inference-bar number.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    mean = float(r.mean()) if n else float("nan")
    std = float(r.std(ddof=1)) if n > 1 else float("nan")
    sharpe_ann = float(mean / std * np.sqrt(12)) if std > 0 else float("nan")

    t = _hac_tstat(r, baseline_mean=0.0) if n >= 2 else float("nan")
    return {
        "label": label,
        "n_months": int(n),
        "mean_log_ret": mean,
        "mean_pct": float(np.expm1(mean)) * 100 if not np.isnan(mean) else float("nan"),
        "std_log_ret": std,
        "sharpe_ann": sharpe_ann,
        "tstat": t,
    }


def bh_vs_strategy(mret: pd.DataFrame, long_months: list[int] | None = None) -> dict:
    """Compare seasonal long/flat vs buy-and-hold, both summarised via ``summarize``.

    Returns a dict with 'bh' and 'strategy' sub-dicts, plus 'diff' (t-stat on the
    monthly spread = strategy − B&H, which must clear |t| >= 2 to claim an edge).
    """
    if long_months is None:
        long_months = [10]
    df = seasonal_strategy(mret, long_months=long_months)
    bh = summarize(df["bh_log_ret"].to_numpy(), label="buy_and_hold")
    strat = summarize(df["strategy_log_ret"].to_numpy(), label="seasonal_long_flat")
    diff_t = _hac_tstat(df["diff"].to_numpy(), baseline_mean=0.0)
    return {
        "bh": bh,
        "strategy": strat,
        "long_months": sorted(long_months),
        "diff_tstat": diff_t,
        "fraction_months_invested": len(long_months) / 12.0,
    }
