"""Strategy and honest controls — Study 119 (Real-Rate-Regime).

The folk recipe: "Don't fight the Fed — when the real long rate is high or rising,
step aside; when it falls, be in." We implement three tests and pin each against the
only comparison that decides whether the rule does anything: **buy-and-hold (BaH)**.

Three sub-tests:

1. **Level sort.** Divide the historical real-rate distribution into four quartiles;
   compare the forward 12-month real return across bins. If the "high rate → bad
   for equities" story is right, Q4 (highest rates) should show the lowest returns.

2. **Momentum sort.** Split months into "rising" (12m change > 0) and "falling"
   (12m change ≤ 0) rate regimes; compare forward 12-month returns. The claim
   predicts rising → lower returns.

3. **Timing overlay.** A mechanical month-by-month rule: be *in* the market (=BaH
   monthly real log-return) when the 12-month real-rate change was ≤ 0 in the prior
   month; be in *cash* (= 0 return) when it was rising. Compare its realised Sharpe
   and mean return to buy-and-hold. No look-ahead: the signal at month *t* is the
   real-rate change known through *t*; the return is earned at *t+1*.

The Newey-West HAC t-statistic is used throughout for inference, so overlapping
12-month forward windows (which induce serial correlation) do not overstate
significance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Level sort — forward return by real-rate quartile
# ---------------------------------------------------------------------------
def level_sort(
    df: pd.DataFrame,
    n_buckets: int = 4,
    fwd_col: str = "fwd12_real",
    rr_col: str = "real_rate",
) -> pd.DataFrame:
    """Forward 12-month real return sorted by the level of the real rate.

    Each observation is assigned to a real-rate quartile (Q1 = cheapest money,
    Q4 = most expensive). Returns a DataFrame with the mean forward return and
    count per bucket. The "Don't fight the Fed" thesis predicts a monotone
    *descending* pattern: Q1 best, Q4 worst.
    """
    df = df.dropna(subset=[rr_col, fwd_col]).copy()
    labels = [f"Q{i+1}" for i in range(n_buckets)]
    df["bucket"] = pd.qcut(df[rr_col], n_buckets, labels=labels)
    out = (
        df.groupby("bucket", observed=True)[fwd_col]
        .agg(["mean", "count", "std"])
        .rename(columns={"mean": "fwd_mean", "count": "n", "std": "fwd_std"})
    )
    out["fwd_se"] = out["fwd_std"] / np.sqrt(out["n"])
    return out


# ---------------------------------------------------------------------------
# Momentum sort — rising vs falling regime
# ---------------------------------------------------------------------------
def momentum_sort(
    df: pd.DataFrame,
    fwd_col: str = "fwd12_real",
    chg_col: str = "real_rate_chg",
) -> pd.DataFrame:
    """Forward 12-month real return by real-rate direction (rising vs. falling).

    A row is "rising" when the 12-month real-rate change is positive. Returns a
    DataFrame with mean forward return and count for each regime. The claim predicts
    the rising regime delivers lower returns.
    """
    df = df.dropna(subset=[chg_col, fwd_col]).copy()
    df["regime"] = np.where(df[chg_col] > 0, "rising", "falling")
    out = (
        df.groupby("regime")[fwd_col]
        .agg(["mean", "count", "std"])
        .rename(columns={"mean": "fwd_mean", "count": "n", "std": "fwd_std"})
    )
    out["fwd_se"] = out["fwd_std"] / np.sqrt(out["n"])
    return out.loc[["falling", "rising"]]  # canonical order


# ---------------------------------------------------------------------------
# Timing overlay — mechanical risk-off rule vs buy-and-hold
# ---------------------------------------------------------------------------
def timing_overlay(
    df: pd.DataFrame,
    chg_col: str = "real_rate_chg",
    ret_col: str = "monthly_real_ret",
    lag: int = 1,
) -> pd.DataFrame:
    """Monthly timing rule: risk-off (cash) when real rates were rising last month.

    Signal is formed from the 12-month change in real rate at month *t − lag*; the
    return earned is the real S&P log-return at month *t*. ``lag=1`` (default) is the
    minimum no-look-ahead offset. Returns a DataFrame with columns:

        - ``bh_ret``    : buy-and-hold monthly log-return
        - ``strat_ret`` : timing strategy return (signal × bh_ret)
        - ``signal``    : 1 = in the market, 0 = in cash (risk-off)

    Both series are in log-return units so they compound additively.
    """
    df = df.copy()
    # Apply the no-look-ahead lag
    df["signal"] = (df[chg_col].shift(lag) <= 0).astype(float)
    df["strat_ret"] = df["signal"] * df[ret_col]
    df["bh_ret"] = df[ret_col]
    return df[["signal", "strat_ret", "bh_ret"]].dropna()


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def summarize(returns: pd.Series, periods_per_year: int = 12) -> dict:
    """Headline stats for a monthly return series (log-returns in).

    Returns mean (annualised), std, Sharpe (annualised), and a Newey-West HAC
    t-statistic on the mean. The lag count uses the standard rule-of-thumb
    ``floor(4 * (n/100)^(2/9))``, appropriate for serially-correlated monthly data.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    if n < 2:
        return {k: float("nan") for k in ("n", "mean_ann", "std_ann", "sharpe", "tstat")}

    mu = r.mean()
    sd = r.std(ddof=1)
    sharpe = mu / sd * np.sqrt(periods_per_year) if sd > 0 else float("nan")
    mean_ann = mu * periods_per_year

    # Newey-West HAC on the mean (handles overlapping windows / autocorrelation)
    e = r - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    tstat = float(mu / se) if se > 0 else float("nan")

    return {
        "n": int(n),
        "mean_ann": float(mean_ann),
        "std_ann": float(sd * np.sqrt(periods_per_year)),
        "sharpe": float(sharpe),
        "tstat": float(tstat),
    }


def diff_tstat(r1: pd.Series, r2: pd.Series) -> float:
    """Newey-West HAC t-statistic on the mean of the *difference* (r1 − r2).

    Used to test whether two time series have a significantly different mean —
    e.g. the timing strategy vs buy-and-hold. Both series must be aligned on the
    same index; differences are formed pair-wise.
    """
    d = np.asarray(r1, dtype=float) - np.asarray(r2, dtype=float)
    d = d[np.isfinite(d)]
    n = d.size
    if n < 6:
        return float("nan")
    mu = d.mean()
    e = d - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")
