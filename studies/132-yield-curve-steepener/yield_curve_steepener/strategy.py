"""The strategy and its honest controls — Study 132 (Yield-Curve-Steepener).

The claim: the slope of the Treasury yield curve (10Y minus 3M, proxied either by
^TNX minus ^IRX or by relative ETF performance of TLT vs SHY/IEF) predicts forward
TLT returns.  A steep curve (positive slope) signals that long bonds are fairly
compensated; an inverted curve signals that long bonds may be expensive relative to
short-term rates and thus the long end may underperform.  A steepener timing rule
— long TLT only when the curve is steep, flat otherwise — is tested against an
honest buy-and-hold baseline.

Three complementary tests share one inference engine:

1. **Quintile-sorted forward TLT returns** — divide all days into five slope-quintiles
   (out-of-sample rolling rank) and measure forward 1-day, 5-day, and 21-day TLT
   log-returns in each bucket.  The decisive test: Q5 (steepest curve) − Q1
   (most inverted) spread, HAC t-stat.

2. **Binary timing overlay** — long TLT when the prior-day slope > threshold (default 0),
   flat (cash) otherwise, vs. buy-and-hold TLT.  The active-minus-passive daily spread
   is the edge to test.  Costs charged on each regime switch.

3. **Regime statistics** — compare TLT statistics across full-sample, steep-curve, and
   flat/inverted-curve regimes.  A real effect shows materially better Sharpe in the
   steep sub-sample.

No look-ahead: slope at close of day *t* forms the signal; the forward return starts
at the close of day *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Signal construction helpers
# ---------------------------------------------------------------------------

def slope_quintile(
    slope: pd.Series,
    window: int = 252,
    min_periods: int = 63,
) -> pd.Series:
    """Rolling slope quintile rank (1 = most inverted, 5 = steepest).

    Ranks are formed strictly out-of-sample: the percentile at day *t* uses only
    history up to and including day *t*.  Returns integer quintile labels 1–5
    (as a nullable integer Series to handle NaN warmup).
    """
    pct = slope.rolling(window, min_periods=min_periods).rank(pct=True)
    quintile = np.ceil(pct * 5).clip(1, 5).astype("Int64")
    return quintile


def forward_return(tlt_close: pd.Series, horizon: int = 1) -> pd.Series:
    """Log forward return from day *t* close to day *t+horizon* close.

    Signal at day *t* uses the slope known at the close of *t*; the forward return is
    ``log(close[t+horizon]) - log(close[t])``, aligned at day *t*.  This is a
    holding-period return: enter at close *t*, exit at close *t+horizon*.  The last
    ``horizon`` observations are NaN (no future data available).
    """
    log_close = np.log(tlt_close)
    return log_close.shift(-horizon) - log_close


def steep_mask(slope: pd.Series, threshold: float = 0.0) -> pd.Series:
    """Boolean mask: True when the curve is steep (slope > threshold)."""
    return slope > threshold


# ---------------------------------------------------------------------------
# Quintile-sorted forward return table
# ---------------------------------------------------------------------------

def quintile_table(
    df: pd.DataFrame,
    horizons: list[int] | None = None,
    window: int = 252,
    min_periods: int = 63,
) -> pd.DataFrame:
    """Forward-return statistics for each slope quintile, across multiple horizons.

    Parameters
    ----------
    df : DataFrame with columns ``[slope, TLT_close]`` and a DatetimeIndex.
    horizons : list of forward-return horizons in business days (default [1, 5, 21]).
    window : rolling lookback for the quintile rank.
    min_periods : minimum observations before a rank is computed.

    Returns
    -------
    A DataFrame with columns ``[quintile, horizon, n, mean_bps, std_bps, t_stat]``.
    """
    if horizons is None:
        horizons = [1, 5, 21]
    q = slope_quintile(df["slope"], window=window, min_periods=min_periods)
    rows = []
    for h in horizons:
        fwd = forward_return(df["TLT_close"], horizon=h)
        for qi in range(1, 6):
            mask = (q == qi) & fwd.notna()
            r = fwd[mask].to_numpy(dtype=float)
            n = r.size
            if n < 5:
                rows.append({"quintile": qi, "horizon": h, "n": n,
                             "mean_bps": np.nan, "std_bps": np.nan, "t_stat": np.nan})
                continue
            rows.append({
                "quintile": qi,
                "horizon": h,
                "n": n,
                "mean_bps": float(r.mean() * 1e4),
                "std_bps": float(r.std(ddof=1) * 1e4),
                "t_stat": float(_hac_tstat(r)),
            })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Spread: top quintile − bottom quintile (the decisive test)
# ---------------------------------------------------------------------------

def top_minus_bottom(
    df: pd.DataFrame,
    horizon: int = 1,
    window: int = 252,
    min_periods: int = 63,
) -> dict:
    """HAC t-stat on the spread: mean(fwd_TLT | Q5 steep) − mean(fwd_TLT | Q1 inverted).

    This is the central test of the claim.  If the yield-curve slope predicts TLT
    returns, Q5 (steepest curve) should deliver better forward TLT returns than Q1
    (most inverted).  Returns a dict with keys
    ``{n_top, n_bot, mean_top_bps, mean_bot_bps, spread_bps, t_spread, t_top, t_bot}``.
    """
    q = slope_quintile(df["slope"], window=window, min_periods=min_periods)
    fwd = forward_return(df["TLT_close"], horizon=horizon)
    r_top = fwd[(q == 5) & fwd.notna()].to_numpy(dtype=float)
    r_bot = fwd[(q == 1) & fwd.notna()].to_numpy(dtype=float)
    spread = np.concatenate([r_top, -r_bot])  # sign-adjusted: steep minus inverted
    return {
        "n_top": int(r_top.size),
        "n_bot": int(r_bot.size),
        "mean_top_bps": float(r_top.mean() * 1e4) if r_top.size else np.nan,
        "mean_bot_bps": float(r_bot.mean() * 1e4) if r_bot.size else np.nan,
        "spread_bps": float((r_top.mean() - r_bot.mean()) * 1e4)
            if (r_top.size and r_bot.size) else np.nan,
        "t_spread": float(_hac_tstat(spread)) if spread.size > 5 else np.nan,
        "t_top": float(_hac_tstat(r_top)) if r_top.size > 5 else np.nan,
        "t_bot": float(_hac_tstat(r_bot)) if r_bot.size > 5 else np.nan,
    }


# ---------------------------------------------------------------------------
# Binary timing overlay: long TLT when steep, flat otherwise
# ---------------------------------------------------------------------------

def timing_overlay(
    df: pd.DataFrame,
    threshold: float = 0.0,
    cost_bps: float = 2.0,
) -> pd.DataFrame:
    """Daily return of the curve-steepener timer vs unconditional buy-and-hold TLT.

    The timer is 'risk-on' (long TLT) when the prior-day slope is above ``threshold``,
    and flat (cash) when the curve is flat or inverted.  The baseline is always long
    TLT (buy-and-hold).  The 'active minus passive' spread is the edge to test.
    ``cost_bps`` is charged on each *switch* (steep→flat or flat→steep) as a round-trip.

    Returns a DataFrame with columns ``[r_passive, r_active, r_spread, is_steep, switched]``
    aligned on day *t+1* (the day returns are earned).
    """
    fwd1 = forward_return(df["TLT_close"], horizon=1)
    # shift(1) lags the slope signal by one day (no look-ahead)
    signal_raw = steep_mask(df["slope"], threshold=threshold).shift(1)
    signal = signal_raw.astype("boolean")
    switch = signal_raw.ne(signal_raw.shift(1)) & signal_raw.notna() & signal_raw.shift(1).notna()

    r_passive = fwd1
    r_active = fwd1.where(signal.fillna(False), 0.0) - switch.astype(float) * cost_bps * 1e-4

    out = pd.DataFrame({
        "r_passive": r_passive,
        "r_active": r_active,
        "r_spread": r_active - r_passive,
        "is_steep": signal,
        "switched": switch,
    }).dropna(subset=["r_passive", "r_active"])
    return out


# ---------------------------------------------------------------------------
# Regime statistics (steep vs flat/inverted curve)
# ---------------------------------------------------------------------------

def regime_stats(df: pd.DataFrame, threshold: float = 0.0) -> pd.DataFrame:
    """Compare TLT statistics across full-sample, steep-curve, and flat/inverted regimes.

    The signal is the lagged slope: regime on day *t* is determined by the prior day's
    slope (no look-ahead).  Returns a DataFrame indexed by regime with columns
    ``[n, n_pct, mean_bps, std_bps, sharpe_ann, t_stat]``.
    """
    fwd1 = forward_return(df["TLT_close"], horizon=1)
    signal = steep_mask(df["slope"], threshold=threshold).shift(1).astype("boolean")
    valid = fwd1.notna() & signal.notna()

    rows = []
    for label, mask in [
        ("full sample", valid),
        ("steep curve (slope>0)", valid & signal.fillna(False)),
        ("flat/inverted (slope<=0)", valid & ~signal.fillna(True)),
    ]:
        r = fwd1[mask].to_numpy(dtype=float)
        n = r.size
        mean_bps = float(r.mean() * 1e4) if n else np.nan
        std_bps = float(r.std(ddof=1) * 1e4) if n > 1 else np.nan
        sharpe_ann = float(r.mean() / r.std(ddof=1) * np.sqrt(252)) if n > 1 and r.std() > 0 else np.nan
        t_stat = float(_hac_tstat(r)) if n > 5 else np.nan
        rows.append({
            "regime": label,
            "n": n,
            "n_pct": float(n / valid.sum() * 100) if valid.sum() > 0 else np.nan,
            "mean_bps": mean_bps,
            "std_bps": std_bps,
            "sharpe_ann": sharpe_ann,
            "t_stat": t_stat,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Summarise any return series with HAC inference
# ---------------------------------------------------------------------------

def summarize(returns: "np.ndarray | pd.Series", label: str = "") -> dict:
    """Headline statistics for any return series, with HAC t-stat.

    Returns a dict with keys ``{label, n, mean_bps, std_bps, sharpe_ann, skew, t_stat}``.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    out = {
        "label": label,
        "n": int(n),
        "mean_bps": float(r.mean() * 1e4) if n else np.nan,
        "std_bps": float(r.std(ddof=1) * 1e4) if n > 1 else np.nan,
        "sharpe_ann": float(r.mean() / r.std(ddof=1) * np.sqrt(252)) if n > 1 and r.std() > 0 else np.nan,
        "skew": float(pd.Series(r).skew()) if n > 2 else np.nan,
        "t_stat": float(_hac_tstat(r)) if n > 5 else np.nan,
    }
    return out


# ---------------------------------------------------------------------------
# Internal: Newey-West HAC t-stat
# ---------------------------------------------------------------------------

def _hac_tstat(r: np.ndarray) -> float:
    """Newey-West HAC t-statistic for H0: mean(r) = 0."""
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
