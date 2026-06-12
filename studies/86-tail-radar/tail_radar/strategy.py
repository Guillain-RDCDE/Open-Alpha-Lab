"""The strategy and its honest controls — Study 86 (Tail-Radar).

The claim: the CBOE SKEW index is a 'black-swan radar' — when SKEW is high (put skew
expensive, fat-tail risk priced in), the market should subsequently *fall* (or at least
deliver worse-than-normal returns), because sophisticated players are hedging against a
crash.  We test this in three dimensions:

1. **SKEW quintile forward returns** — sort days into SKEW quintiles, compute forward
   1-day / 5-day / 21-day SPY log-returns for each bucket, and run a HAC t-stat on
   the *top quintile vs bottom quintile* spread.  The claim says top-SKEW days should
   predict negative forward returns; the null says they're flat.

2. **Tail of the distribution** — SKEW is specifically about *crashes*, so we ask the
   specific tail question: does the fraction of large-loss days (SPY < -2% over the
   next 21 calendar days) increase after high-SKEW readings?  This is a binary
   regime test, not a mean test.

3. **SKEW vs VIX incremental value** — run a panel regression of forward returns on
   standardised SKEW and standardised VIX to see if SKEW adds anything beyond VIX
   (a naively simpler fear gauge).  A t-stat near zero on the SKEW coefficient, while
   VIX carries a real coefficient, would show SKEW is just a noisy VIX.

No look-ahead: SKEW and VIX measured at the close of day *t* are used to rank/signal;
the forward return is the log-return from day *t+1* close to day *t+h* close.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Signal construction helpers
# ---------------------------------------------------------------------------

def skew_quintile(skew: pd.Series, window: int = 252, min_periods: int = 63) -> pd.Series:
    """Rolling SKEW quintile rank (1 = bottom 20%, 5 = top 20%).

    Ranks are formed strictly out-of-sample: the percentile at day *t* uses only
    history up to and including day *t*.  Returns integer quintile labels 1–5.
    """
    pct = skew.rolling(window, min_periods=min_periods).rank(pct=True)
    quintile = np.ceil(pct * 5).clip(1, 5).astype("Int64")
    return quintile


def forward_return(spy_close: pd.Series, horizon: int = 1) -> pd.Series:
    """Log forward return from day *t* close to day *t+horizon* close.

    Signal at day *t* uses SKEW/VIX known at the close of *t*; the forward
    return is ``log(close[t+horizon]) - log(close[t])``, aligned at day *t*.
    This is a *holding-period* return: buy at close *t*, exit at close *t+horizon*.
    The last ``horizon`` observations are NaN (no future data available).
    """
    log_close = np.log(spy_close)
    fwd = log_close.shift(-horizon) - log_close
    return fwd


# ---------------------------------------------------------------------------
# Quintile-sorted forward return table
# ---------------------------------------------------------------------------

def quintile_table(
    df: pd.DataFrame,
    horizons: list[int] | None = None,
    window: int = 252,
    min_periods: int = 63,
) -> pd.DataFrame:
    """Forward-return statistics for each SKEW quintile, across multiple horizons.

    Parameters
    ----------
    df : DataFrame with columns ``[SKEW, SPY_close]`` and a DatetimeIndex.
    horizons : list of forward-return horizons in calendar days (default [1, 5, 21]).
    window : rolling lookback for the quintile rank.
    min_periods : minimum observations before a rank is computed.

    Returns
    -------
    A DataFrame indexed by (quintile, horizon) with columns
    ``[n, mean_bps, std_bps, t_stat]``.
    """
    if horizons is None:
        horizons = [1, 5, 21]
    q = skew_quintile(df["SKEW"], window=window, min_periods=min_periods)
    rows = []
    for h in horizons:
        fwd = forward_return(df["SPY_close"], horizon=h)
        for qi in range(1, 6):
            mask = (q == qi) & fwd.notna()
            r = fwd[mask].to_numpy(dtype=float)
            n = r.size
            if n < 5:
                rows.append({"quintile": qi, "horizon": h, "n": n,
                             "mean_bps": np.nan, "std_bps": np.nan, "t_stat": np.nan})
                continue
            mean_bps = r.mean() * 1e4
            std_bps = r.std(ddof=1) * 1e4
            t_stat = _hac_tstat(r)
            rows.append({"quintile": qi, "horizon": h, "n": n,
                         "mean_bps": mean_bps, "std_bps": std_bps, "t_stat": t_stat})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Spread: top quintile − bottom quintile
# ---------------------------------------------------------------------------

def top_minus_bottom(
    df: pd.DataFrame,
    horizon: int = 1,
    window: int = 252,
    min_periods: int = 63,
) -> dict:
    """HAC t-stat on the spread: mean(forward_return | SKEW Q5) − mean(... | SKEW Q1).

    The decisive test of the claim: does the highest-SKEW quintile predict *worse*
    forward returns than the lowest-SKEW quintile?  Returns a dict with keys
    ``{n_top, n_bot, mean_top_bps, mean_bot_bps, spread_bps, t_spread,
       t_top, t_bot}``.
    """
    q = skew_quintile(df["SKEW"], window=window, min_periods=min_periods)
    fwd = forward_return(df["SPY_close"], horizon=horizon)
    r_top = fwd[(q == 5) & fwd.notna()].to_numpy(dtype=float)
    r_bot = fwd[(q == 1) & fwd.notna()].to_numpy(dtype=float)
    spread = np.concatenate([r_top, -r_bot])  # top − bot, sign-adjusted
    return {
        "n_top": int(r_top.size),
        "n_bot": int(r_bot.size),
        "mean_top_bps": float(r_top.mean() * 1e4) if r_top.size else np.nan,
        "mean_bot_bps": float(r_bot.mean() * 1e4) if r_bot.size else np.nan,
        "spread_bps": float((r_top.mean() - r_bot.mean()) * 1e4)
            if r_top.size and r_bot.size else np.nan,
        "t_spread": float(_hac_tstat(spread)) if spread.size > 5 else np.nan,
        "t_top": float(_hac_tstat(r_top)) if r_top.size > 5 else np.nan,
        "t_bot": float(_hac_tstat(r_bot)) if r_bot.size > 5 else np.nan,
    }


# ---------------------------------------------------------------------------
# Tail-risk analysis: crash-frequency after high SKEW
# ---------------------------------------------------------------------------

def crash_frequency(
    df: pd.DataFrame,
    horizon: int = 21,
    crash_threshold: float = -0.05,
    high_skew_pct: float = 0.80,
    window: int = 252,
    min_periods: int = 63,
) -> dict:
    """Does high SKEW predict an elevated frequency of large forward losses?

    'High SKEW' is defined as SKEW above its rolling ``high_skew_pct`` percentile;
    'crash' is a forward horizon-day log-return below ``crash_threshold`` (default -5%).
    Returns a dict with keys ``{n_high, n_low, crash_rate_high, crash_rate_low,
    rate_ratio, fisher_pvalue}``.
    """
    pct = df["SKEW"].rolling(window, min_periods=min_periods).rank(pct=True)
    high_mask = pct >= high_skew_pct
    fwd = forward_return(df["SPY_close"], horizon=horizon)
    valid = fwd.notna()

    r_high = fwd[high_mask & valid].to_numpy(dtype=float)
    r_low = fwd[~high_mask & valid].to_numpy(dtype=float)

    crash_high = int((r_high < crash_threshold).sum())
    crash_low = int((r_low < crash_threshold).sum())
    n_high = int(r_high.size)
    n_low = int(r_low.size)
    rate_high = crash_high / n_high if n_high else np.nan
    rate_low = crash_low / n_low if n_low else np.nan

    # Fisher exact test on the 2×2 contingency table.
    from scipy.stats import fisher_exact

    _, pval = fisher_exact(
        [[crash_high, n_high - crash_high], [crash_low, n_low - crash_low]],
        alternative="greater",
    )
    return {
        "n_high": n_high,
        "n_low": n_low,
        "crash_rate_high": float(rate_high),
        "crash_rate_low": float(rate_low),
        "rate_ratio": float(rate_high / rate_low) if rate_low else np.nan,
        "fisher_pvalue": float(pval),
    }


# ---------------------------------------------------------------------------
# SKEW vs VIX incremental regression
# ---------------------------------------------------------------------------

def skew_vs_vix_regression(
    df: pd.DataFrame,
    horizon: int = 1,
    window: int = 252,
    min_periods: int = 63,
) -> dict:
    """OLS of forward_return on standardised SKEW and VIX, with HAC inference.

    Both predictors are standardised (z-scored over the rolling window) so their
    coefficients are directly comparable.  Returns a dict with keys
    ``{n, beta_skew, t_skew, beta_vix, t_vix, r2}``.

    A tiny ``beta_skew`` and small ``|t_skew|`` while ``|t_vix|`` is large would mean
    SKEW adds nothing over VIX — consistent with the 'Mirage' verdict.
    """
    fwd = forward_return(df["SPY_close"], horizon=horizon)

    # Rolling z-scores (out-of-sample standardisation).
    skew_z = (
        (df["SKEW"] - df["SKEW"].rolling(window, min_periods=min_periods).mean())
        / df["SKEW"].rolling(window, min_periods=min_periods).std(ddof=1)
    )
    vix_z = (
        (df["VIX"] - df["VIX"].rolling(window, min_periods=min_periods).mean())
        / df["VIX"].rolling(window, min_periods=min_periods).std(ddof=1)
    )

    combined = pd.DataFrame({"fwd": fwd, "skew_z": skew_z, "vix_z": vix_z}).dropna()
    n = len(combined)
    if n < 30:
        return {"n": n, "beta_skew": np.nan, "t_skew": np.nan,
                "beta_vix": np.nan, "t_vix": np.nan, "r2": np.nan}

    y = combined["fwd"].to_numpy()
    X = np.column_stack([np.ones(n), combined["skew_z"].to_numpy(),
                         combined["vix_z"].to_numpy()])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    yhat = X @ beta
    resid = y - yhat
    ss_tot = np.sum((y - y.mean()) ** 2)
    ss_res = np.sum(resid ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else np.nan

    # HAC standard errors (Newey-West) for each coefficient.
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    XtX_inv = np.linalg.inv(X.T @ X)
    Xe = X * resid[:, None]
    S = Xe.T @ Xe / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        cov_k = Xe[k:].T @ Xe[:-k] / n
        S += w * (cov_k + cov_k.T)
    var_beta = n * XtX_inv @ S @ XtX_inv
    se = np.sqrt(np.diag(var_beta))

    return {
        "n": n,
        "beta_skew": float(beta[1]),
        "t_skew": float(beta[1] / se[1]) if se[1] > 0 else np.nan,
        "beta_vix": float(beta[2]),
        "t_vix": float(beta[2] / se[2]) if se[2] > 0 else np.nan,
        "r2": r2,
    }


# ---------------------------------------------------------------------------
# Summarise a return series with HAC inference (shared with notebooks)
# ---------------------------------------------------------------------------

def summarize(returns: np.ndarray | pd.Series, label: str = "") -> dict:
    """Headline statistics for any return series, with HAC t-stat.

    Returns a dict with keys ``{label, n, mean_bps, std_bps, sharpe, skew, t_stat}``.
    """
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    n = r.size
    out = {
        "label": label,
        "n": int(n),
        "mean_bps": float(r.mean() * 1e4) if n else np.nan,
        "std_bps": float(r.std(ddof=1) * 1e4) if n > 1 else np.nan,
        "sharpe": float(r.mean() / r.std(ddof=1)) if n > 1 and r.std() > 0 else np.nan,
        "skew": float(pd.Series(r).skew()) if n > 2 else np.nan,
        "t_stat": float(_hac_tstat(r)) if n > 5 else np.nan,
    }
    return out


# ---------------------------------------------------------------------------
# Internal: Newey–West HAC t-stat
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
