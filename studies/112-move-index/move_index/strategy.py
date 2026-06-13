"""The strategy and its honest controls — Study 112 (Move-Index).

The claim: the MOVE index (ICE BofA bond-market implied volatility) is a cross-asset
risk gauge for equities — when MOVE is high or rising, bond markets are pricing
uncertainty and equity drawdowns should follow.  We test this in three dimensions:

1. **MOVE quintile forward returns** — sort days into MOVE quintiles (by rolling
   percentile rank), compute forward 1-day / 5-day / 21-day SPY log-returns for each
   bucket, and run a HAC t-stat on the *top quintile vs bottom quintile* spread.  The
   claim says top-MOVE days should predict worse forward equity returns; the null says
   they're flat.

2. **MOVE vs VIX incremental value** — run an OLS regression of forward returns on
   standardised MOVE and standardised VIX (both z-scored rolling OOS) to see if MOVE
   adds anything beyond VIX (the equity-world fear gauge).  A t-stat near zero on the
   MOVE coefficient, while VIX carries a real coefficient, would show MOVE is just a
   noisy VIX proxy.

3. **MOVE/VIX ratio signal** — a rising MOVE/VIX ratio suggests bond vol rising faster
   than equity vol (a cross-asset stress signal).  We test whether a high or rising
   MOVE/VIX ratio predicts negative SPY forward returns better than MOVE alone.

No look-ahead: MOVE and VIX measured at the close of day *t* are used to rank/signal;
the forward return is the log-return from day *t* close to day *t+h* close.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Signal construction helpers
# ---------------------------------------------------------------------------

def move_quintile(move: pd.Series, window: int = 252, min_periods: int = 63) -> pd.Series:
    """Rolling MOVE quintile rank (1 = bottom 20%, 5 = top 20%).

    Ranks are formed strictly out-of-sample: the percentile at day *t* uses only
    history up to and including day *t*.  Returns integer quintile labels 1–5.
    """
    pct = move.rolling(window, min_periods=min_periods).rank(pct=True)
    quintile = np.ceil(pct * 5).clip(1, 5).astype("Int64")
    return quintile


def forward_return(spy_close: pd.Series, horizon: int = 1) -> pd.Series:
    """Log forward return from day *t* close to day *t+horizon* close.

    Signal at day *t* uses MOVE/VIX known at the close of *t*; the forward return is
    ``log(close[t+horizon]) - log(close[t])``, aligned at day *t*.  This is a
    *holding-period* return: buy at close *t*, exit at close *t+horizon*.  The last
    ``horizon`` observations are NaN (no future data available).
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
    """Forward-return statistics for each MOVE quintile, across multiple horizons.

    Parameters
    ----------
    df : DataFrame with columns ``[MOVE, SPY_close]`` and a DatetimeIndex.
    horizons : list of forward-return horizons in trading days (default [1, 5, 21]).
    window : rolling lookback for the quintile rank.
    min_periods : minimum observations before a rank is computed.

    Returns
    -------
    A DataFrame indexed by (quintile, horizon) with columns
    ``[n, mean_bps, std_bps, t_stat]``.
    """
    if horizons is None:
        horizons = [1, 5, 21]
    q = move_quintile(df["MOVE"], window=window, min_periods=min_periods)
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
    """HAC t-stat on the spread: mean(forward_return | MOVE Q5) − mean(... | MOVE Q1).

    The decisive test of the claim: does the highest-MOVE quintile predict *worse*
    forward returns than the lowest-MOVE quintile?  Returns a dict with keys
    ``{n_top, n_bot, mean_top_bps, mean_bot_bps, spread_bps, t_spread,
       t_top, t_bot}``.
    """
    q = move_quintile(df["MOVE"], window=window, min_periods=min_periods)
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
# MOVE vs VIX incremental regression
# ---------------------------------------------------------------------------

def move_vs_vix_regression(
    df: pd.DataFrame,
    horizon: int = 1,
    window: int = 252,
    min_periods: int = 63,
) -> dict:
    """OLS of forward_return on standardised MOVE and VIX, with HAC inference.

    Both predictors are standardised (z-scored over the rolling window) so their
    coefficients are directly comparable.  Returns a dict with keys
    ``{n, beta_move, t_move, beta_vix, t_vix, r2}``.

    A tiny ``beta_move`` and small ``|t_move|`` while ``|t_vix|`` is large would mean
    MOVE adds nothing over VIX alone — consistent with the 'Mirage' verdict.
    """
    fwd = forward_return(df["SPY_close"], horizon=horizon)

    # Rolling z-scores (out-of-sample standardisation).
    move_z = (
        (df["MOVE"] - df["MOVE"].rolling(window, min_periods=min_periods).mean())
        / df["MOVE"].rolling(window, min_periods=min_periods).std(ddof=1)
    )
    vix_z = (
        (df["VIX"] - df["VIX"].rolling(window, min_periods=min_periods).mean())
        / df["VIX"].rolling(window, min_periods=min_periods).std(ddof=1)
    )

    combined = pd.DataFrame({"fwd": fwd, "move_z": move_z, "vix_z": vix_z}).dropna()
    n = len(combined)
    if n < 30:
        return {"n": n, "beta_move": np.nan, "t_move": np.nan,
                "beta_vix": np.nan, "t_vix": np.nan, "r2": np.nan}

    y = combined["fwd"].to_numpy()
    X = np.column_stack([np.ones(n), combined["move_z"].to_numpy(),
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
        "beta_move": float(beta[1]),
        "t_move": float(beta[1] / se[1]) if se[1] > 0 else np.nan,
        "beta_vix": float(beta[2]),
        "t_vix": float(beta[2] / se[2]) if se[2] > 0 else np.nan,
        "r2": r2,
    }


# ---------------------------------------------------------------------------
# MOVE/VIX ratio signal
# ---------------------------------------------------------------------------

def move_vix_ratio_signal(
    df: pd.DataFrame,
    horizon: int = 5,
    window: int = 252,
    min_periods: int = 63,
) -> dict:
    """Test whether the MOVE/VIX ratio (or its change) predicts forward SPY returns.

    We compute the MOVE/VIX ratio, then its rolling z-score (OOS), and run OLS of the
    forward return on the lagged ratio z-score.  A significant negative coefficient
    would mean a rising MOVE/VIX ratio (bond vol elevated relative to equity vol) is a
    cross-asset stress predictor for equities.  Returns a dict with keys
    ``{n, beta_ratio, t_ratio, mean_top_bps, mean_bot_bps, spread_bps, t_spread}``.
    """
    ratio = df["MOVE"] / df["VIX"].replace(0, np.nan)
    fwd = forward_return(df["SPY_close"], horizon=horizon)

    ratio_z = (
        (ratio - ratio.rolling(window, min_periods=min_periods).mean())
        / ratio.rolling(window, min_periods=min_periods).std(ddof=1)
    )

    combined = pd.DataFrame({"fwd": fwd, "ratio_z": ratio_z}).dropna()
    n = len(combined)
    if n < 30:
        return {"n": n, "beta_ratio": np.nan, "t_ratio": np.nan,
                "mean_top_bps": np.nan, "mean_bot_bps": np.nan,
                "spread_bps": np.nan, "t_spread": np.nan}

    y = combined["fwd"].to_numpy()
    X = np.column_stack([np.ones(n), combined["ratio_z"].to_numpy()])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    yhat = X @ beta
    resid = y - yhat

    # HAC inference
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

    # Top/bot quintile analysis of the ratio
    ratio_q = move_quintile(ratio.dropna(), window=window, min_periods=min_periods)
    ratio_q = ratio_q.reindex(fwd.index)
    r_top = fwd[(ratio_q == 5) & fwd.notna()].to_numpy(dtype=float)
    r_bot = fwd[(ratio_q == 1) & fwd.notna()].to_numpy(dtype=float)
    spread = np.concatenate([r_top, -r_bot])

    return {
        "n": n,
        "beta_ratio": float(beta[1]),
        "t_ratio": float(beta[1] / se[1]) if se[1] > 0 else np.nan,
        "mean_top_bps": float(r_top.mean() * 1e4) if r_top.size else np.nan,
        "mean_bot_bps": float(r_bot.mean() * 1e4) if r_bot.size else np.nan,
        "spread_bps": float((r_top.mean() - r_bot.mean()) * 1e4)
            if r_top.size and r_bot.size else np.nan,
        "t_spread": float(_hac_tstat(spread)) if spread.size > 5 else np.nan,
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
