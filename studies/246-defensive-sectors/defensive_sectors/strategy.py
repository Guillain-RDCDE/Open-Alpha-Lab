"""The strategy and its honest controls -- Study 246 (Defensive-Sectors Leadership).

The folk recipe: when *both* consumer staples (XLP) and utilities (XLU) outperform
SPY on a combined relative-strength basis (rising 20-day momentum of the average
XLP+XLU / SPY log-ratio), exit or reduce equity exposure because a drawdown is coming.
We test this three ways:

1. **Quintile-sorted forward returns** -- sort trading days by the rolling percentile
   rank of the combined defensive RS momentum into five buckets and compute forward
   1-day / 5-day / 21-day SPY log-returns per bucket.  The folk claim predicts a
   *monotone descent* from bottom (SPY outperforming, risk-on) to top (defensives
   strongly outperforming, risk-off -> bad outcomes).
   Inference: HAC t-stat on the Q5 - Q1 spread (the decisive test).

2. **Binary timing overlay** -- a "risk-off" signal that is *flat* (cash) when the
   prior-day combined RS momentum is in the top quintile (defensives strongly
   outperforming), and *long SPY* otherwise.  The fair baseline is unconditional
   buy-and-hold.  Inference: HAC t-stat on the active-minus-passive daily return spread.

3. **Regime stats** -- SPY statistics conditional on being in a "defensive alert"
   regime (combined RS momentum > rolling median) vs a "risk-on" (calm) regime.

No look-ahead: the combined RS momentum at close of day *t* forms the signal; the
forward return starts at the close of day *t+1*.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Signal construction helpers
# ---------------------------------------------------------------------------

def combined_rs_momentum(df: pd.DataFrame, window: int = 20) -> pd.Series:
    """20-day change in the average log(XLP/SPY) + log(XLU/SPY) combined RS ratio.

    Positive means defensive sectors (XLP+XLU) have been collectively outperforming SPY
    over the past ``window`` days -- the "defensive alert" state the folk signal uses.
    Using the average of both sector RS ratios reduces noise relative to a single-sector
    canary and is more consistent with the multi-sector narrative.
    """
    rs_xlp = np.log(df["XLP_close"]) - np.log(df["SPY_close"])
    rs_xlu = np.log(df["XLU_close"]) - np.log(df["SPY_close"])
    combined_rs = 0.5 * (rs_xlp + rs_xlu)
    return combined_rs.diff(window)


def rs_momentum_rank(
    rs_mom: pd.Series,
    window: int = 252,
    min_periods: int = 63,
) -> pd.Series:
    """Rolling out-of-sample percentile rank of combined RS momentum (0-1).

    A high rank (close to 1) means defensive sectors have been outperforming SPY more
    aggressively than at almost any point in the prior year -- the "strong defensive
    alert" state.
    """
    return rs_mom.rolling(window, min_periods=min_periods).rank(pct=True)


def forward_return(spy_close: pd.Series, horizon: int = 1) -> pd.Series:
    """Log forward return from day *t* close to day *t+horizon* close.

    Signal at day *t* uses the combined RS momentum known at the close of *t*; the
    forward return is ``log(close[t+horizon]) - log(close[t])``, aligned at day *t*.
    The last ``horizon`` observations are NaN (no future data).
    """
    log_close = np.log(spy_close)
    return log_close.shift(-horizon) - log_close


def defensive_quintile(
    rs_mom: pd.Series,
    window: int = 252,
    min_periods: int = 63,
) -> pd.Series:
    """Rolling defensive quintile rank (1 = SPY outperforming, 5 = defensives strongly leading).

    Ranks are formed strictly out-of-sample.  Returns integer quintile labels 1-5.
    """
    pct = rs_mom.rolling(window, min_periods=min_periods).rank(pct=True)
    quintile = np.ceil(pct * 5).clip(1, 5).astype("Int64")
    return quintile


# ---------------------------------------------------------------------------
# Quintile-sorted forward return table (the central test)
# ---------------------------------------------------------------------------

def quintile_table(
    df: pd.DataFrame,
    horizons: list[int] | None = None,
    rs_window: int = 20,
    rank_window: int = 252,
    min_periods: int = 63,
) -> pd.DataFrame:
    """Forward-return statistics for each defensive quintile, across multiple horizons.

    Parameters
    ----------
    df : DataFrame with columns ``[XLP_close, XLU_close, SPY_close]`` and a DatetimeIndex.
    horizons : list of forward-return horizons in trading days (default [1, 5, 21]).
    rs_window : lookback window for combined RS momentum (default 20 trading days).
    rank_window : lookback for the percentile rank (default 252 trading days).
    min_periods : minimum observations before a rank is computed.

    Returns
    -------
    A DataFrame with columns ``[quintile, horizon, n, mean_bps, std_bps, t_stat]``.
    Quintile 1 = SPY outperforming (risk-on); quintile 5 = defensives strongly outperforming
    (risk-off alert).  The folk claim predicts mean_bps[Q5] < mean_bps[Q1] and a monotone
    descent.
    """
    if horizons is None:
        horizons = [1, 5, 21]
    rs_mom = combined_rs_momentum(df, window=rs_window)
    q = defensive_quintile(rs_mom, window=rank_window, min_periods=min_periods)
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
# Spread: top quintile - bottom quintile (the decisive test)
# ---------------------------------------------------------------------------

def top_minus_bottom(
    df: pd.DataFrame,
    horizon: int = 1,
    rs_window: int = 20,
    rank_window: int = 252,
    min_periods: int = 63,
) -> dict:
    """HAC t-stat on the spread: mean(fwd | Q1 calm) - mean(fwd | Q5 defensive alert).

    This is the central test.  If defensive sectors leading genuinely forecasts weak SPY
    returns, the Q5 ("strong defensive alert") forward returns should be materially lower
    than Q1 ("SPY leading, risk-on") forward returns.

    Note the sign convention: Q1 - Q5 (positive means the calm regime outperforms the
    alert regime, consistent with the folk claim).

    Returns a dict with keys
    ``{n_alert, n_calm, mean_alert_bps, mean_calm_bps, spread_bps,
       t_spread, t_alert, t_calm}``.
    Here "alert" = Q5 (defensives outperforming) and "calm" = Q1 (SPY outperforming).
    """
    rs_mom = combined_rs_momentum(df, window=rs_window)
    q = defensive_quintile(rs_mom, window=rank_window, min_periods=min_periods)
    fwd = forward_return(df["SPY_close"], horizon=horizon)

    # Q5 = "defensive alert" (defensives strongly outperforming)
    r_alert = fwd[(q == 5) & fwd.notna()].to_numpy(dtype=float)
    # Q1 = "calm" (SPY outperforming -- no alert)
    r_calm = fwd[(q == 1) & fwd.notna()].to_numpy(dtype=float)

    # spread = calm - alert; positive supports the folk claim
    spread = np.concatenate([r_calm, -r_alert])
    return {
        "n_alert": int(r_alert.size),
        "n_calm": int(r_calm.size),
        "mean_alert_bps": float(r_alert.mean() * 1e4) if r_alert.size else np.nan,
        "mean_calm_bps": float(r_calm.mean() * 1e4) if r_calm.size else np.nan,
        "spread_bps": float((r_calm.mean() - r_alert.mean()) * 1e4)
            if (r_calm.size and r_alert.size) else np.nan,
        "t_spread": float(_hac_tstat(spread)) if spread.size > 5 else np.nan,
        "t_alert": float(_hac_tstat(r_alert)) if r_alert.size > 5 else np.nan,
        "t_calm": float(_hac_tstat(r_calm)) if r_calm.size > 5 else np.nan,
    }


# ---------------------------------------------------------------------------
# Binary timing overlay: go to cash when defensives are sounding
# ---------------------------------------------------------------------------

def timing_overlay(
    df: pd.DataFrame,
    horizon: int = 1,
    rs_window: int = 20,
    rank_window: int = 252,
    min_periods: int = 63,
    top_pct: float = 0.80,
    cost_bps: float = 1.0,
) -> pd.DataFrame:
    """Daily return of the defensive-sector timer vs unconditional buy-and-hold.

    The timer is *flat* (in cash, earns 0) when the prior-day combined RS momentum rank
    exceeds ``top_pct`` (default 80th percentile -- "strong defensive alert"), and *long
    SPY* otherwise.  The baseline is always long SPY.  The 'active minus passive' spread
    is the edge to test.  ``cost_bps`` is charged on each *switch* as a one-way round-trip.

    Returns a DataFrame with columns
    ``[r_passive, r_active, r_spread, is_alert, switched]``
    aligned on day *t+1* (the day returns are earned).
    """
    rs_mom = combined_rs_momentum(df, window=rs_window)
    rank = rs_mom.rolling(rank_window, min_periods=min_periods).rank(pct=True)
    fwd1 = forward_return(df["SPY_close"], horizon=horizon)

    # Signal: True = "alert, go to cash" (prior-day rank > top_pct)
    signal_raw = (rank > top_pct).shift(1)
    signal = signal_raw.astype("boolean")
    switch = (
        signal_raw.ne(signal_raw.shift(1))
        & signal_raw.notna()
        & signal_raw.shift(1).notna()
    )

    r_passive = fwd1
    # When alert: earn 0 (cash). When calm: earn SPY return. Minus switch cost.
    r_active = fwd1.where(~signal.fillna(False), 0.0) - switch.astype(float) * cost_bps * 1e-4

    out = pd.DataFrame({
        "r_passive": r_passive,
        "r_active": r_active,
        "r_spread": r_active - r_passive,
        "is_alert": signal,
        "switched": switch,
    }).dropna(subset=["r_passive", "r_active"])
    return out


# ---------------------------------------------------------------------------
# Regime statistics
# ---------------------------------------------------------------------------

def regime_stats(
    df: pd.DataFrame,
    rs_window: int = 20,
    rank_window: int = 252,
    min_periods: int = 63,
    threshold: float = 0.50,
) -> pd.DataFrame:
    """Compare SPY statistics across full-sample, alert, and calm regimes.

    Alert = prior-day combined RS momentum rank > ``threshold`` (default 50th pct).
    Returns a DataFrame indexed by regime with columns
    ``[n, n_pct, mean_bps, std_bps, sharpe_ann, t_stat]``.
    """
    rs_mom = combined_rs_momentum(df, window=rs_window)
    rank = rs_mom.rolling(rank_window, min_periods=min_periods).rank(pct=True)
    fwd1 = forward_return(df["SPY_close"], horizon=1)
    signal = (rank > threshold).shift(1).astype("boolean")
    valid = fwd1.notna() & signal.notna()

    rows = []
    for label, mask in [
        ("full sample", valid),
        ("calm (def Q1-Q2)", valid & ~signal.fillna(True)),
        ("alert (def Q3-Q5)", valid & signal.fillna(False)),
    ]:
        r = fwd1[mask].to_numpy(dtype=float)
        n = r.size
        mean_bps = float(r.mean() * 1e4) if n else np.nan
        std_bps = float(r.std(ddof=1) * 1e4) if n > 1 else np.nan
        sharpe_ann = (
            float(r.mean() / r.std(ddof=1) * np.sqrt(252))
            if n > 1 and r.std() > 0
            else np.nan
        )
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

def summarize(returns: np.ndarray | pd.Series, label: str = "") -> dict:
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
        "sharpe_ann": (
            float(r.mean() / r.std(ddof=1) * np.sqrt(252))
            if n > 1 and r.std() > 0
            else np.nan
        ),
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
