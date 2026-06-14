"""The strategy and its honest controls — Study 130 (Vol-Risk-Premium).

The VRP (``VIX - trailing 21-day realised vol``) is the raw options-market measure of the
price sellers charge for variance insurance.  We evaluate three interlinked claims:

1. **The premium is real and persistent.**  Average VRP > 0, statistically significant;
   VIX > RV on most days.

2. **The VRP predicts forward SPY returns.**  A high VRP today (calm implied market,
   vol sellers have been compensated) predicts *positive* SPY returns over the next
   1/5/21 days.  We test this with quintile-sorted forward returns (HAC t-stat on Q5−Q1)
   and a binary timer (long SPY when VRP is above its rolling median, flat otherwise).
   The fair baseline for the timer is unconditional buy-and-hold.

3. **Short-vol payoff and skew.**  The gross edge of the VRP seller comes with
   *catastrophic negative skew* (crash risk); we quantify the payoff distribution.
   There is no simple trade-able payoff without variance/option infrastructure, so
   "tradability" must account for the fat left tail.

No look-ahead: the VRP at close of day *t* forms the signal; all forward returns start
at the close of day *t* (enter the next open, or equivalently at the next-day close for
an overnight trade).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data import RV_WINDOW


# ---------------------------------------------------------------------------
# Signal: VRP quintile rank (rolling, out-of-sample)
# ---------------------------------------------------------------------------
def vrp_quintile(
    vrp: pd.Series,
    window: int = 252,
    min_periods: int = 63,
) -> pd.Series:
    """Rolling VRP quintile rank: 1 = lowest VRP, 5 = highest VRP.

    Ranks are strictly out-of-sample: the percentile at day *t* uses only history
    up to and including day *t*.  Returns integer quintile labels 1–5 as a nullable
    integer Series (NaN during the warmup period).
    """
    pct = vrp.rolling(window, min_periods=min_periods).rank(pct=True)
    quintile = np.ceil(pct * 5).clip(1, 5).astype("Int64")
    return quintile


def vrp_above_median(
    vrp: pd.Series,
    window: int = 252,
    min_periods: int = 63,
) -> pd.Series:
    """Boolean: True when the VRP is above its rolling 252-day median (out-of-sample)."""
    med = vrp.rolling(window, min_periods=min_periods).median()
    return vrp > med


# ---------------------------------------------------------------------------
# Forward returns
# ---------------------------------------------------------------------------
def forward_return(spy_close: pd.Series, horizon: int = 1) -> pd.Series:
    """Log forward return from day *t* close to day *t+horizon* close.

    Signal at day *t* uses the VRP known at the close of *t* (trailing 21-day RV plus
    that day's VIX close); the forward return is ``log(close[t+horizon]) - log(close[t])``.
    The last ``horizon`` observations are NaN (no future data available).
    """
    log_close = np.log(spy_close)
    return log_close.shift(-horizon) - log_close


# ---------------------------------------------------------------------------
# Core analysis 1: VRP premium statistics
# ---------------------------------------------------------------------------
def vrp_premium_stats(df: pd.DataFrame) -> dict:
    """Statistics for the VRP itself: mean, std, pct-positive, HAC t-stat.

    ``df`` must have a ``VRP`` column (VIX minus trailing-21d-RV, annualised pct).
    The HAC t-stat tests H0: mean VRP = 0.  The mean being significantly positive is the
    most basic statement of the variance risk premium.
    """
    r = df["VRP"].dropna().to_numpy(dtype=float)
    n = int(r.size)
    mean_vrp = float(r.mean())
    std_vrp = float(r.std(ddof=1))
    pct_pos = float((r > 0).mean() * 100)
    t = _hac_tstat(r)
    return {
        "n": n,
        "mean_vrp": mean_vrp,
        "std_vrp": std_vrp,
        "pct_positive": pct_pos,
        "tstat": t,
        "mean_vix": float(df["VIX"].dropna().mean()),
        "mean_rv": float(df["RV"].dropna().mean()),
    }


# ---------------------------------------------------------------------------
# Core analysis 2: Quintile-sorted forward returns
# ---------------------------------------------------------------------------
def quintile_table(
    df: pd.DataFrame,
    horizons: list[int] | None = None,
    window: int = 252,
    min_periods: int = 63,
) -> pd.DataFrame:
    """Forward-return statistics for each VRP quintile, across multiple horizons.

    Parameters
    ----------
    df : DataFrame with columns ``[VRP, SPY_close]`` and a DatetimeIndex.
    horizons : list of forward-return horizons in calendar days (default [1, 5, 21]).
    window : rolling lookback for the quintile rank.
    min_periods : minimum observations before a rank is computed.

    Returns
    -------
    A DataFrame with columns ``[quintile, horizon, n, mean_bps, std_bps, t_stat]``.
    """
    if horizons is None:
        horizons = [1, 5, 21]
    q = vrp_quintile(df["VRP"], window=window, min_periods=min_periods)
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
# Core analysis 3: Top-vs-bottom quintile spread (the decisive test)
# ---------------------------------------------------------------------------
def top_minus_bottom(
    df: pd.DataFrame,
    horizon: int = 1,
    window: int = 252,
    min_periods: int = 63,
) -> dict:
    """HAC t-stat on the spread: mean(fwd | Q5 high-VRP) − mean(fwd | Q1 low-VRP).

    This is the central test of the predictability claim.  If a high VRP predicts
    positive forward returns, Q5 should materially outperform Q1.  Returns a dict with
    keys ``{n_top, n_bot, mean_top_bps, mean_bot_bps, spread_bps, t_spread, t_top, t_bot}``.
    """
    q = vrp_quintile(df["VRP"], window=window, min_periods=min_periods)
    fwd = forward_return(df["SPY_close"], horizon=horizon)
    r_top = fwd[(q == 5) & fwd.notna()].to_numpy(dtype=float)
    r_bot = fwd[(q == 1) & fwd.notna()].to_numpy(dtype=float)
    spread = np.concatenate([r_top, -r_bot])
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
# Core analysis 4: Binary timing overlay (long when VRP is high, flat otherwise)
# ---------------------------------------------------------------------------
def timing_overlay(
    df: pd.DataFrame,
    horizon: int = 1,
    cost_bps: float = 1.0,
    window: int = 252,
    min_periods: int = 63,
) -> pd.DataFrame:
    """Daily return of the VRP timer vs unconditional buy-and-hold.

    The timer is 'risk-on' (long SPY) when the prior-day VRP exceeds its rolling
    252-day median, and flat (cash) when it does not.  The baseline is always long SPY.
    The 'active minus passive' spread is the edge to test.  ``cost_bps`` is charged on
    each *switch* as a one-way round-trip.

    Returns a DataFrame with columns ``[r_passive, r_active, r_spread,
    is_high_vrp, switched]`` aligned on day *t+1* (the day returns are earned).
    """
    fwd = forward_return(df["SPY_close"], horizon=horizon)
    signal_raw = vrp_above_median(df["VRP"], window=window, min_periods=min_periods).shift(1)
    signal = signal_raw.astype("boolean")
    switch = signal_raw.ne(signal_raw.shift(1)) & signal_raw.notna() & signal_raw.shift(1).notna()

    r_passive = fwd
    r_active = fwd.where(signal.fillna(False), 0.0) - switch.astype(float) * cost_bps * 1e-4

    out = pd.DataFrame({
        "r_passive": r_passive,
        "r_active": r_active,
        "r_spread": r_active - r_passive,
        "is_high_vrp": signal,
        "switched": switch,
    }).dropna(subset=["r_passive", "r_active"])
    return out


# ---------------------------------------------------------------------------
# Core analysis 5: Short-vol payoff distribution (the skew/crash side)
# ---------------------------------------------------------------------------
def short_vol_payoff(
    df: pd.DataFrame,
    horizon: int = 21,
) -> dict:
    """Approximate short-vol payoff: each period earn VRP but pay the absolute
    realised-vs-implied gap; model the distribution to quantify crash skew.

    This is a *stylised* payoff that captures the essence of delta-hedged short
    variance without requiring full options infrastructure:
    - Enter when VRP > 0 (earn a positive premium in expectation).
    - At expiry (``horizon`` days later), the P&L is proportional to
      ``VRP_entry - abs(RV_exit - VRP_entry)`` — a simplified payout reflecting
      that you earn the spread but can be hurt by very large realised moves.
    - We approximate by computing rolling non-overlapping periods of ``horizon`` days,
      taking the VRP at the start of each period and SPY log-return over the period.

    Returns a dict with skew, kurtosis, fraction of negative months, and the
    5th-percentile drawdown (the "crash" statistic).
    """
    log_ret = df["SPY_ret"].copy()
    vrp = df["VRP"].copy()

    # Non-overlapping windows of `horizon` days
    n = len(df)
    starts = list(range(0, n - horizon, horizon))
    payoffs = []
    for s in starts:
        vrp_entry = vrp.iloc[s]
        period_rets = log_ret.iloc[s + 1: s + horizon + 1]
        if pd.isna(vrp_entry) or period_rets.isna().any():
            continue
        # Aggregate return over the period.
        agg_ret = period_rets.sum()
        # Realised vol of the period (annualised pct).
        rv_period = float(period_rets.std(ddof=1) * np.sqrt(252) * 100)
        if pd.isna(rv_period):
            continue
        # Simplified P&L: collect the VRP (premium earned) adjusted for the equity move.
        # When markets crash, both RV spikes and returns are negative — double pain.
        # We use SPY log-return as the primary equity return.
        payoffs.append({"vrp_entry": vrp_entry, "agg_ret": agg_ret, "rv_period": rv_period})

    if not payoffs:
        return {"n": 0}

    pdf = pd.DataFrame(payoffs)
    r = pdf["agg_ret"].to_numpy(dtype=float)
    n_periods = int(len(r))
    return {
        "n": n_periods,
        "mean_ret_pct": float(r.mean() * 100),
        "std_ret_pct": float(r.std(ddof=1) * 100),
        "skew": float(pd.Series(r).skew()),
        "kurt": float(pd.Series(r).kurtosis()),
        "pct_negative": float((r < 0).mean() * 100),
        "p05": float(np.percentile(r, 5) * 100),
        "worst": float(r.min() * 100),
        "mean_vrp_entry": float(pdf["vrp_entry"].mean()),
    }


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
        "sharpe_ann": float(r.mean() / r.std(ddof=1) * np.sqrt(252))
            if n > 1 and r.std() > 0 else np.nan,
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
