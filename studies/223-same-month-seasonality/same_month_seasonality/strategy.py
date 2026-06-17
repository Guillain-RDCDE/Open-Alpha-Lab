"""Strategy and honest controls -- Study 223 (Same-Month Seasonality).

The Heston & Sadka (2008) recipe: each month t (calendar month m), sort stocks
by their average past return in the *same* calendar month m; long the top
decile (stocks that historically do best in month m), short the bottom decile
(historically worst in month m); hold for 1 month.  Repeat every month,
updating the historical average with the latest observation.

This is a *within-calendar-month* cross-sectional sort -- it is NOT the same
as a time-series seasonal (which exploits the stock market's monthly pattern
in aggregate).  The claim is stock-specific: some stocks have a persistent
tendency to outperform in January (or March, or October...) and that tendency
is forecastable from their own history.

Three honest comparisons settle the matter:

1. **High vs Low** -- the classic long-short spread (top decile minus bottom
   decile monthly return).
2. **High vs Equal-weight market** -- the long-only top-decile excess return,
   which is the more practical benchmark.
3. **Random portfolio control** -- a randomly-drawn decile of the *same size*,
   to test whether the same-month signal carries information beyond pure luck.

**No look-ahead**: the signal for month t uses past same-calendar-month returns
ending at t-12 (last year's occurrence of the same month).  A minimum of
``min_years`` prior same-month observations is required; the burn-in is
``min_years`` years.

**Survivorship bias**: the real-tape universe is a fixed large-cap basket
projected backwards.  All real-tape results are upper bounds; the bias is
named on the Signal axis.

**Execution lag**: positions are formed at the last trading day of month t-1
(signal computed from data through t-12), entered at t+1 open (one-month lag
convention).  In practice this is a restatement of the fact that we use t-12
as the last data point.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Signal construction
# ---------------------------------------------------------------------------
def same_month_signal(
    prices: pd.DataFrame,
    min_years: int = 5,
) -> pd.DataFrame:
    """Average historical return in the *next* calendar month, used as a predictor.

    For each month-end date t, this signal predicts the return earned *during*
    month t+1 (calendar month m+1).  The signal is the average of all past
    returns in calendar month m+1, using data strictly before t (i.e., ending
    at t-11 months, which is the most recent prior occurrence of month m+1
    that has already settled).

    Design rationale::

        t = end of December 2015
        next month = January 2016
        signal = avg(all prior January returns ending Dec 2014 and earlier)
        fwd return (at row t in decile_returns) = Jan-2016 return

    This is the classic Heston-Sadka (2008) design: at the end of month t,
    rank stocks by their historical performance in the *coming* calendar month,
    and hold through that month.  No look-ahead because we exclude the
    most-recent prior occurrence (11-month lag keeps the current calendar-year
    value out of the signal).

    Parameters
    ----------
    prices :
        Month-end price panel (date x ticker).
    min_years :
        Minimum number of historical same-month observations required before
        a signal value is emitted.  Default 5 (5 prior Januaries to forecast
        the coming January, etc.).

    Returns
    -------
    DataFrame of same-month signal scores (same shape as ``prices``).
    High values = historically strong in the *next* calendar month.
    NaN where history is insufficient (burn-in period).
    """
    # Monthly simple returns (pct_change)
    monthly_ret = prices.pct_change(periods=1)
    monthly_ret.index = pd.DatetimeIndex(monthly_ret.index)

    # Assign calendar month to each row (the month EARNED, 1..12)
    months_of_year = monthly_ret.index.month  # 1..12

    signal_rows: dict[pd.Timestamp, pd.Series] = {}

    for t in monthly_ret.index:
        # We are at end of month t.  The forward return at row t (fwd = pct_change.shift(-1))
        # is the return earned IN month t+1, whose calendar month is:
        next_cal_month = (t.month % 12) + 1  # 1..12, wraps Jan -> Feb
        # Past occurrences of next_cal_month, ending strictly before t
        # (so the most recent eligible observation is 11+ months ago)
        past_mask = (months_of_year == next_cal_month) & (monthly_ret.index < t - pd.DateOffset(months=11))
        past_ret = monthly_ret.loc[past_mask]
        if len(past_ret) < min_years:
            signal_rows[t] = pd.Series(np.nan, index=prices.columns)
        else:
            signal_rows[t] = past_ret.mean(axis=0)

    signal = pd.DataFrame.from_dict(signal_rows, orient="index")
    signal.index.name = "date"
    signal = signal[prices.columns]  # preserve column order
    return signal


# ---------------------------------------------------------------------------
# Portfolio construction
# ---------------------------------------------------------------------------
def decile_returns(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    q: float = 0.10,
    min_stocks: int = 10,
) -> pd.DataFrame:
    """Monthly top-decile-minus-bottom-decile spread plus equal-weight market return.

    For each month t, stocks are sorted by signal.loc[t] into deciles.
    The top ``q`` fraction (highest same-month avg return = *high* seasonality)
    is the long portfolio; the bottom ``q`` fraction is the short leg.

    The forward return is the simple return from month t to t+1.

    Returns a DataFrame with columns:
    - ``high``    : equal-weight mean return of the top decile
    - ``low``     : equal-weight mean return of the bottom decile
    - ``market``  : equal-weight mean return of all ranked stocks
    - ``spread``  : high - low
    - ``n_total`` : number of stocks with valid signal and forward return
    - ``n_high``  : stocks in the high-seasonality decile
    - ``n_low``   : stocks in the low-seasonality decile
    """
    # Monthly forward simple return
    fwd = prices.pct_change(periods=1).shift(-1)  # month t -> t+1

    rows: list[dict] = []
    for t in signal.index:
        if t not in fwd.index:
            continue
        s = signal.loc[t].dropna()
        f = fwd.loc[t].dropna()
        common = s.index.intersection(f.index)
        if len(common) < min_stocks:
            continue
        s = s.loc[common]
        f = f.loc[common]

        hi_thresh = s.quantile(1.0 - q)
        lo_thresh = s.quantile(q)
        high_mask = s >= hi_thresh
        low_mask = s <= lo_thresh

        high_ret = float(f[high_mask].mean())
        low_ret = float(f[low_mask].mean())
        market_ret = float(f.mean())

        rows.append(
            {
                "date": t,
                "high": high_ret,
                "low": low_ret,
                "market": market_ret,
                "spread": high_ret - low_ret,
                "n_total": int(len(common)),
                "n_high": int(high_mask.sum()),
                "n_low": int(low_mask.sum()),
            }
        )
    result = pd.DataFrame(rows).set_index("date")
    result.index = pd.DatetimeIndex(result.index)
    return result


def random_portfolio_returns(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    q: float = 0.10,
    n_draws: int = 300,
    seed: int = 223,
    min_stocks: int = 10,
) -> pd.Series:
    """Null distribution of random-decile excess returns (top-decile size).

    For each month, draw ``n_draws`` random subsets of the same number of
    stocks as the top-decile portfolio.  Returns a Series of (random excess
    vs market) across all (month x draw) combinations.
    """
    rng = np.random.default_rng(seed)
    fwd = prices.pct_change(periods=1).shift(-1)
    excesses: list[float] = []

    for t in signal.index:
        if t not in fwd.index:
            continue
        s = signal.loc[t].dropna()
        f = fwd.loc[t].dropna()
        common = s.index.intersection(f.index)
        if len(common) < min_stocks:
            continue
        f = f.loc[common]
        n_high = max(1, int(round(len(common) * q)))
        mkt = f.to_numpy()
        mkt_mean = mkt.mean()
        all_idx = np.arange(len(mkt))
        for _ in range(n_draws):
            pick = rng.choice(all_idx, size=n_high, replace=False)
            excesses.append(float(mkt[pick].mean() - mkt_mean))

    return pd.Series(excesses, name="random_excess")


# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
def summarize(series: pd.Series) -> dict:
    """Headline statistics for a monthly return/spread series.

    Uses Newey-West HAC standard error for the t-stat.
    """
    r = pd.Series(series).astype(float).dropna()
    n = int(r.size)
    if n < 3:
        return {k: np.nan for k in ("mean", "vol", "sharpe", "tstat", "hit_rate", "n")}

    mu = float(r.mean())
    std = float(r.std(ddof=1))
    sr = mu / std if std > 0 else float("nan")

    # Newey-West HAC t-stat
    e = r.to_numpy() - mu
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    tstat = float(mu / se) if se > 0 else float("nan")

    eq = (1.0 + r).cumprod()
    max_dd = float((eq / eq.cummax() - 1.0).min())

    return {
        "mean": float(mu),
        "vol": float(std),
        "sharpe": float(sr),
        "tstat": float(tstat),
        "hit_rate": float((r > 0).mean()),
        "max_drawdown": float(max_dd),
        "n": n,
    }


def annualise_monthly(stats: dict, periods: int = 12) -> dict:
    """Convert monthly stats to annualised equivalents."""
    out = dict(stats)
    if "mean" in out and np.isfinite(out["mean"]):
        out["mean_ann"] = float(out["mean"] * periods)
    if "vol" in out and np.isfinite(out["vol"]):
        out["vol_ann"] = float(out["vol"] * np.sqrt(periods))
    if "mean_ann" in out and "vol_ann" in out and out["vol_ann"] > 0:
        out["sharpe_ann"] = float(out["mean_ann"] / out["vol_ann"])
    return out


def turnover_cost_drag(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    q: float = 0.10,
    one_way_bps: float = 10.0,
    min_stocks: int = 10,
) -> pd.Series:
    """Monthly cost drag from portfolio turnover (assuming full rebalance each month).

    Each month the top-decile composition changes; the fraction of the portfolio
    that turns over is the overlap complement.  Cost drag is
    turnover x 2 x one-way cost (round-trip).

    Returns a Series of monthly cost drag (in fraction, so 0.001 = 0.1% = 10 bps).
    """
    rows: list[dict] = []
    prev_high: set = set()

    for t in signal.index:
        s = signal.loc[t].dropna()
        f = prices.loc[t].dropna() if t in prices.index else pd.Series(dtype=float)
        common = s.index.intersection(f.index)
        if len(common) < min_stocks:
            prev_high = set()
            continue
        s = s.loc[common]
        hi_thresh = s.quantile(1.0 - q)
        curr_high = set(s[s >= hi_thresh].index.tolist())
        if prev_high:
            retained = prev_high & curr_high
            turnover = 1.0 - len(retained) / len(curr_high) if curr_high else 1.0
        else:
            turnover = 1.0
        drag = turnover * 2.0 * one_way_bps * 1e-4
        rows.append({"date": t, "turnover": turnover, "drag": drag})
        prev_high = curr_high

    if not rows:
        return pd.Series(dtype=float, name="drag")
    df = pd.DataFrame(rows).set_index("date")
    return df["drag"].rename("drag")
