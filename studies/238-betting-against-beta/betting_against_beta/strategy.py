"""The Betting-Against-Beta (BAB) strategy -- Frazzini & Pedersen (2014).

Construction:
  1. Estimate rolling betas using a 252-day (1-year) trailing window.
  2. Each month, rank stocks by estimated beta.
  3. Low-beta portfolio: stocks in the bottom half of the beta distribution.
     Leverage the long leg so it has unit beta (weight = 1 / avg_beta_low).
  4. High-beta portfolio: stocks in the top half.
     Deleverage the short leg to unit beta (weight = 1 / avg_beta_high).
  5. BAB return = (leveraged low-beta return) - (deleveraged high-beta return).

This makes the portfolio ex-ante beta-neutral. Returns are measured as monthly excess
returns (vs the risk-free rate proxied by the 1-month T-bill). We approximate the
risk-free rate as a constant 3%/yr (0.25%/month) -- close to the long-run average and
sufficient for ranking purposes.

Lever cap: the leverage on the long leg is capped at 2.0 to avoid unrealistic
leverage assumptions on a retail/institutional level. We state this cap explicitly.

The panel is **survivorship-biased** (current S&P 500 projected backwards).
All results should be treated as upper-bound estimates.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Approximate risk-free rate (annualised); used to compute excess returns.
# We do NOT pull live T-bill data to avoid FRED dependency.
RF_ANNUAL = 0.03
RF_DAILY = RF_ANNUAL / 252

# Rolling beta estimation window (trading days)
BETA_WINDOW = 252

# Rebalancing frequency (business day offset for month-start)
REBAL_FREQ = "MS"

# Leverage cap on the long leg (cap avoids unrealistic assumptions)
MAX_LEVERAGE = 2.0


# ---------------------------------------------------------------------------
# Beta estimation
# ---------------------------------------------------------------------------
def rolling_betas(
    stock_rets: pd.DataFrame,
    mkt_rets: pd.Series,
    window: int = BETA_WINDOW,
) -> pd.DataFrame:
    """Rolling OLS beta for each stock vs the market.

    Beta_{i,t} = Cov(r_i, r_mkt) / Var(r_mkt) on the trailing ``window`` days.
    Uses pandas rolling to avoid looping; NaN when fewer than window//2 obs are available.

    Returns a (date × ticker) DataFrame of rolling betas.
    """
    common_idx = stock_rets.index.intersection(mkt_rets.index)
    sr = stock_rets.reindex(common_idx)
    mr = mkt_rets.reindex(common_idx)

    # Demean using rolling mean
    roll_cov = sr.rolling(window, min_periods=window // 2).cov(mr)
    roll_var = mr.rolling(window, min_periods=window // 2).var()

    # Broadcast: (date,) -> (date, ticker)
    betas = roll_cov.div(roll_var, axis=0)
    return betas


# ---------------------------------------------------------------------------
# BAB portfolio construction
# ---------------------------------------------------------------------------
def bab_portfolio(
    prices: pd.DataFrame,
    spy_prices: pd.Series,
    beta_window: int = BETA_WINDOW,
    rebal_freq: str = REBAL_FREQ,
    min_stocks: int = 20,
    leverage_cap: float = MAX_LEVERAGE,
) -> pd.DataFrame:
    """Construct monthly BAB portfolio returns.

    Steps for each rebalancing date *t*:
    1. Compute daily excess returns for stocks and SPY up to *t*.
    2. Roll betas on the trailing ``beta_window`` days.
    3. Rank by beta; split at median.
    4. Long low-beta leg (leverage = min(1/avg_beta_L, leverage_cap)).
    5. Short high-beta leg (deleverage = 1/avg_beta_H).
    6. Hold for one calendar month, measuring the equal-weight return of each leg.

    Returns a DataFrame with columns:
        ``low_beta_ret``  -- equal-weight return of the low-beta leg (un-leveraged)
        ``high_beta_ret`` -- equal-weight return of the high-beta leg (un-leveraged)
        ``avg_beta_low``  -- average beta of the long leg at rebalancing
        ``avg_beta_high`` -- average beta of the short leg at rebalancing
        ``leverage``      -- leverage applied to the long leg (capped)
        ``deleverge``     -- leverage applied to the short leg (= 1/avg_beta_high)
        ``bab_ret``       -- leveraged long leg minus deleveraged short leg
        ``mkt_ret``       -- SPY return over the same period
        ``n``             -- number of stocks in the universe
    """
    # Daily returns
    stock_rets = prices.pct_change().dropna(how="all")
    mkt_rets = spy_prices.pct_change().dropna()

    # Excess returns
    stock_excess = stock_rets.sub(RF_DAILY)
    mkt_excess = mkt_rets.sub(RF_DAILY)

    # Rolling betas
    betas = rolling_betas(stock_excess, mkt_excess, window=beta_window)

    # Monthly rebalancing dates (month starts within the price series range)
    rebal_dates = pd.date_range(
        start=prices.index[beta_window],
        end=prices.index[-1],
        freq=rebal_freq,
    )
    rebal_dates = [d for d in rebal_dates if d <= prices.index[-1]]

    rows: list[dict] = []

    for i, rebal_dt in enumerate(rebal_dates[:-1]):
        next_rebal = rebal_dates[i + 1]

        # Find beta snapshot at rebalancing date (last available beta <= rebal_dt)
        beta_avail = betas[betas.index <= rebal_dt]
        if beta_avail.empty:
            continue
        beta_snap = beta_avail.iloc[-1].dropna()

        # Filter to stocks with valid betas, positive betas > 0
        beta_snap = beta_snap[beta_snap > 0]
        if len(beta_snap) < min_stocks:
            continue

        # Split at median
        med = beta_snap.median()
        low_tickers = beta_snap[beta_snap <= med].index
        high_tickers = beta_snap[beta_snap > med].index

        avg_beta_low = float(beta_snap[low_tickers].mean())
        avg_beta_high = float(beta_snap[high_tickers].mean())

        if avg_beta_low <= 0 or avg_beta_high <= 0:
            continue

        # Leverage: long leg scaled to unit beta, capped; short leg deleveraged
        leverage = min(1.0 / avg_beta_low, leverage_cap)
        deleverage = 1.0 / avg_beta_high

        # Holding period: [rebal_dt, next_rebal)
        hold = stock_rets.loc[
            (stock_rets.index > rebal_dt) & (stock_rets.index <= next_rebal)
        ]
        mkt_hold = mkt_rets.loc[
            (mkt_rets.index > rebal_dt) & (mkt_rets.index <= next_rebal)
        ]
        if hold.empty or mkt_hold.empty:
            continue

        # Compound daily returns over the holding period
        def _compound(ser: pd.Series) -> float:
            return float((1.0 + ser.dropna()).prod() - 1.0)

        low_ret = float(hold[low_tickers].apply(_compound).mean())
        high_ret = float(hold[high_tickers].apply(_compound).mean())
        mkt_ret = _compound(mkt_hold)

        bab_ret = leverage * low_ret - deleverage * high_ret

        rows.append(
            {
                "date": rebal_dt,
                "low_beta_ret": low_ret,
                "high_beta_ret": high_ret,
                "avg_beta_low": avg_beta_low,
                "avg_beta_high": avg_beta_high,
                "leverage": leverage,
                "deleverage": deleverage,
                "bab_ret": bab_ret,
                "mkt_ret": mkt_ret,
                "n": int(len(beta_snap)),
            }
        )

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows).set_index("date")
    return df


# ---------------------------------------------------------------------------
# Annualised summary statistics
# ---------------------------------------------------------------------------
def summary(monthly_returns: pd.Series, periods_per_year: int = 12) -> dict:
    """Annualised statistics for a monthly return series.

    Returns mean (annualised), vol (annualised), Sharpe, a Newey-West HAC t-stat,
    hit-rate (months > 0), max drawdown, and *n* (number of months).
    The HAC t-stat is the inference-bar number.
    """
    r = pd.Series(monthly_returns).astype(float).dropna()
    n = len(r)
    if n < 2:
        return {k: np.nan for k in ("mean", "vol", "sharpe", "tstat", "hit_rate", "max_drawdown", "n")}

    mean_m = float(r.mean())
    vol_m = float(r.std(ddof=1))
    mean_ann = mean_m * periods_per_year
    vol_ann = vol_m * np.sqrt(periods_per_year)
    sr = mean_ann / vol_ann if vol_ann > 0 else np.nan

    eq = (1.0 + r).cumprod()
    dd = float((eq / eq.cummax() - 1.0).min())

    # Newey-West HAC t-stat (monthly series)
    arr = r.to_numpy()
    mu = arr.mean()
    e = arr - mu
    lags = max(1, int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0))))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    tstat = float(mu / se) if se > 0 else np.nan

    return {
        "mean": mean_ann,
        "vol": vol_ann,
        "sharpe": sr,
        "tstat": tstat,
        "hit_rate": float((r > 0).mean()),
        "max_drawdown": dd,
        "n": n,
    }
