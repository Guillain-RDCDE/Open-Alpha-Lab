"""Analysis engine for Study 212 (Cannabis Stocks).

Three tests, one honest story:

1. **Beta and alpha vs SPY** — OLS of daily cannabis (or ETF) log-returns on SPY log-returns.
   The slope (beta) captures market sensitivity; the intercept (alpha) captures the sector's
   alpha vs the benchmark. For a genuine disruptive investment, alpha should be positive;
   for a wealth-shredder, it should be negative and significant.

2. **Buy-and-hold drawdown and Sharpe** — Pure performance comparison: SPY, MJ (since 2017),
   MSOS (since 2020), and individual names (TLRY, CGC, CRON) vs SPY over their common windows.
   Reports CAGR, Sharpe, annualised vol, max drawdown, and HAC t-stats.

3. **Sector momentum timing rule** — Buy the cannabis ETF when it is above its 200-day SMA,
   hold SPY otherwise (a simple trend-following overlay). The honest comparison is simply
   holding SPY throughout. Tests whether any trend signal can rescue the sector's economics.

No look-ahead: signals use data up to day *t*; positions are entered at the next day's
return (``signal.shift(1)`` pattern before multiplication).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS_PER_YEAR = 252


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute daily log-returns for all columns. First row is NaN and dropped."""
    return np.log(prices / prices.shift(1)).dropna()


# ---------------------------------------------------------------------------
# Test 1: OLS beta / alpha of cannabis on SPY
# ---------------------------------------------------------------------------
def beta_alpha(prices: pd.DataFrame, cannabis_col: str = "cannabis") -> dict:
    """OLS regression of cannabis log-returns on SPY log-returns.

    Estimates:
      r_cannabis = alpha + beta * r_spy + eps

    Returns the beta, annualised alpha, HAC t-stats for both via the Newey-West
    sandwich estimator, R-squared, residual volatility (idio vol).
    """
    r = log_returns(prices)
    spy_col = "spy" if "spy" in r.columns else [c for c in r.columns if c != cannabis_col][0]
    y = r[cannabis_col].to_numpy()
    x = r[spy_col].to_numpy()
    n = len(y)

    # OLS design matrix: [x, 1] -> [beta, alpha]
    X = np.column_stack([x, np.ones(n)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    beta, alpha = float(coef[0]), float(coef[1])
    resid = y - X @ coef

    # Newey-West sandwich: HAC covariance of score U = X * resid (n x 2)
    U = X * resid[:, None]
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    S = U.T @ U / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        Gk = U[k:].T @ U[:-k] / n
        S += w * (Gk + Gk.T)

    XtX_inv = np.linalg.inv(X.T @ X / n)
    cov_hat = XtX_inv @ S @ XtX_inv / n
    se = np.sqrt(np.maximum(np.diag(cov_hat), 0.0))
    t_beta = float(beta / se[0]) if se[0] > 0 else float("nan")
    t_alpha = float(alpha / se[1]) if se[1] > 0 else float("nan")

    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # Annualised statistics
    alpha_ann = float(alpha * TRADING_DAYS_PER_YEAR)
    idio_vol_ann = float(resid.std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    spy_vol_ann = float(r[spy_col].std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    can_vol_ann = float(r[cannabis_col].std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))

    return {
        "n": int(n),
        "beta": beta,
        "alpha_daily_bps": float(alpha * 1e4),
        "alpha_ann_pct": float(alpha_ann * 100),
        "t_beta": t_beta,
        "t_alpha": t_alpha,
        "r_squared": r2,
        "idio_vol_ann_pct": float(idio_vol_ann * 100),
        "spy_vol_ann_pct": float(spy_vol_ann * 100),
        "can_vol_ann_pct": float(can_vol_ann * 100),
        "implied_leverage": float(can_vol_ann / spy_vol_ann) if spy_vol_ann > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Test 2: Buy-and-hold performance summary
# ---------------------------------------------------------------------------
def summary(returns: pd.Series, periods_per_year: int = TRADING_DAYS_PER_YEAR) -> dict:
    """Headline annualised statistics for a daily return series.

    Returns CAGR, Sharpe ratio, annualised vol, max drawdown, and the HAC t-stat
    (Newey-West) on the mean daily return.
    """
    r = pd.Series(returns).astype(float).dropna()
    n = len(r)
    mu = r.mean()
    std = r.std(ddof=1)
    sharpe = float(mu / std * np.sqrt(periods_per_year)) if std > 0 else float("nan")
    vol_ann = float(std * np.sqrt(periods_per_year))

    cum = np.exp(r.cumsum())
    years = n / periods_per_year
    cagr = float(cum.iloc[-1] ** (1.0 / years) - 1.0) if years > 0 and cum.iloc[-1] > 0 else float("nan")

    eq = cum
    max_dd = float((eq / eq.cummax() - 1.0).min())

    e = r.to_numpy() - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    tstat = float(mu / se) if se > 0 else float("nan")

    return {
        "n_days": int(n),
        "cagr_pct": float(cagr * 100),
        "sharpe": sharpe,
        "vol_ann_pct": float(vol_ann * 100),
        "max_drawdown_pct": float(max_dd * 100),
        "mean_daily_bps": float(mu * 1e4),
        "tstat": tstat,
    }


def perf_table(prices: pd.DataFrame) -> dict[str, dict]:
    """Run buy-and-hold summaries for all columns in ``prices``.

    Returns a dict mapping each ticker/column name to its ``summary()`` dict.
    """
    r = log_returns(prices)
    return {col: summary(r[col]) for col in r.columns}


# ---------------------------------------------------------------------------
# Test 3: Sector momentum timing rule
# ---------------------------------------------------------------------------
def timing_signal(prices: pd.DataFrame, sma_n: int = 200, cannabis_col: str = "cannabis") -> pd.Series:
    """Cannabis-vs-SPY sector momentum timing signal.

    Signal = 1 (hold cannabis ETF) when the cannabis price is *above* its ``sma_n``-day SMA,
    = 0 (hold SPY) when below. A simple trend-following overlay on the cannabis price itself.

    No look-ahead: the signal formed at day *t* is applied to the return at day *t+1*
    (the series is shifted by one before multiplication in ``run_backtest``).
    """
    can_price = prices[cannabis_col]
    sma = can_price.rolling(sma_n, min_periods=sma_n).mean()
    above = (can_price > sma).astype(float)
    return above.shift(1).rename("signal")


def run_backtest(
    prices: pd.DataFrame,
    signal: pd.Series,
    cost_bps: float = 5.0,
    cannabis_col: str = "cannabis",
) -> pd.DataFrame:
    """Apply a cannabis/SPY rotation signal to daily returns.

    When ``signal = 1`` (hold cannabis), the day's return is cannabis log-return.
    When ``signal = 0`` (hold SPY), the day's return is SPY log-return.
    A round-trip transaction cost is charged on each switch (one-way ``cost_bps``
    per leg change).

    Returns a DataFrame with columns:
        ``r_spy``      — daily log-return of holding SPY (the baseline).
        ``r_cannabis`` — daily log-return of holding the cannabis ETF.
        ``signal``     — the applied signal (0 = SPY, 1 = cannabis).
        ``r_strategy`` — signal * r_cannabis + (1-signal) * r_spy − switch_cost.
    """
    r = log_returns(prices)
    spy_col = "spy" if "spy" in r.columns else [c for c in r.columns if c != cannabis_col][0]
    r_spy = r[spy_col]
    r_can = r[cannabis_col]

    sig = signal.reindex(r.index).fillna(0.0)
    switches = sig.diff().abs().fillna(0.0)
    cost_per_switch = cost_bps * 1e-4

    r_strat = (sig * r_can + (1.0 - sig) * r_spy - switches * cost_per_switch).rename("r_strategy")

    return pd.concat([
        r_spy.rename("r_spy"),
        r_can.rename("r_cannabis"),
        sig.rename("signal"),
        r_strat,
    ], axis=1).dropna()


def compare_strategies(
    prices: pd.DataFrame,
    sma_n: int = 200,
    cost_bps: float = 5.0,
    cannabis_col: str = "cannabis",
) -> dict:
    """Run SPY buy-and-hold, cannabis buy-and-hold, and the cannabis/SPY timing rule.

    Returns a dict with keys ``'spy'``, ``'cannabis'``, ``'timing'`` — each a ``summary()``
    dict — plus ``'in_market_frac'`` (fraction of time holding cannabis in the timing arm)
    and ``'n_switches'``.
    """
    sig = timing_signal(prices, sma_n=sma_n, cannabis_col=cannabis_col)
    bt = run_backtest(prices, sig, cost_bps=cost_bps, cannabis_col=cannabis_col)

    in_mkt = float(bt["signal"].mean())

    r = log_returns(prices)
    spy_col = "spy" if "spy" in r.columns else [c for c in r.columns if c != cannabis_col][0]

    return {
        "spy": summary(r[spy_col]),
        "cannabis": summary(r[cannabis_col]),
        "timing": summary(bt["r_strategy"]),
        "in_market_frac": in_mkt,
        "n_switches": int((bt["signal"].diff().abs() > 0.5).sum()),
    }
