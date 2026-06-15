"""The analysis engine for Study 208 (Gold-Miners).

Three tests, one honest story:

1. **Beta and alpha** — OLS of daily GDX log-returns on GLD log-returns. The point
   estimate (beta ~1.5-2x) is the "leveraged gold" claim; the intercept (alpha) captures
   the operational drag (mining costs, hedging, management fees). We report alpha per year
   and the HAC t-stat on both.

2. **Leverage asymmetry** — split returns into gold-up days and gold-down days and
   estimate the upside vs downside beta separately. The widely-cited "gives more downside
   than upside" claim implies ``beta_down > beta_up``. We test this with a Newey-West
   t-stat on the difference in residuals and report the upside/downside beta estimates.

3. **Miners-vs-bullion timing rule** — buy GDX when GDX/GLD ratio > its 200-day SMA,
   hold GLD otherwise (a relative-strength momentum rule). The honest control is simply
   holding GLD throughout. If the timing rule adds value beyond GLD's own return, the
   Sharpe should improve significantly; if not, it is a Mirage.

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
# Test 1: OLS beta / alpha of GDX on GLD
# ---------------------------------------------------------------------------
def beta_alpha(prices: pd.DataFrame) -> dict:
    """OLS regression of GDX log-returns on GLD log-returns.

    Estimates:
      r_GDX = alpha + beta * r_GLD + eps

    Returns the beta, annualised alpha, HAC t-stats for both via the Newey-West
    sandwich estimator, R-squared, residual volatility (idio vol).

    The sandwich HAC SE for each coefficient k is:
        SE_k = sqrt( [(X'X)^{-1} S (X'X)^{-1}]_{kk} )
    where S = (1/n) sum_t u_t u_t' is the Newey-West long-run covariance of the
    score vectors u_t = X_t * eps_t.
    """
    r = log_returns(prices)
    y = r["gdx"].to_numpy()
    x = r["gld"].to_numpy()
    n = len(y)

    # OLS design matrix: [x, 1] -> [beta, alpha]
    X = np.column_stack([x, np.ones(n)])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    beta, alpha = float(coef[0]), float(coef[1])
    resid = y - X @ coef

    # Newey-West sandwich: HAC covariance of score U = X * resid (n x 2)
    U = X * resid[:, None]  # each row is u_t = X_t * eps_t
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
    gld_vol_ann = float(r["gld"].std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))
    gdx_vol_ann = float(r["gdx"].std(ddof=1) * np.sqrt(TRADING_DAYS_PER_YEAR))

    return {
        "n": int(n),
        "beta": beta,
        "alpha_daily_bps": float(alpha * 1e4),
        "alpha_ann_pct": float(alpha_ann * 100),
        "t_beta": t_beta,
        "t_alpha": t_alpha,
        "r_squared": r2,
        "idio_vol_ann_pct": float(idio_vol_ann * 100),
        "gld_vol_ann_pct": float(gld_vol_ann * 100),
        "gdx_vol_ann_pct": float(gdx_vol_ann * 100),
        "implied_leverage": float(gdx_vol_ann / gld_vol_ann) if gld_vol_ann > 0 else float("nan"),
    }


# ---------------------------------------------------------------------------
# Test 2: Asymmetric (upside vs downside) beta
# ---------------------------------------------------------------------------
def asymmetric_beta(prices: pd.DataFrame) -> dict:
    """Estimate GDX beta to GLD separately on gold-up vs gold-down days.

    The popular claim predicts ``beta_dn > beta_up`` (more downside amplification).
    We fit a dual-slope model with a shared intercept:
        r_GDX = alpha + beta_up * r_GLD * I(r_GLD >= 0) + beta_dn * r_GLD * I(r_GLD < 0) + eps
    and compute the HAC (Newey-West sandwich) t-stat on the linear contrast
    (beta_dn - beta_up) — the correct test for the asymmetry claim.

    Positive ``t_asymmetry`` supports the popular claim (more downside beta);
    negative ``t_asymmetry`` means miners have *more* upside amplification (opposite claim).
    """
    r = log_returns(prices)
    r_gld = r["gld"].to_numpy()
    r_gdx = r["gdx"].to_numpy()
    n = len(r_gld)

    up = (r_gld >= 0).astype(float)
    dn = 1.0 - up

    # Dual-slope OLS with shared intercept: [x*up, x*dn, 1]
    X = np.column_stack([r_gld * up, r_gld * dn, np.ones(n)])
    coef, *_ = np.linalg.lstsq(X, r_gdx, rcond=None)
    beta_up, beta_dn, alpha = float(coef[0]), float(coef[1]), float(coef[2])
    resid = r_gdx - X @ coef

    # Newey-West HAC sandwich for the full 3-coefficient model
    U = X * resid[:, None]
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    S = U.T @ U / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        Gk = U[k:].T @ U[:-k] / n
        S += w * (Gk + Gk.T)
    XtX_inv = np.linalg.inv(X.T @ X / n)
    cov_hat = XtX_inv @ S @ XtX_inv / n

    # Contrast: c = [-1, 1, 0] tests beta_dn - beta_up
    c = np.array([-1.0, 1.0, 0.0])
    var_diff = float(c @ cov_hat @ c)
    se_diff = np.sqrt(max(var_diff, 0.0))
    beta_diff = beta_dn - beta_up
    t_asymmetry = float(beta_diff / se_diff) if se_diff > 0 else float("nan")

    return {
        "n_up": int(up.sum()),
        "n_dn": int(dn.sum()),
        "beta_up": beta_up,
        "beta_dn": beta_dn,
        "beta_diff": beta_diff,
        "asymmetry_ratio": float(beta_dn / beta_up) if beta_up > 0 else float("nan"),
        "t_asymmetry": t_asymmetry,
        "alpha_ann_pct": float(alpha * TRADING_DAYS_PER_YEAR * 100),
    }


# ---------------------------------------------------------------------------
# Test 3: Miners-vs-bullion timing rule
# ---------------------------------------------------------------------------
def timing_signal(prices: pd.DataFrame, sma_n: int = 200) -> pd.Series:
    """GDX/GLD relative-strength timing signal.

    Signal = 1 (hold GDX) when GDX/GLD ratio is *above* its ``sma_n``-day SMA,
    = 0 (hold GLD) when below. The ratio above the SMA indicates miners are
    outperforming bullion — the timing rule bets on continuation.

    No look-ahead: the signal formed at day *t* is applied to the return at day *t+1*
    (the series is shifted by one before multiplication in ``run_backtest``).
    """
    ratio = prices["gdx"] / prices["gld"]
    sma = ratio.rolling(sma_n, min_periods=sma_n).mean()
    above = (ratio > sma).astype(float)
    return above.shift(1).rename("signal")


def run_backtest(
    prices: pd.DataFrame,
    signal: pd.Series,
    cost_bps: float = 5.0,
) -> pd.DataFrame:
    """Apply a GDX/GLD rotation signal to daily returns.

    When ``signal = 1`` (hold GDX), the day's return is GDX's log-return.
    When ``signal = 0`` (hold GLD), the day's return is GLD's log-return.
    A round-trip transaction cost is charged on each switch (one-way ``cost_bps``
    per leg change).

    Returns a DataFrame with columns:
        ``r_gld``      — daily log-return of holding GLD (the baseline).
        ``r_gdx``      — daily log-return of holding GDX.
        ``signal``     — the applied signal (0 = GLD, 1 = GDX).
        ``r_strategy`` — signal * r_gdx + (1-signal) * r_gld − switch_cost.
    """
    r = log_returns(prices)
    r_gld = r["gld"]
    r_gdx = r["gdx"]

    sig = signal.reindex(r.index).fillna(0.0)
    switches = sig.diff().abs().fillna(0.0)
    cost_per_switch = cost_bps * 1e-4

    r_strat = (sig * r_gdx + (1.0 - sig) * r_gld - switches * cost_per_switch).rename("r_strategy")

    r_gld = r_gld.rename("r_gld")
    r_gdx = r_gdx.rename("r_gdx")
    return pd.concat([r_gld, r_gdx, sig.rename("signal"), r_strat], axis=1).dropna()


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


def compare_strategies(
    prices: pd.DataFrame,
    sma_n: int = 200,
    cost_bps: float = 5.0,
) -> dict[str, dict]:
    """Run GLD buy-and-hold, GDX buy-and-hold, and the GDX/GLD timing rule.

    Returns a dict with keys ``'gld'``, ``'gdx'``, ``'timing'`` — each a ``summary()``
    dict — plus ``'in_market_frac'`` (fraction of time holding GDX in the timing arm).
    """
    sig = timing_signal(prices, sma_n=sma_n)
    bt = run_backtest(prices, sig, cost_bps=cost_bps)

    in_mkt = float(bt["signal"].mean())

    r = log_returns(prices)

    return {
        "gld": summary(r["gld"]),
        "gdx": summary(r["gdx"]),
        "timing": summary(bt["r_strategy"]),
        "in_market_frac": in_mkt,
        "n_switches": int((bt["signal"].diff().abs() > 0.5).sum()),
    }
