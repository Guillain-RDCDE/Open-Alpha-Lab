"""The strategy and its honest controls — Study 139 (AI-Powered-ETF).

The claim: AIEQ (the IBM Watson / EquBot AI-Powered Equity ETF, 0.75% expense ratio)
picks stocks better than the market, delivering alpha net of fees — "harnessing the
power of AI to consistently outperform".

We test it against two controls that are fair by construction:

1. **Buy-and-hold SPY** — the cheapest, simplest index fund (0.0945% ER). If AIEQ
   cannot beat this, the AI wrapper adds nothing.
2. **CAPM alpha** — the fund's return unexplained by market beta exposure (Jensen's
   alpha). A positive, statistically significant alpha is the minimum required to claim
   "AI adds value". We use Newey-West HAC inference to account for autocorrelated
   returns over the ~8.5-year window.

No look-ahead: returns are formed on close-to-close price changes, computed daily,
with no forward information. The beta estimate uses the full window (a static linear
model), which is slightly optimistic for AIEQ — the real beta might be estimated
rolling in a stricter protocol, but here the bias goes against our null (it helps AIEQ).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core return computation
# ---------------------------------------------------------------------------
def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple (arithmetic) daily returns from a price DataFrame.

    Returns a DataFrame of the same shape, shifted by 1 (first row is NaN, dropped).
    """
    return prices.pct_change().dropna()


# ---------------------------------------------------------------------------
# CAPM / alpha decomposition
# ---------------------------------------------------------------------------
def capm_decompose(
    r_fund: pd.Series,
    r_market: pd.Series,
) -> dict:
    """OLS CAPM decomposition: r_fund = alpha + beta * r_market + eps.

    Returns alpha (daily and annualised), beta, R-squared, and the alpha residual
    series for downstream HAC inference. The OLS estimator here is the standard
    full-sample static beta; no rolling or Bayesian shrinkage is applied.

    Parameters
    ----------
    r_fund : daily return series for the fund under study.
    r_market : daily return series for the benchmark (SPY).

    The bias goes against the null (a slightly wrong beta attribution inflates the
    residual variance, widening confidence bands around alpha — a conservative choice).
    """
    mask = r_fund.notna() & r_market.notna()
    rf = r_fund[mask].to_numpy(dtype=float)
    rm = r_market[mask].to_numpy(dtype=float)

    # OLS with intercept: rf = beta * rm + alpha
    _coeffs = np.polyfit(rm, rf, 1)
    beta, alpha = float(_coeffs[0]), float(_coeffs[1])
    alpha_series = pd.Series(rf - (beta * rm + alpha), index=r_fund[mask].index,
                             name="alpha_residual")

    # R-squared
    resid = rf - (beta * rm + alpha)
    ss_tot = float(np.sum((rf - rf.mean()) ** 2))
    r_squared = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else float("nan")

    # Annualise
    alpha_ann = (1.0 + alpha) ** 252 - 1.0
    beta_contrib_daily = beta * rm.mean()

    # HAC t-stat on OLS alpha via sandwich estimator
    _, tstat_alpha_hac = _hac_ols_tstat_alpha(r_fund.to_numpy(dtype=float), r_market.to_numpy(dtype=float))
    return {
        "alpha_daily_bps": float(alpha * 1e4),
        "alpha_ann_pct": float(alpha_ann * 100),
        "beta": float(beta),
        "r_squared": float(r_squared),
        "alpha_series": alpha_series,
        "beta_contrib_daily_bps": float(beta_contrib_daily * 1e4),
        "tstat_alpha_hac": tstat_alpha_hac,
        "n": int(mask.sum()),
    }


# ---------------------------------------------------------------------------
# Headline statistics with HAC inference
# ---------------------------------------------------------------------------
def summarize(
    r_fund: pd.Series,
    r_market: pd.Series,
) -> dict:
    """Full headline stats for AIEQ vs market benchmark — the study's spine.

    Returns per-series annualised figures plus the CAPM alpha with a Newey-West
    HAC t-stat (the inference-bar number that decides Signal). The function is
    deliberately self-contained (no quantlab import) so the synthetic demo runs
    offline without the repo root on the path.

    Keys returned
    -------------
    cagr_fund, cagr_market              annualised compound growth (pct)
    sharpe_fund, sharpe_market          annualised Sharpe (mean/std*sqrt(252))
    alpha_daily_bps                     Jensen alpha in basis points/day
    alpha_ann_pct                       annualised Jensen alpha (%)
    beta                                OLS beta
    tstat_alpha_hac                     Newey-West t-stat on alpha (the critical number)
    excess_mean_bps                     mean daily excess return (AIEQ - SPY) in bps
    tstat_excess_hac                    HAC t on raw excess (AIEQ - SPY)
    max_dd_fund, max_dd_market          maximum drawdown (negative fraction)
    n                                   number of trading days
    n_years                             sample length in years
    """
    r_f = r_fund.dropna()
    r_m = r_market.dropna()
    common = r_f.index.intersection(r_m.index)
    r_f = r_f.loc[common]
    r_m = r_m.loc[common]
    n = len(r_f)
    n_years = n / 252.0

    # CAGR (annualised compound growth from daily returns)
    cagr_f = float((1.0 + r_f).prod() ** (1.0 / n_years) - 1.0)
    cagr_m = float((1.0 + r_m).prod() ** (1.0 / n_years) - 1.0)

    # Annualised Sharpe (no risk-free rate, conservative)
    sharpe_f = float(r_f.mean() / r_f.std(ddof=1) * np.sqrt(252))
    sharpe_m = float(r_m.mean() / r_m.std(ddof=1) * np.sqrt(252))

    # CAPM decomposition
    decomp = capm_decompose(r_f, r_m)
    alpha_series = decomp["alpha_series"]

    # HAC t-stat on Jensen alpha via OLS sandwich (the residuals have mean 0 by construction
    # so we cannot simply call _hac_tstat on them -- we need the HAC SE of the intercept)
    tstat_alpha_hac = decomp["tstat_alpha_hac"]

    # Raw excess return AIEQ - SPY
    excess = r_f.values - r_m.values
    excess_mean_bps = float(excess.mean() * 1e4)
    tstat_excess_hac = _hac_tstat(excess)

    # Max drawdown
    max_dd_f = _max_drawdown(r_f)
    max_dd_m = _max_drawdown(r_m)

    return {
        "n": n,
        "n_years": float(n_years),
        "cagr_fund_pct": float(cagr_f * 100),
        "cagr_market_pct": float(cagr_m * 100),
        "sharpe_fund": sharpe_f,
        "sharpe_market": sharpe_m,
        "alpha_daily_bps": decomp["alpha_daily_bps"],
        "alpha_ann_pct": decomp["alpha_ann_pct"],
        "beta": decomp["beta"],
        "r_squared": decomp["r_squared"],
        "tstat_alpha_hac": tstat_alpha_hac,
        "excess_mean_bps": excess_mean_bps,
        "tstat_excess_hac": tstat_excess_hac,
        "max_dd_fund_pct": float(max_dd_f * 100),
        "max_dd_market_pct": float(max_dd_m * 100),
        "alpha_series": alpha_series,
    }


# ---------------------------------------------------------------------------
# Rolling analysis — does the gap open up over time?
# ---------------------------------------------------------------------------
def rolling_alpha(
    r_fund: pd.Series,
    r_market: pd.Series,
    window: int = 252,
) -> pd.Series:
    """Rolling 12-month Jensen alpha (annualised, in pct).

    A negative and worsening rolling alpha would show the AI fund deteriorating
    relative to the market over time. A flat-then-negative pattern would suggest
    early mean reversion or diminishing returns to the AI model.
    """
    common = r_fund.index.intersection(r_market.index)
    rf = r_fund.loc[common]
    rm = r_market.loc[common]

    alphas = []
    for end in range(window, len(rf) + 1):
        rf_w = rf.iloc[end - window: end].to_numpy()
        rm_w = rm.iloc[end - window: end].to_numpy()
        _c = np.polyfit(rm_w, rf_w, 1); beta_w, alpha_w = float(_c[0]), float(_c[1])
        alpha_ann = float((1.0 + alpha_w) ** 252 - 1.0) * 100
        alphas.append(alpha_ann)

    idx = rf.index[window - 1:]
    return pd.Series(alphas, index=idx, name="rolling_alpha_ann_pct")


def rolling_outperformance(
    r_fund: pd.Series,
    r_market: pd.Series,
    window: int = 252,
) -> pd.Series:
    """Rolling 12-month cumulative excess return (AIEQ - SPY in pct).

    Positive means AIEQ beat SPY over that window; negative means it lagged.
    """
    common = r_fund.index.intersection(r_market.index)
    rf = r_fund.loc[common]
    rm = r_market.loc[common]
    excess = rf.values - rm.values
    ser = pd.Series(excess, index=common)
    return ser.rolling(window).sum().dropna() * 100  # in pct


# ---------------------------------------------------------------------------
# Synthetic experiment — recover planted alpha?
# ---------------------------------------------------------------------------
def detect_alpha(prices: pd.DataFrame) -> dict:
    """Run summarize() on the 'aieq' vs 'spy' columns of a price DataFrame.

    This is the entry point for the synthetic positive/negative control: given a
    synthetic tape (from data.synthetic_daily), does the analysis correctly read out
    the planted alpha_ann?
    """
    rets = daily_returns(prices)
    return summarize(rets["aieq"], rets["spy"])


# ---------------------------------------------------------------------------
# Private helpers (no quantlab dependency for offline use)
# ---------------------------------------------------------------------------
def _hac_tstat(arr: np.ndarray) -> float:
    """Newey-West HAC t-statistic on the sample mean.

    Uses the rule-of-thumb lag count ``floor(4*(n/100)^(2/9))``.
    """
    arr = arr[np.isfinite(arr)]
    n = arr.size
    if n < 6:
        return float("nan")
    mu = arr.mean()
    e = arr - mu
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = float(e @ e) / n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(e[k:] @ e[:-k]) / n
    se = np.sqrt(max(lrv, 0.0) / n)
    return float(mu / se) if se > 0 else float("nan")




def _hac_ols_tstat_alpha(rf, rm):
    """Newey-West HAC t-stat on the OLS intercept (Jensen alpha) from rf = alpha + beta*rm.

    Returns (alpha_daily, t_alpha_hac). Uses the HAC sandwich covariance matrix on
    the [alpha, beta] vector, not just the OLS residual mean (which is 0 by construction).
    """
    mask = np.isfinite(rf) & np.isfinite(rm)
    rf2, rm2 = rf[mask], rm[mask]
    n = len(rf2)
    if n < 10:
        return float("nan"), float("nan")
    X = np.column_stack([np.ones(n), rm2])
    coeffs = np.linalg.lstsq(X, rf2, rcond=None)[0]
    alpha_hat = float(coeffs[0])
    resid = rf2 - X @ coeffs
    lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    S = np.zeros((2, 2))
    for t in range(n):
        S += resid[t] ** 2 * np.outer(X[t], X[t])
    S /= n
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        Gk = np.zeros((2, 2))
        for t in range(k, n):
            Gk += resid[t] * resid[t - k] * np.outer(X[t], X[t - k])
        Gk /= n
        S += w * (Gk + Gk.T)
    XTX_inv = np.linalg.inv(X.T @ X / n)
    V = XTX_inv @ S @ XTX_inv / n
    se_alpha = float(np.sqrt(max(V[0, 0], 0.0)))
    return alpha_hat, float(alpha_hat / se_alpha) if se_alpha > 0 else float("nan")

def _max_drawdown(returns: pd.Series) -> float:
    """Maximum peak-to-trough drawdown from a return series (returns a negative number)."""
    cum = (1.0 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    return float(dd.min())
