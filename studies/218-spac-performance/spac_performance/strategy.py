"""The strategy and its honest controls — Study 218 (SPAC-Performance).

The claim: SPACs (Special Purpose Acquisition Companies) offered a faster, cheaper
route to public markets and delivered superior post-merger returns vs the S&P 500,
backed by credible sponsors and a structurally sound underwriting model.

We test it on two complementary tapes:

1. **SPAK ETF vs SPY** — the Defiance Next Gen SPAC Derived ETF (ER ~0.45%) was
   purpose-built to track de-SPAC equity. It traded from 2020-10-01 to 2022-09-01
   and provides a clean, index-level view of the SPAC universe over its full life.

2. **Surviving de-SPAC basket vs SPY** — nine high-profile de-SPACs (LCID, RIVN,
   OPEN, PSFE, CLOV, SKLZ, DKNG, SPCE, QS) from 2021-11-10 onward. This basket is
   survivorship-biased upward (delisted names excluded) — so any underperformance
   here understates the true damage.

Both use CAPM Jensen alpha (OLS intercept) with Newey-West HAC t-statistics as the
inference-bar statistic, and total-return adjusted prices throughout.

Structural-dilution note: classic SPAC mechanics give sponsors 20% "promote" warrants
at no cost (the "promote"), creating a ~20% dilution drag for non-sponsor shareholders
at merger. This mechanical headwind is baked into the return series, not estimated
separately.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core return computation
# ---------------------------------------------------------------------------
def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute simple daily returns from a price DataFrame.

    Returns a DataFrame of the same shape, first row NaN dropped.
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

    Returns alpha (daily bps and annualised pct), beta, R-squared, and the HAC
    t-stat on the intercept via the Newey-West sandwich covariance. The OLS estimator
    uses the full sample (static beta). A rolling beta would be stricter but the
    static choice is conservative here — it does not bias against finding underperformance.
    """
    mask = r_fund.notna() & r_market.notna()
    rf = r_fund[mask].to_numpy(dtype=float)
    rm = r_market[mask].to_numpy(dtype=float)

    _coeffs = np.polyfit(rm, rf, 1)
    beta, alpha = float(_coeffs[0]), float(_coeffs[1])

    resid = rf - (beta * rm + alpha)
    ss_tot = float(np.sum((rf - rf.mean()) ** 2))
    r_squared = 1.0 - float(np.sum(resid ** 2)) / ss_tot if ss_tot > 0 else float("nan")

    alpha_ann = (1.0 + alpha) ** 252 - 1.0

    _, tstat_alpha_hac = _hac_ols_tstat_alpha(rf, rm)

    alpha_series = pd.Series(rf - (beta * rm + alpha), index=r_fund[mask].index,
                             name="alpha_residual")

    return {
        "alpha_daily_bps": float(alpha * 1e4),
        "alpha_ann_pct": float(alpha_ann * 100),
        "beta": float(beta),
        "r_squared": float(r_squared),
        "alpha_series": alpha_series,
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
    """Full headline stats for a SPAC proxy vs SPY — the study's spine.

    Keys returned
    -------------
    cagr_fund_pct, cagr_market_pct   annualised compound growth (%)
    sharpe_fund, sharpe_market        annualised Sharpe (mean/std * sqrt(252))
    alpha_daily_bps                   Jensen alpha in basis points/day
    alpha_ann_pct                     annualised Jensen alpha (%)
    beta                              OLS beta
    tstat_alpha_hac                   Newey-West t-stat on alpha (the critical number)
    excess_mean_bps                   mean daily excess return in bps
    tstat_excess_hac                  HAC t on raw excess (fund - market)
    max_dd_fund_pct, max_dd_market_pct  maximum drawdown (negative, %)
    n                                 number of trading days
    n_years                           sample length in years
    """
    r_f = r_fund.dropna()
    r_m = r_market.dropna()
    common = r_f.index.intersection(r_m.index)
    r_f = r_f.loc[common]
    r_m = r_m.loc[common]
    n = len(r_f)
    n_years = n / 252.0

    cagr_f = float((1.0 + r_f).prod() ** (1.0 / n_years) - 1.0)
    cagr_m = float((1.0 + r_m).prod() ** (1.0 / n_years) - 1.0)

    sharpe_f = float(r_f.mean() / r_f.std(ddof=1) * np.sqrt(252))
    sharpe_m = float(r_m.mean() / r_m.std(ddof=1) * np.sqrt(252))

    decomp = capm_decompose(r_f, r_m)
    tstat_alpha_hac = decomp["tstat_alpha_hac"]

    excess = r_f.values - r_m.values
    excess_mean_bps = float(excess.mean() * 1e4)
    tstat_excess_hac = _hac_tstat(excess)

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
        "alpha_series": decomp["alpha_series"],
    }


# ---------------------------------------------------------------------------
# Synthetic experiment — recover planted alpha?
# ---------------------------------------------------------------------------
def detect_alpha(prices: pd.DataFrame) -> dict:
    """Run summarize() on the 'spac' vs 'spy' columns of a price DataFrame.

    Entry point for the synthetic positive/negative control: given a synthetic tape,
    does the analysis correctly read out the planted alpha_ann?
    """
    rets = daily_returns(prices)
    return summarize(rets["spac"], rets["spy"])


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


def _hac_ols_tstat_alpha(rf: np.ndarray, rm: np.ndarray) -> tuple[float, float]:
    """Newey-West HAC t-stat on the OLS intercept (Jensen alpha).

    Returns (alpha_daily, t_alpha_hac). Uses the HAC sandwich covariance matrix
    on the [alpha, beta] vector.
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
