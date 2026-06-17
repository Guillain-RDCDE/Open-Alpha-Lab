"""Strategy and analysis layer for Study 273 (Lego-Returns).

The claim (Dobrynskaya & Kishilova 2022): retired LEGO sets earn an
equity-beating, low-beta return on the secondary market -- "the toy of smart
investors."  We implement the honest comparison:

  - **Excess return** of the Lego leg over the S&P 500 each year, with a
    **HAC (Newey-West) t-stat** on the mean excess -- the bar for a REAL signal.
  - **Net-of-cost** Lego returns: a one-way resale friction (eBay/BrickLink fees
    of ~12% by default) is charged on the Lego leg, because realising a
    collectible's price means paying a marketplace its cut.  Gross vs net are
    both reported.
  - **CAPM-style beta** of Lego on the market, to test the "low correlation,
    diversifier" half of the claim.
  - A **paired sign test** (how often does Lego beat the market in a year?) with
    the correct binomial null.

The reckoning: even if the gross index outpaces the price-only S&P, (a) the
equity total-return benchmark (price + ~2%/yr dividends) closes much of the gap,
(b) resale frictions are large and recurring, and (c) the index is survivorship-
/selection-biased upward.  The HAC t-stat on net excess return is the decisive
number.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Cost model
# ---------------------------------------------------------------------------
def net_lego_return(
    lego_return: pd.Series | np.ndarray,
    one_way_cost: float = 0.12,
    holding_years: float = 1.0,
) -> np.ndarray:
    """Charge a one-way resale friction on the Lego leg.

    A collector realises a set's secondary-market price only by selling through a
    marketplace that takes a cut (eBay ~13%, BrickLink ~3% + shipping/photos/time).
    We amortise a single ``one_way_cost`` round of friction over ``holding_years``
    of holding -- the longer you hold, the smaller the annualised drag.  Returns
    the cost-adjusted annual return array.
    """
    lr = np.asarray(lego_return, dtype=float)
    annual_drag = one_way_cost / max(holding_years, 1e-9)
    return lr - annual_drag


# ---------------------------------------------------------------------------
# HAC (Newey-West) t-stat on a mean
# ---------------------------------------------------------------------------
def hac_tstat(x: np.ndarray, lags: int | None = None) -> tuple[float, float]:
    """Newey-West HAC t-stat for H0: mean(x) = 0.

    Returns ``(mean, t_stat)``.  ``lags`` defaults to floor(4*(n/100)^(2/9))
    (the standard Newey-West rule of thumb); annual data has little
    autocorrelation so lags is typically 1.
    """
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    n = len(x)
    if n < 3:
        return float(np.mean(x)) if n else float("nan"), float("nan")
    if lags is None:
        lags = int(np.floor(4 * (n / 100.0) ** (2.0 / 9.0)))
    mu = float(np.mean(x))
    e = x - mu
    gamma0 = float(np.dot(e, e) / n)
    var = gamma0
    for k in range(1, lags + 1):
        w = 1.0 - k / (lags + 1.0)
        cov = float(np.dot(e[k:], e[:-k]) / n)
        var += 2.0 * w * cov
    se = np.sqrt(max(var, 0.0) / n)
    t = mu / se if se > 0 else float("nan")
    return mu, float(t)


# ---------------------------------------------------------------------------
# CAPM-style beta
# ---------------------------------------------------------------------------
def beta_to_market(lego_return: np.ndarray, market_return: np.ndarray) -> dict:
    """OLS slope (beta), alpha and correlation of Lego on the market."""
    lr = np.asarray(lego_return, dtype=float)
    mr = np.asarray(market_return, dtype=float)
    mask = ~(np.isnan(lr) | np.isnan(mr))
    lr, mr = lr[mask], mr[mask]
    if len(lr) < 3:
        return {"beta": float("nan"), "alpha": float("nan"), "corr": float("nan")}
    slope, intercept, r, _, _ = scipy_stats.linregress(mr, lr)
    return {"beta": float(slope), "alpha": float(intercept), "corr": float(r)}


# ---------------------------------------------------------------------------
# The headline comparison
# ---------------------------------------------------------------------------
def compare(
    df: pd.DataFrame,
    one_way_cost: float = 0.12,
    holding_years: float = 5.0,
    seed: int = 273,
) -> dict:
    """Full Lego-vs-market comparison on a merged annual frame.

    ``df`` needs columns ``lego_return`` and ``market_return``.  Returns a flat
    dict of headline numbers: geometric annual returns (gross/net Lego, market),
    mean excess return, HAC t-stat (gross and net), beta/alpha/correlation, and
    the paired sign test (how often Lego beats the market, with binomial p).
    """
    lr = df["lego_return"].to_numpy(dtype=float)
    mr = df["market_return"].to_numpy(dtype=float)
    n = len(lr)

    lr_net = net_lego_return(lr, one_way_cost=one_way_cost, holding_years=holding_years)

    def _geo(x):
        return float(np.prod(1.0 + x) ** (1.0 / len(x)) - 1.0)

    lego_cagr = _geo(lr)
    lego_net_cagr = _geo(lr_net)
    mkt_cagr = _geo(mr)

    excess_gross = lr - mr
    excess_net = lr_net - mr
    mu_g, t_g = hac_tstat(excess_gross)
    mu_n, t_n = hac_tstat(excess_net)

    b = beta_to_market(lr, mr)

    # Paired sign test: in how many years does Lego beat the market?
    wins = int(np.sum(lr > mr))
    sign_p = float(scipy_stats.binomtest(wins, n, 0.5, alternative="greater").pvalue)

    return {
        "n": int(n),
        "lego_cagr": round(lego_cagr, 4),
        "lego_net_cagr": round(lego_net_cagr, 4),
        "market_cagr": round(mkt_cagr, 4),
        "mean_excess_gross": round(float(mu_g), 4),
        "mean_excess_net": round(float(mu_n), 4),
        "hac_t_gross": round(float(t_g), 3),
        "hac_t_net": round(float(t_n), 3),
        "beta": round(b["beta"], 3),
        "alpha": round(b["alpha"], 4),
        "corr": round(b["corr"], 3),
        "lego_win_years": wins,
        "lego_win_rate": round(wins / n, 4),
        "sign_test_p": round(sign_p, 4),
        "one_way_cost": one_way_cost,
        "holding_years": holding_years,
    }


def summarize(df: pd.DataFrame, one_way_cost: float = 0.12,
              holding_years: float = 5.0, seed: int = 273) -> dict:
    """Alias kept for parity with sibling studies; see :func:`compare`."""
    return compare(df, one_way_cost=one_way_cost,
                   holding_years=holding_years, seed=seed)
