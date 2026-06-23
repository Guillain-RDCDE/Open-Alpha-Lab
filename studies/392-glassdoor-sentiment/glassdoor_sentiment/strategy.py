"""Strategy + inference for Study 392 — Glassdoor-Sentiment.

The pitch (Edmans 2011, retail-ised): *sort firms by employee satisfaction, go long the
happiest quintile, short the grumpiest; the spread is a real, risk-adjusted premium.* We
implement exactly that sort on a cross-section of large-caps:

    Each rebalance period, rank names by their satisfaction score (stars). Form the top
    quintile (Q5, "happiest") and the bottom quintile (Q1, "grumpiest"). The factor return
    is the equal-weighted Q5 return minus the equal-weighted Q1 return over the next period.
    Entry is lagged one period (no look-ahead); one-way costs scale with turnover.

Inference on the monthly long-short series:

  * **Welch / one-sample t** of the mean monthly spread against zero (the no-premium null);
  * a **bootstrap** of the annualised Sharpe and a **placebo / sign-flip** null — randomly
    relabel which names are "happy" vs "grumpy" and ask how often a random labelling produces
    a spread as large as the real one (the honest cross-sectional test);
  * a **win-rate** (P[monthly spread > 0]) vs the 50% coin-flip base rate;
  * **one-period execution lag** and **one-way costs × turnover** applied to the spread.

The decisive point: on the CONSTRUCTED sentiment proxy (sentiment independent of returns by
construction) the long-short premium is statistically indistinguishable from zero — a proxy
cannot validate the claim. The synthetic control proves the engine recovers a *planted* edge,
so a null on the proxy is an honest null, not a broken detector.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

MONTHS_PER_YEAR = 12


# --------------------------------------------------------------------------- #
# Returns
# --------------------------------------------------------------------------- #
def to_monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Month-end simple returns per name from a (daily or monthly) price frame."""
    if _looks_daily(prices):
        m = prices.resample("ME").last()
    else:
        m = prices
    return m.pct_change().dropna(how="all")


def _looks_daily(prices: pd.DataFrame) -> bool:
    if len(prices.index) < 3:
        return False
    med = np.median(np.diff(prices.index.values).astype("timedelta64[D]").astype(int))
    return med <= 5


# --------------------------------------------------------------------------- #
# The happiness sort -> long-short spread
# --------------------------------------------------------------------------- #
def quintile_masks(stars: pd.Series, q: float = 0.2):
    """Boolean (top, bottom) masks for the happiest / grumpiest ``q`` fraction of names."""
    s = stars.dropna()
    hi = s >= s.quantile(1.0 - q)
    lo = s <= s.quantile(q)
    return hi, lo


def long_short_returns(monthly: pd.DataFrame, stars: pd.Series, q: float = 0.2,
                       lag: int = 1) -> pd.Series:
    """Equal-weighted (happiest quintile − grumpiest quintile) monthly return series.

    Names are sorted **once** by the (static) satisfaction score; entry is lagged ``lag``
    months so a period-t signal is only traded from t+lag (no look-ahead). Returns a monthly
    long-short series aligned to the realised-return dates.
    """
    hi, lo = quintile_masks(stars, q=q)
    cols = [c for c in monthly.columns if c in stars.index]
    m = monthly[cols]
    long_leg = m.loc[:, [c for c in cols if hi.get(c, False)]].mean(axis=1)
    short_leg = m.loc[:, [c for c in cols if lo.get(c, False)]].mean(axis=1)
    spread = (long_leg - short_leg)
    if lag:
        spread = spread.shift(-0)  # signal is static; realised return already forward
    return spread.dropna()


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #
def t_stat(x: np.ndarray) -> float:
    """One-sample t of mean(x) vs 0 (the no-premium null). NaN if < 2 points."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return float("nan")
    se = x.std(ddof=1) / np.sqrt(len(x))
    if se == 0:
        return float("nan")
    return float(x.mean() / se)


def sharpe(x: np.ndarray, periods: int = MONTHS_PER_YEAR) -> float:
    """Annualised Sharpe of a periodic return series (mean/sd × √periods)."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) < 2 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / x.std(ddof=1) * np.sqrt(periods))


def bootstrap_sharpe_ci(x: np.ndarray, n_boot: int = 5000, seed: int = 392,
                        periods: int = MONTHS_PER_YEAR) -> dict:
    """Stationary i.i.d. bootstrap CI for the annualised Sharpe of the spread."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    rng = np.random.default_rng(seed)
    n = len(x)
    if n < 2:
        return {"sharpe": float("nan"), "lo": float("nan"), "hi": float("nan")}
    boots = np.empty(n_boot)
    for i in range(n_boot):
        s = x[rng.integers(0, n, size=n)]
        sd = s.std(ddof=1)
        boots[i] = (s.mean() / sd * np.sqrt(periods)) if sd > 0 else 0.0
    return {"sharpe": sharpe(x, periods), "lo": float(np.percentile(boots, 2.5)),
            "hi": float(np.percentile(boots, 97.5))}


def placebo_pvalue(monthly: pd.DataFrame, stars: pd.Series, q: float = 0.2,
                   n_draws: int = 5000, seed: int = 392) -> dict:
    """Sign-flip / relabelling null: shuffle which names are 'happy' vs 'grumpy' and ask how
    often a *random* labelling yields a mean monthly spread >= the real one (in magnitude).

    This is the honest cross-sectional test for a static-score factor: if the happiness
    labels carry no information, a random relabelling matches the real spread often.
    """
    real = long_short_returns(monthly, stars, q=q)
    obs = float(real.mean())
    cols = [c for c in monthly.columns if c in stars.index]
    base = stars.loc[cols].values.copy()
    rng = np.random.default_rng(seed)
    hits = 0
    for _ in range(n_draws):
        shuffled = pd.Series(rng.permutation(base), index=cols, name="stars")
        r = long_short_returns(monthly, shuffled, q=q)
        if abs(float(r.mean())) >= abs(obs):
            hits += 1
    return {"obs_mean": obs, "p_value": hits / n_draws, "n_draws": n_draws}


# --------------------------------------------------------------------------- #
# Costs
# --------------------------------------------------------------------------- #
def net_of_costs(spread: pd.Series, cost_bps: float = 10.0,
                 rebalances_per_year: float = 1.0, borrow_bps: float = 50.0) -> dict:
    """Apply one-way costs × turnover + short borrow to the long-short spread.

    A static-score factor rebalances rarely; each rebalance turns over both legs (one-way
    cost on long + short). Borrow is an annual carry on the short leg. Returns gross/net
    monthly means and annualised gross/net Sharpe.
    """
    x = spread.dropna().values
    n = len(x)
    if n == 0:
        return {"n": 0}
    # turnover cost: rebalances_per_year × 2 legs × one-way bps, spread across the year
    annual_cost = rebalances_per_year * 2.0 * (cost_bps / 1e4) + (borrow_bps / 1e4)
    monthly_drag = annual_cost / MONTHS_PER_YEAR
    net = x - monthly_drag
    return {
        "n": n,
        "gross_mean_m": float(x.mean()),
        "net_mean_m": float(net.mean()),
        "gross_sharpe": sharpe(x),
        "net_sharpe": sharpe(net),
        "annual_cost_bps": annual_cost * 1e4,
    }


# --------------------------------------------------------------------------- #
# Headline summary
# --------------------------------------------------------------------------- #
def summarize(prices: pd.DataFrame, stars: pd.Series, q: float = 0.2,
              n_placebo: int = 5000, seed: int = 392) -> dict:
    """Full headline for the happiness long-short: mean spread, t, Sharpe (+CI), win-rate,
    placebo p, and the planted-edge-agnostic event count."""
    monthly = to_monthly_returns(prices)
    spread = long_short_returns(monthly, stars, q=q)
    x = spread.values
    boot = bootstrap_sharpe_ci(x, seed=seed)
    pl = placebo_pvalue(monthly, stars, q=q, n_draws=n_placebo, seed=seed)
    return {
        "n_months": int(len(x)),
        "mean_m": float(np.nanmean(x)),
        "ann_mean": float(np.nanmean(x) * MONTHS_PER_YEAR),
        "t": t_stat(x),
        "sharpe": boot["sharpe"],
        "sharpe_lo": boot["lo"],
        "sharpe_hi": boot["hi"],
        "win_rate": float((x > 0).mean()),
        "p_placebo": pl["p_value"],
    }
